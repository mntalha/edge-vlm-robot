"""Back-of-the-envelope memory and speed estimates for VLMs on edge GPUs.

Two rules of thumb drive almost every edge-deployment decision:

1. **Memory**: weights + KV cache + activations + runtime overhead must fit in
   the *usable* unified memory (the OS / ROS / camera stack need some too).
2. **Speed**: autoregressive decoding is *memory-bandwidth bound*. Each new
   token has to stream (roughly) all LM weights from DRAM once, so
   ``tokens/s <= bandwidth / bytes_of_LM_weights``. Quantization therefore
   speeds up decoding almost linearly, not just saves memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .hardware import HardwareProfile

BYTES_PER_PARAM: Dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    # 4-bit weights + one fp16 scale per group of 128 -> 4.125 bits/param
    "int4": 0.5 + 2.0 / 128,
}

GB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    """Just enough architecture info to budget a VLM."""

    name: str
    vision_params: float
    lm_params: float
    lm_layers: int
    lm_kv_heads: int
    lm_head_dim: int
    image_tokens: int  # visual tokens fed to the LM per image (or per tile)

    @property
    def total_params(self) -> float:
        return self.vision_params + self.lm_params


# Approximate public numbers - good enough for budgeting, not for papers.
KNOWN_MODELS: Dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec("SmolVLM-256M", 93e6, 135e6, 30, 3, 64, 64),
        ModelSpec("SmolVLM-500M", 93e6, 362e6, 32, 5, 64, 64),
        ModelSpec("SmolVLM2-2.2B", 428e6, 1.71e9, 24, 32, 64, 81),
        ModelSpec("Qwen2.5-VL-3B", 670e6, 3.09e9, 36, 2, 128, 256),
        ModelSpec("Qwen2.5-VL-7B", 670e6, 7.6e9, 28, 4, 128, 256),
    ]
}


def weights_bytes(n_params: float, precision: str = "fp16") -> float:
    return n_params * BYTES_PER_PARAM[precision]


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    precision: str = "fp16",
) -> float:
    """K and V tensors for every layer: 2 * L * kv_heads * head_dim * T * B."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * BYTES_PER_PARAM[precision]


def vision_token_count(image_size: int, patch_size: int, pixel_shuffle: int = 1) -> int:
    patches = (image_size // patch_size) ** 2
    return patches // (pixel_shuffle**2)


def decode_tokens_per_sec_upper_bound(lm_weight_bytes: float, bandwidth_gbs: float) -> float:
    """Roofline upper bound for batch-1 decoding (real numbers land ~50-80% of this)."""
    return bandwidth_gbs * 1e9 / lm_weight_bytes


@dataclass
class Budget:
    model: str
    hardware: str
    precision: str
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_gb: float
    usable_gb: float
    fits: bool
    max_decode_tok_s: float

    def as_row(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "precision": self.precision,
            "weights_GB": round(self.weights_gb, 2),
            "kv_GB": round(self.kv_cache_gb, 3),
            "total_GB": round(self.total_gb, 2),
            "usable_GB": round(self.usable_gb, 2),
            "fits": "yes" if self.fits else "NO",
            "max_tok/s": round(self.max_decode_tok_s, 1),
        }


def plan_budget(
    spec: ModelSpec,
    hw: HardwareProfile,
    precision: str = "fp16",
    vision_precision: Optional[str] = None,
    text_tokens: int = 256,
    n_images: int = 1,
    kv_precision: str = "fp16",
    runtime_overhead: float = 0.15,
) -> Budget:
    """Estimate whether ``spec`` fits on ``hw`` and how fast it can decode.

    ``runtime_overhead`` covers activations, workspace and allocator slack as
    a fraction of weights (TensorRT/PyTorch both need scratch memory).
    """
    vision_precision = vision_precision or ("fp16" if precision in ("int4", "int8", "fp8") else precision)
    lm_w = weights_bytes(spec.lm_params, precision)
    vis_w = weights_bytes(spec.vision_params, vision_precision)
    seq = text_tokens + spec.image_tokens * n_images
    kv = kv_cache_bytes(spec.lm_layers, spec.lm_kv_heads, spec.lm_head_dim, seq, precision=kv_precision)
    overhead = runtime_overhead * (lm_w + vis_w)
    total = lm_w + vis_w + kv + overhead
    return Budget(
        model=spec.name,
        hardware=hw.key,
        precision=precision,
        weights_gb=(lm_w + vis_w) / GB,
        kv_cache_gb=kv / GB,
        overhead_gb=overhead / GB,
        total_gb=total / GB,
        usable_gb=hw.usable_gb,
        fits=total / GB <= hw.usable_gb,
        max_decode_tok_s=decode_tokens_per_sec_upper_bound(lm_w, hw.memory_bandwidth_gbs),
    )


def format_table(rows: Iterable[Dict[str, object]]) -> str:
    """Tiny dependency-free table printer for notebooks/CLI."""
    rows = list(rows)
    if not rows:
        return ""
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols}
    line = " | ".join(str(c).ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = [" | ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join([line, sep, *body])


def budget_rows(
    specs: Iterable[ModelSpec], profiles: Iterable[HardwareProfile], precisions: Iterable[str]
) -> List[Dict[str, object]]:
    precisions = list(precisions)
    profiles = list(profiles)
    return [plan_budget(s, hw, p).as_row() for s in specs for hw in profiles for p in precisions]
