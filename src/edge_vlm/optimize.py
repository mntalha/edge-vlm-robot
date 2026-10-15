"""Model compression helpers: precision casting, weight-only quantization, benchmarking.

Weight-only quantization (W8A16 / W4A16) is the workhorse for LLMs on Jetson:
weights are stored in int8/int4 with per-channel or per-group scales and
de-quantized on the fly inside the matmul kernel. Activations stay in fp16.
Because decoding is memory-bandwidth bound, halving the weight bytes roughly
doubles tokens/s. TensorRT Edge-LLM, llama.cpp (GGUF Q4/Q8), AWQ and GPTQ all
build on this idea - here we implement the core of it in ~50 lines so you can
see exactly what happens to the numbers.
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightOnlyQuantLinear(nn.Module):
    """Symmetric weight-only quantized ``nn.Linear`` (per-channel or per-group)."""

    def __init__(self, linear: nn.Linear, bits: int = 8, group_size: Optional[int] = None):
        super().__init__()
        assert bits in (2, 3, 4, 8)
        w = linear.weight.detach().float()
        out_f, in_f = w.shape
        gs = group_size or in_f
        assert in_f % gs == 0, f"in_features {in_f} not divisible by group_size {gs}"
        wg = w.view(out_f, in_f // gs, gs)
        qmax = 2 ** (bits - 1) - 1
        scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax).to(torch.int8)
        self.register_buffer("qweight", q)  # int8 container; real kernels pack 2x int4 per byte
        self.register_buffer("scale", scale.half())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.bits, self.group_size, self.in_features, self.out_features = bits, gs, in_f, out_f

    def dequantize(self) -> torch.Tensor:
        return (self.qweight.float() * self.scale.float()).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def storage_bytes(self) -> float:
        """Bytes a packed deployment format would need (not the int8 container)."""
        b = self.qweight.numel() * self.bits / 8 + self.scale.numel() * 2
        return b + (0 if self.bias is None else self.bias.numel() * 2)

    def extra_repr(self) -> str:
        return f"{self.in_features}->{self.out_features}, W{self.bits}, group={self.group_size}"


def quantize_weights(
    model: nn.Module,
    bits: int = 8,
    group_size: Optional[int] = None,
    skip: Iterable[str] = (),
    inplace: bool = False,
) -> nn.Module:
    """Replace every ``nn.Linear`` (except names containing a ``skip`` substring)."""
    model = model if inplace else copy.deepcopy(model)
    skip = tuple(skip)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                gs = group_size if group_size and child.in_features % group_size == 0 else None
                setattr(module, child_name, WeightOnlyQuantLinear(child, bits, gs))
    return model


def model_storage_mb(model: nn.Module) -> float:
    """Deployment size: packed quantized weights + other params at their current dtype."""
    total = 0.0
    quant_param_ids = set()
    for m in model.modules():
        if isinstance(m, WeightOnlyQuantLinear):
            total += m.storage_bytes()
            if m.bias is not None:
                quant_param_ids.add(id(m.bias))
    for p in model.parameters():
        if id(p) not in quant_param_ids:
            total += p.numel() * p.element_size()
    return total / 1024**2


def quantization_error(model: nn.Module, qmodel: nn.Module) -> Dict[str, float]:
    """Mean relative weight error per quantized layer (sanity check)."""
    ref = dict(model.named_modules())
    errs: Dict[str, float] = {}
    for name, m in qmodel.named_modules():
        if isinstance(m, WeightOnlyQuantLinear):
            w = ref[name].weight.detach().float()
            errs[name] = float((m.dequantize() - w).norm() / w.norm())
    return errs


def benchmark(fn: Callable[[], object], warmup: int = 5, iters: int = 30, device: Optional[torch.device] = None) -> Dict[str, float]:
    """Wall-clock latency in ms with proper GPU synchronization."""
    def sync():
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        elif device is not None and device.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"mean_ms": sum(times) / len(times), "p50_ms": times[len(times) // 2], "p90_ms": times[int(0.9 * (len(times) - 1))]}
