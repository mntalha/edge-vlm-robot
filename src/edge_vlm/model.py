"""TinyVLM: a from-scratch, edge-friendly vision-language model.

Architecture (same blueprint as LLaVA / SmolVLM / Idefics3, just tiny):

    image ──► ViT encoder ──► pixel-shuffle ──► MLP projector ──┐  visual tokens
                                                                ▼
    "<bos> <image>*N cmd: bring me ... <sep>" ──► embeddings ──► merge ──► decoder LM ──► "pick red cup"

Edge-relevant design choices you will also see in production small models:

* **Pixel shuffle** trades spatial tokens for channels -> 4x fewer visual
  tokens -> shorter sequence -> less compute and a smaller KV cache.
* **Grouped-query attention (GQA)**: few K/V heads shared by many Q heads ->
  KV cache shrinks by ``n_heads / n_kv_heads``.
* **RoPE** positions (no learned position table in the LM).
* **RMSNorm + SwiGLU** like Llama/Qwen/SmolLM.
* **Tied input/output embeddings** -> saves ``vocab * dim`` parameters.
* **KV cache** for fast autoregressive decoding.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TinyVLMConfig:
    vocab_size: int = 128
    image_token_id: int = 5
    pad_token_id: int = 0
    # vision encoder
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    vision_dim: int = 128
    vision_layers: int = 3
    vision_heads: int = 4
    pixel_shuffle: int = 2
    # language model
    lm_dim: int = 192
    lm_layers: int = 4
    lm_heads: int = 6
    lm_kv_heads: int = 2
    lm_ffn_dim: int = 512
    max_seq_len: int = 128
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_image_tokens(self) -> int:
        return self.num_patches // (self.pixel_shuffle**2)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Vision encoder (a mini ViT, bidirectional attention)
# ----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Cut the image into PxP patches and linearly embed each (a strided conv)."""

    def __init__(self, cfg: TinyVLMConfig):
        super().__init__()
        self.proj = nn.Conv2d(cfg.in_channels, cfg.vision_dim, cfg.patch_size, stride=cfg.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.vision_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W) -> (B,N,D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.heads = heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: TinyVLMConfig):
        super().__init__()
        self.patch = PatchEmbed(cfg)
        self.blocks = nn.ModuleList(ViTBlock(cfg.vision_dim, cfg.vision_heads, cfg.dropout) for _ in range(cfg.vision_layers))
        self.norm = nn.LayerNorm(cfg.vision_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch(pixel_values)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def pixel_shuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """(B, N, D) on a sqrt(N) grid -> (B, N/scale^2, D*scale^2).

    Groups each ``scale x scale`` neighbourhood of patches into one token by
    concatenating their features (SmolVLM/Idefics3 use scale=3 or 4).
    """
    if scale == 1:
        return x
    B, N, D = x.shape
    g = int(math.isqrt(N))
    assert g * g == N and g % scale == 0, "patch grid must be square and divisible by scale"
    x = x.view(B, g, g // scale, D * scale)  # merge `scale` neighbours along width
    x = x.permute(0, 2, 1, 3)  # (B, g/s, g, D*s)
    x = x.reshape(B, g // scale, g // scale, D * scale * scale)  # merge along height
    x = x.permute(0, 2, 1, 3)
    return x.reshape(B, N // (scale * scale), D * scale * scale)


class Projector(nn.Module):
    """Maps vision features into the LM's embedding space (the 'connector')."""

    def __init__(self, cfg: TinyVLMConfig):
        super().__init__()
        in_dim = cfg.vision_dim * cfg.pixel_shuffle**2
        self.scale = cfg.pixel_shuffle
        self.mlp = nn.Sequential(nn.Linear(in_dim, cfg.lm_dim), nn.GELU(), nn.Linear(cfg.lm_dim, cfg.lm_dim))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.mlp(pixel_shuffle(feats, self.scale))


# ----------------------------------------------------------------------------
# Decoder-only language model (Llama-style)
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(seq_len: int, head_dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate query/key pairs by a position-dependent angle. x: (B,H,T,hd), cos/sin: (T,hd)."""
    return x * cos + _rotate_half(x) * sin


class GQAttention(nn.Module):
    def __init__(self, cfg: TinyVLMConfig):
        super().__init__()
        assert cfg.lm_dim % cfg.lm_heads == 0 and cfg.lm_heads % cfg.lm_kv_heads == 0
        self.h, self.kvh = cfg.lm_heads, cfg.lm_kv_heads
        self.hd = cfg.lm_dim // cfg.lm_heads
        self.q_proj = nn.Linear(cfg.lm_dim, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.lm_dim, self.kvh * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.lm_dim, self.kvh * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, cfg.lm_dim, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kvh, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)  # cache stores only kv_heads -> this is the GQA memory win
        rep = self.h // self.kvh
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        # Causal mask only needed when processing >1 new token without history
        # (prefill / training). A single decode step may attend to everything.
        causal = past is None and T > 1
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.o_proj(a.transpose(1, 2).reshape(B, T, -1)), present


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: TinyVLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.lm_dim)
        self.attn = GQAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.lm_dim)
        self.mlp = SwiGLU(cfg.lm_dim, cfg.lm_ffn_dim)

    def forward(self, x, cos, sin, past=None):
        a, present = self.attn(self.attn_norm(x), cos, sin, past)
        x = x + a
        return x + self.mlp(self.mlp_norm(x)), present


# ----------------------------------------------------------------------------
# The full VLM
# ----------------------------------------------------------------------------
class TinyVLM(nn.Module):
    def __init__(self, cfg: TinyVLMConfig):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg)
        self.projector = Projector(cfg)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.lm_dim)
        self.layers = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.lm_layers))
        self.norm = RMSNorm(cfg.lm_dim)
        cos, sin = rope_tables(cfg.max_seq_len, cfg.lm_dim // cfg.lm_heads, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # -- pieces ---------------------------------------------------------------
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B, num_image_tokens, lm_dim) visual tokens."""
        return self.projector(self.vision(pixel_values))

    def merge_embeddings(self, input_ids: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """Replace each ``<image>`` placeholder embedding with the next visual token.

        Uses gather + where (instead of boolean indexing) so it exports cleanly
        to ONNX / TensorRT.
        """
        text = self.embed(input_ids)
        mask = input_ids == self.cfg.image_token_id
        idx = (mask.long().cumsum(dim=1) - 1).clamp(min=0, max=image_embeds.shape[1] - 1)
        gathered = torch.gather(image_embeds, 1, idx.unsqueeze(-1).expand(-1, -1, image_embeds.shape[-1]))
        return torch.where(mask.unsqueeze(-1), gathered.to(text.dtype), text)

    def lm_forward(
        self, h: torch.Tensor, start_pos: int = 0, past: Optional[KVCache] = None
    ) -> Tuple[torch.Tensor, KVCache]:
        T = h.shape[1]
        cos = self.rope_cos[start_pos : start_pos + T].to(h.dtype)
        sin = self.rope_sin[start_pos : start_pos + T].to(h.dtype)
        presents: KVCache = []
        for i, layer in enumerate(self.layers):
            h, p = layer(h, cos, sin, None if past is None else past[i])
            presents.append(p)
        logits = F.linear(self.norm(h), self.embed.weight)  # tied output head
        return logits, presents

    # -- training forward ------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.merge_embeddings(input_ids, self.encode_image(pixel_values))
        logits, _ = self.lm_forward(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # -- inference -------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 8,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Greedy decoding for a single example (batch size 1)."""
        assert input_ids.shape[0] == 1, "generate() supports batch size 1"
        h = self.merge_embeddings(input_ids, self.encode_image(pixel_values))
        out: List[int] = []
        if use_cache:
            logits, past = self.lm_forward(h)  # prefill: whole prompt at once
            pos = h.shape[1]
            for _ in range(max_new_tokens):
                nxt = int(logits[0, -1].argmax())
                if nxt == eos_token_id:
                    break
                out.append(nxt)
                if pos >= self.cfg.max_seq_len:
                    break
                step = self.embed(torch.tensor([[nxt]], device=h.device))
                logits, past = self.lm_forward(step, start_pos=pos, past=past)  # decode: 1 token
                pos += 1
        else:  # recompute the whole sequence each step (slow - for comparison)
            for _ in range(max_new_tokens):
                logits, _ = self.lm_forward(h)
                nxt = int(logits[0, -1].argmax())
                if nxt == eos_token_id or h.shape[1] >= self.cfg.max_seq_len:
                    break
                out.append(nxt)
                h = torch.cat([h, self.embed(torch.tensor([[nxt]], device=h.device))], dim=1)
        return torch.tensor(out, dtype=torch.long)

    # -- introspection ---------------------------------------------------------
    def param_breakdown(self) -> Dict[str, int]:
        count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        lm = count(self.layers) + count(self.norm)
        return {
            "vision_encoder": count(self.vision),
            "projector": count(self.projector),
            "embeddings (tied)": count(self.embed),
            "language_model": lm,
            "total": count(self),
        }
