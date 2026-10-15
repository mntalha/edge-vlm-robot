"""Benchmark TinyVLM latency per stage and precision on the current device.

Run this on your desktop GPU *and* on the Jetson to compare:
    python scripts/benchmark.py --ckpt checkpoints/tiny_robot_vlm.pt
On Jetson, first:  sudo nvpmodel -m 0 && sudo jetson_clocks   (max performance)
"""
from __future__ import annotations

import argparse
import json

import torch

from edge_vlm.data import generate_samples, image_to_tensor
from edge_vlm.memory import format_table
from edge_vlm.optimize import benchmark, model_storage_mb, quantize_weights
from edge_vlm.train import get_device, load_checkpoint


def run(model, tok, device, dtype, sample, iters):
    model = model.to(device=device, dtype=dtype).eval()
    ids = torch.tensor([tok.encode_prompt(sample.prompt, model.cfg.num_image_tokens)], device=device)
    px = image_to_tensor(sample.image).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        img = model.encode_image(px)
        h = model.merge_embeddings(ids, img)
        _, past = model.lm_forward(h)
        step = model.embed(ids[:, -1:])
        return {
            "vision_ms": benchmark(lambda: model.encode_image(px), iters=iters, device=device)["p50_ms"],
            "prefill_ms": benchmark(lambda: model.lm_forward(h), iters=iters, device=device)["p50_ms"],
            "decode_tok_ms": benchmark(lambda: model.lm_forward(step, h.shape[1], past), iters=iters, device=device)["p50_ms"],
            "full_action_ms": benchmark(
                lambda: model.generate(ids, px, max_new_tokens=4, eos_token_id=tok.eos_id), iters=iters, device=device
            )["p50_ms"],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tiny_robot_vlm.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    device = get_device(args.device)
    model, tok = load_checkpoint(args.ckpt)
    sample = generate_samples(1, seed=42)[0]
    half = torch.float16 if device.type == "cuda" else torch.bfloat16
    variants = {
        "fp32": (model, torch.float32),
        str(half).split(".")[-1]: (model, half),
        "W8A16 (weight-only int8)": (quantize_weights(model, bits=8), torch.float32 if device.type == "cpu" else half),
        "W4A16 (int4, group 32)": (quantize_weights(model, bits=4, group_size=32), torch.float32 if device.type == "cpu" else half),
    }
    rows = []
    for name, (m, dt) in variants.items():
        size = model_storage_mb(m)  # measure before casting
        r = run(m, tok, device, dt, sample, args.iters)
        rows.append({"variant": name, "size_MB": round(size, 2), **{k: round(v, 2) for k, v in r.items()}})
    print(f"device: {device}")
    print(format_table(rows))
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
