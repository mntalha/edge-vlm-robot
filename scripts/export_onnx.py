"""Export a trained TinyVLM checkpoint to ONNX (vision + decoder) and verify it.

    python scripts/export_onnx.py --ckpt checkpoints/tiny_robot_vlm.pt --out exports/
Then on the Jetson:  bash deploy/jetson/build_trt_engines.sh exports/
"""
from __future__ import annotations

import argparse

from edge_vlm.export import export_onnx, verify_onnx
from edge_vlm.train import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tiny_robot_vlm.pt")
    ap.add_argument("--out", default="exports")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, _ = load_checkpoint(args.ckpt)
    paths = export_onnx(model, args.out, opset=args.opset)
    print("exported:", {k: str(v) for k, v in paths.items()})
    print("onnxruntime vs pytorch:", verify_onnx(model, paths))


if __name__ == "__main__":
    main()
