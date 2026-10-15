"""ONNX export - the hand-off point to TensorRT on Jetson.

Production VLM runtimes split the model into (at least) two engines:

1. ``vision.onnx``  : pixel_values -> image_embeds   (runs once per image)
2. ``decoder.onnx`` : input_ids + image_embeds -> logits (runs per token)

TensorRT (``trtexec`` / TensorRT Edge-LLM) then builds an optimized engine for
the exact GPU on the robot. The decoder exported here has no KV cache inputs to
stay readable; Edge-LLM / TensorRT-LLM add cache I/O and fused attention for you.
"""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn

from .model import TinyVLM


class VisionWrapper(nn.Module):
    def __init__(self, model: TinyVLM):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(pixel_values)


class DecoderWrapper(nn.Module):
    def __init__(self, model: TinyVLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        h = self.model.merge_embeddings(input_ids, image_embeds)
        logits, _ = self.model.lm_forward(h)
        return logits


def _export(module, args, path, input_names, output_names, dynamic_axes, opset):
    kwargs = dict(input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset)
    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        torch.onnx.export(module, args, str(path), **kwargs)  # older torch: classic exporter only
        return
    try:
        # classic TorchScript exporter: mature dynamic_axes support
        torch.onnx.export(module, args, str(path), dynamo=False, **kwargs)
    except Exception:  # newer torch may drop the classic exporter -> torch.export based one
        torch.onnx.export(module, args, str(path), dynamo=True, **kwargs)


def export_onnx(model: TinyVLM, out_dir: Union[str, Path], opset: int = 17) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).float().cpu().eval()  # never mutate the caller's model
    cfg = model.cfg
    px = torch.randn(1, cfg.in_channels, cfg.image_size, cfg.image_size)
    ids = torch.full((1, cfg.num_image_tokens + 10), 7, dtype=torch.long)
    ids[0, 1 : 1 + cfg.num_image_tokens] = cfg.image_token_id
    with torch.no_grad():
        img = model.encode_image(px)
        vision_path, decoder_path = out_dir / "vision.onnx", out_dir / "decoder.onnx"
        _export(VisionWrapper(model), (px,), vision_path, ["pixel_values"], ["image_embeds"],
                {"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}}, opset)
        _export(DecoderWrapper(model), (ids, img), decoder_path, ["input_ids", "image_embeds"], ["logits"],
                {"input_ids": {0: "batch", 1: "seq"}, "image_embeds": {0: "batch"}, "logits": {0: "batch", 1: "seq"}},
                opset)
    return {"vision": vision_path, "decoder": decoder_path}


def verify_onnx(model: TinyVLM, paths: Dict[str, Path], seq_len: int = 40, atol: float = 1e-3) -> Dict[str, float]:
    """Run both graphs with onnxruntime and compare against PyTorch."""
    import onnxruntime as ort

    model = copy.deepcopy(model).float().cpu().eval()  # never mutate the caller's model
    cfg = model.cfg
    px = torch.randn(1, cfg.in_channels, cfg.image_size, cfg.image_size)
    ids = torch.randint(6, cfg.vocab_size, (1, seq_len))
    ids[0, 1 : 1 + cfg.num_image_tokens] = cfg.image_token_id
    with torch.no_grad():
        ref_img = model.encode_image(px)
        ref_logits, _ = model.lm_forward(model.merge_embeddings(ids, ref_img))
    vs = ort.InferenceSession(str(paths["vision"]), providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(str(paths["decoder"]), providers=["CPUExecutionProvider"])
    img = vs.run(None, {"pixel_values": px.numpy()})[0]
    logits = ds.run(None, {"input_ids": ids.numpy(), "image_embeds": img})[0]
    err = {
        "vision_max_abs_err": float(np.abs(img - ref_img.numpy()).max()),
        "decoder_max_abs_err": float(np.abs(logits - ref_logits.numpy()).max()),
    }
    assert all(v < atol for v in err.values()), f"ONNX mismatch: {err}"
    return err
