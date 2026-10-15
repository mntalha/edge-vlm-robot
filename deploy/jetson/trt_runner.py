"""Run the TinyVLM TensorRT engines on a Jetson and drive the simulated robot.

    python deploy/jetson/trt_runner.py --engines engines/ --episodes 50

Requires the `tensorrt` Python bindings that ship with JetPack (TensorRT 10.x API)
and a CUDA-enabled PyTorch (e.g. the l4t-pytorch container) used for GPU buffers.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict

import tensorrt as trt
import torch

from edge_vlm.agent import evaluate_policy
from edge_vlm.data import image_to_tensor
from edge_vlm.tokenizer import RobotTokenizer

_DTYPES = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


class TRTEngine:
    """Minimal TensorRT 10 engine wrapper using torch CUDA tensors as I/O buffers."""

    def __init__(self, path: Path):
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def __call__(self, **feeds: torch.Tensor) -> Dict[str, torch.Tensor]:
        stream = torch.cuda.current_stream()
        keep = []
        for name in self.inputs:
            t = feeds[name].to("cuda", _DTYPES[self.engine.get_tensor_dtype(name)]).contiguous()
            keep.append(t)
            self.ctx.set_input_shape(name, tuple(t.shape))
            self.ctx.set_tensor_address(name, t.data_ptr())
        outs = {}
        for name in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(name))
            outs[name] = torch.empty(shape, dtype=_DTYPES[self.engine.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, outs[name].data_ptr())
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outs


class TRTPolicy:
    """Same interface as edge_vlm.train.VLMPolicy, backed by TensorRT engines.

    The exported decoder has no KV cache, so each new token re-runs the prompt.
    That is fine for ~4-token actions; for real LLMs use TensorRT Edge-LLM,
    which manages the KV cache for you.
    """

    def __init__(self, engine_dir: Path, tokenizer: RobotTokenizer, num_image_tokens: int = 16, max_new_tokens: int = 6):
        self.vision = TRTEngine(Path(engine_dir) / "vision_fp16.engine")
        self.decoder = TRTEngine(Path(engine_dir) / "decoder_fp16.engine")
        self.tok, self.n_img, self.max_new = tokenizer, num_image_tokens, max_new_tokens

    @torch.no_grad()
    def predict(self, image, prompt: str) -> str:
        ids = self.tok.encode_prompt(prompt, self.n_img)
        img = self.vision(pixel_values=image_to_tensor(image).unsqueeze(0))["image_embeds"]
        out = []
        for _ in range(self.max_new):
            logits = self.decoder(input_ids=torch.tensor([ids + out]), image_embeds=img)["logits"]
            nxt = int(logits[0, -1].argmax())
            if nxt == self.tok.eos_id:
                break
            out.append(nxt)
        return self.tok.decode(out)

    def act(self, state) -> str:
        return self.predict(state.image, state.prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="engines")
    ap.add_argument("--ckpt", default="checkpoints/tiny_robot_vlm.pt", help="only used to read the vocabulary")
    ap.add_argument("--episodes", type=int, default=50)
    args = ap.parse_args()

    vocab = torch.load(args.ckpt, map_location="cpu", weights_only=False)["vocab"]
    policy = TRTPolicy(Path(args.engines), RobotTokenizer(vocab))
    t = time.time()
    stats = evaluate_policy(policy, n_episodes=args.episodes, seed=777)
    print(stats, f"{time.time() - t:.1f}s total")


if __name__ == "__main__":
    main()
