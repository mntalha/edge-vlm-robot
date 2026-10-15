# edge-vlm-robot

**Learn vision-language models by building one for a real robot task, sized for NVIDIA Jetson.**

> *"Bring me the red cup from the kitchen."*
> A small wheeled robot with a camera understands the command, drives to the right room, finds the right object among distractors, picks it up, and brings it back. Everything runs on the robot's own small NVIDIA GPU.

This repo is both a **learning path** (notebooks that explain every piece) and the **starting codebase** for that robot. It follows three pillars:

1. **An optimized small VLM:** build one from scratch, then fine-tune a pretrained one (SmolVLM).
2. **Make it run on the hardware:** ONNX → TensorRT / TensorRT Edge-LLM / llama.cpp on Jetson, ROS 2 integration.
3. **Respect hardware constraints:** memory, bandwidth, power and latency budgets (Orin Nano 8 GB first, then scale up).

```
 voice ─► ASR ─► command ─┐
 camera ─► image ─────────┼─► SYSTEM 2: small VLM (1-5 Hz) ─► skill ─► guardrail ─► SYSTEM 1: Nav2 / grasp / motors
 SLAM ─► room, gripper ───┘     "pick red cup"                              (20-1000 Hz)
```

## Notebooks (do them in order)

| # | Notebook | You learn | Runs on |
|---|---|---|---|
| 00 | [Roadmap & NVIDIA stack](notebooks/00_roadmap_and_nvidia_stack.ipynb) | Jetson lineup, JetPack/TensorRT/Edge-LLM/Isaac, memory & bandwidth math | CPU |
| 01 | [VLM from scratch](notebooks/01_vlm_from_scratch.ipynb) | patchify, ViT, pixel shuffle, projector, RoPE, GQA, KV cache | CPU |
| 02 | [Robot world & dataset](notebooks/02_robot_world_and_dataset.ipynb) | simulator, expert policy, behaviour cloning data, DART noise | CPU |
| 03 | [Train the tiny robot VLM](notebooks/03_train_tiny_robot_vlm.ipynb) | training, open- vs closed-loop eval, vision ablation | GPU recommended |
| 04 | [Pretrained small VLM + LoRA](notebooks/04_pretrained_small_vlm_lora.ipynb) | SmolVLM-256M zero-shot vs LoRA fine-tune | **NVIDIA GPU** |
| 05 | [Optimization & quantization](notebooks/05_optimization_quantization.ipynb) | FP16, INT8/INT4 weight-only, groups, latency per stage, ONNX | GPU recommended |
| 06 | [TensorRT & Jetson deployment](notebooks/06_tensorrt_jetson_deployment.ipynb) | nvpmodel, containers, trtexec, TensorRT runtime, llama.cpp, Edge-LLM | **Jetson** |
| 07 | [Robot agent loop](notebooks/07_robot_agent_loop.ipynb) | guardrails, ROS 2 architecture, real hardware plan | CPU |

## Quick start (on your GPU machine)

```bash
git clone https://github.com/mntalha/edge-vlm-robot.git && cd edge-vlm-robot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                   # ~1 min, CPU is fine
jupyter lab notebooks/                   # start with 00

# or headless:
python scripts/train_tiny_vlm.py         # train + evaluate TinyVLM -> checkpoints/tiny_robot_vlm.pt
python scripts/benchmark.py              # latency per stage & precision
python scripts/export_onnx.py            # -> exports/vision.onnx, exports/decoder.onnx
```

On the Jetson: see [deploy/jetson/setup_jetson.sh](deploy/jetson/setup_jetson.sh), then `bash deploy/jetson/build_trt_engines.sh exports/` and `python deploy/jetson/trt_runner.py`.

## Repository layout

```
src/edge_vlm/
  model.py       TinyVLM: ViT + pixel shuffle + MLP projector + Llama-style decoder (RoPE, GQA, SwiGLU, KV cache)
  tokenizer.py   closed-vocabulary word tokenizer with <image> placeholders
  domain.py      rooms, objects, colors, command templates, prompt format
  actions.py     skill schema, text/JSON parsing, precondition guardrail
  sim/world.py   toy fetch world with 64x64 egocentric camera
  agent.py       expert policy, closed-loop FetchAgent, evaluation
  data.py        imitation-learning dataset (with DART noise)
  train.py       training loop, VLMPolicy, checkpoints
  optimize.py    weight-only INT8/INT4 quantization, storage size, benchmarking
  export.py      ONNX export (vision + decoder) and onnxruntime verification
  memory.py      memory / KV-cache / decode-speed budgeting
  hardware.py    Jetson profiles (Orin Nano/NX, AGX Orin, Thor)
scripts/         train_tiny_vlm.py, benchmark.py, export_onnx.py
deploy/jetson/   setup notes, TensorRT engine build, TensorRT Python runner
deploy/ros2/     VLM planner ROS 2 node (skeleton)
tests/           pytest suite (runs in GitHub Actions)
```

## Hardware targets

Designed for the **Jetson Orin Nano 8 GB** first. Profiles for Orin NX 16 GB, AGX Orin 64 GB and Thor are in `src/edge_vlm/hardware.py`, so you can re-run every budget for a bigger module without changing code.

## Status

See [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for what is done, what is verified, and the next steps.
