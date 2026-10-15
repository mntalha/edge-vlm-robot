# Agent handoff: edge-vlm-robot

This is the context another agent (or you, later) needs to continue this project. Read it before changing anything.

## 1. Goal (from the user)

- The user is learning **vision-language models (VLMs)** by building an application. They target **physical AI** on **NVIDIA** hardware.
- End product: a **small wheeled robot with a camera (and a small arm)**. The user gives a command ("go pick X from the other room and bring it back"). The robot understands it, navigates, finds and picks the object, and returns it to the user. The model runs **on the robot's small GPU**.
- Three pillars: (1) create an optimized small VLM, (2) make it run on the hardware (NVIDIA stack), (3) respect hardware constraints.
- The user wants **notebooks** that teach each concept, not only code.

## 2. User preferences and constraints

| Topic | Decision |
|---|---|
| Target hardware | **Not decided yet.** Design for **Jetson Orin Nano 8 GB**, scale up later (Orin NX / AGX Orin / Thor profiles included). |
| Repo | **Public** GitHub repo **`mntalha/edge-vlm-robot`** (`gh` is authenticated as `mntalha`). |
| Local path | `/Users/talha/Documents/GitHub/edge-vlm-robot` (no worktree). |
| **Execution** | **Do NOT run training, notebooks or heavy code on the user's Mac.** The user said: "this is not the computer to run". The user has an NVIDIA GPU on *another* machine and will clone the repo there. Only write files and use git/gh here. Validation runs in **GitHub Actions CI**. |
| Git identity | Talha Kilic, talha.kilic@u.northwestern.edu. Commits include the `Co-authored-by: Copilot` trailer. |

## 3. What has been done (chronological)

1. Asked the user three questions: target device (→ "not decided, design for Orin Nano"), repo visibility/name (→ public `edge-vlm-robot`), workspace (→ this folder, no worktree).
2. Checked current NVIDIA projects exist (Oct 2026): `NVIDIA/TensorRT-Edge-LLM`, `dusty-nv/jetson-containers`, `huggingface/nanoVLM`, `NVIDIA/Isaac-GR00T` (N1.7), `nvidia-cosmos/cosmos-reason2`.
3. Wrote the `edge_vlm` package (`src/edge_vlm/`): hardware profiles, memory budgeting, domain, actions/guardrail, toy simulator, expert + agent, dataset generation, TinyVLM model, training, quantization, ONNX export.
4. **One local smoke test was run before the user asked to stop running things** (results below). The user then objected. The training run was stopped, and the local `.venv`, `/tmp` checkpoint, egg-info and `__pycache__` were deleted. **Nothing else has been executed locally since.**
5. Wrote tests, scripts, Jetson deploy files (TensorRT build script, TensorRT Python runner, setup notes), a ROS 2 planner node skeleton, a CI workflow, 8 notebooks (00-07), the README and this file.
6. Initial commit and push to `github.com/mntalha/edge-vlm-robot`. The first CI run had 36/37 passing. ONNX export failed because `is_causal` in SDPA was derived from a traced shape (`T > 1`).
7. Fixed it: `causal = past is None` in `GQAttention` (a 1x1 causal mask is harmless). **CI is now green: 37/37 tests pass and all 8 notebooks validate (torch 2.14.1+cpu, Python 3.11).**

### Smoke-test results (only local run, Python 3.9 / torch 2.8 CPU on an M2 that was heavily loaded)
- Expert policy: **100% success**, 4.6 avg steps over 200 episodes.
- 3000 episodes → 13,968 samples (navigate 4968, pick/return/handover 3000 each). Vocab 54 words at the time (now ~57 after adding `? . !`), 0 `<unk>` in prompts.
- TinyVLM default config: **2.35M params** (vision 0.63M, projector 0.14M, LM 1.57M, embeddings 0.01M), **16 image tokens**, sequence length ~47.
- Training steps ran on CPU and MPS without errors. Convergence and accuracy were **not** measured (the run was stopped).

## 4. Design decisions (and why)

- **System 2 / System 1 split** (like GR00T): the VLM picks *high-level skills* at 1-5 Hz (`navigate <room>`, `pick <color> <obj>`, `return`, `handover`). Navigation and grasping are classical / fast controllers. This keeps the VLM small and makes the task feasible on an 8 GB Jetson.
- **Toy simulator first** (64×64 top-down camera, 4 rooms, 20 objects with hard distractors) so everything trains in minutes. The VLM must use the image to decide `pick` vs `navigate`. Notebook 02 shows that identical prompts can need different actions.
- **Robot state is given as text** (room from localization, holding from the gripper, visited rooms from memory). The VLM does the grounding and decision only.
- **Behaviour cloning from a privileged expert** that sees ground-truth objects. **DART noise** perturbs only *navigate* moves (never skips a visible target), so the expert labels stay valid.
- **TinyVLM** mirrors SmolVLM/LLaVA: ViT → pixel shuffle (×2) → MLP projector → Llama-style decoder (RMSNorm, RoPE, GQA 6/2, SwiGLU, tied embeddings, KV cache). `<image>` placeholders are merged with `cumsum+gather+where` so the graph is ONNX/TensorRT friendly.
- **Guardrail** (`actions.parse_action` + `check_preconditions`): any malformed or invalid output causes a safe stop.
- **Quantization taught by hand** (`WeightOnlyQuantLinear`: symmetric per-channel / per-group INT8/INT4/INT3). The text explains that real speedups need fused kernels (TensorRT / Edge-LLM / llama.cpp).
- **Pretrained path:** SmolVLM-256M-Instruct + LoRA (text model only, via a regex target), `do_image_splitting=False` → 64 visual tokens. Deployment options: PyTorch, llama.cpp GGUF, or TensorRT Edge-LLM.
- Code supports **Python ≥3.9** (`from __future__ import annotations`, no `match`, no `X | Y` at runtime).

## 5. Verification status

| Item | Status |
|---|---|
| Package imports, expert, data generation, a training step | ✅ verified locally (smoke test) |
| `pytest` suite (domain, world/agent, model incl. KV-cache equivalence and pixel-shuffle layout, memory/quantization, train smoke + checkpoint roundtrip, ONNX export vs onnxruntime) | ✅ 37/37 pass in GitHub Actions (torch 2.14 CPU) |
| Notebook JSON validity + code cells compile | ✅ CI |
| Notebook *execution* (03 training accuracy, 05 numbers) | ❌ not run. Do it on the user's GPU machine |
| ONNX export + onnxruntime match (torch 2.14) | ✅ CI |
| Notebook 04 (HF transformers / peft API details for SmolVLM) | ❌ not run; needs GPU + internet |
| TensorRT engine build / `trt_runner.py` / int64 `input_ids` handling in TRT 10 | ❌ needs a Jetson |
| ROS 2 node (`deploy/ros2/vlm_planner_node.py`) | ❌ skeleton, untested |

## 6. Known risks / TODO for the next agent

1. Keep CI green. Any change to `model.py` attention must keep control flow independent of traced tensor shapes (ONNX export).
2. On the GPU machine, run notebook 03 and record real accuracy and closed-loop success. Tune `EPISODES`/`EPOCHS`/`lr` if `pick` accuracy is low (options: lower `pixel_shuffle` to 1, more vision layers, larger images).
3. Notebook 04: confirm the `AutoModelForImageTextToText` import, the chat template, and the label masking with the installed `transformers` version. Report LoRA results.
4. Jetson: confirm `trtexec` handles int64 `input_ids` (otherwise cast to int32 in `DecoderWrapper` and re-export). Add KV-cache I/O to the decoder export if needed.
5. Future work: skill-executor ROS 2 node (Nav2 action client, grasp pipeline), Isaac Sim data generation, a real-data fine-tune, an Edge-LLM path for a Qwen-VL / Cosmos Reason model, and a VLA (GR00T / LeRobot) stretch goal.
6. Hardware numbers in `hardware.py` come from public datasheets. Re-verify them for the module the user buys. A web search mentioned an "Orin Nano 2" (unverified, so it was not added).

## 7. How to continue

```bash
cd /Users/talha/Documents/GitHub/edge-vlm-robot   # (user's Mac: edit + git only)
gh run list -R mntalha/edge-vlm-robot --limit 3     # CI status
gh run view <id> --log-failed                        # if red
# on the GPU machine:
git clone https://github.com/mntalha/edge-vlm-robot && cd edge-vlm-robot
pip install -r requirements.txt && pip install -e . && pytest
jupyter lab notebooks/
```
