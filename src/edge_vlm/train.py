"""Training, evaluation, checkpointing and the VLM-backed robot policy."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from .agent import AgentState
from .data import RobotVLMDataset, Sample, generate_samples, image_to_tensor, make_collate
from .model import TinyVLM, TinyVLMConfig
from .tokenizer import RobotTokenizer


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(tokenizer: RobotTokenizer, **overrides) -> TinyVLM:
    cfg = TinyVLMConfig(vocab_size=len(tokenizer), image_token_id=tokenizer.image_id, pad_token_id=tokenizer.pad_id)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return TinyVLM(cfg)


def train(
    model: TinyVLM,
    tokenizer: RobotTokenizer,
    train_samples: List[Sample],
    val_samples: Optional[List[Sample]] = None,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 0.05,
    device: Optional[torch.device] = None,
    log_every: int = 50,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """Plain AdamW + warmup/cosine schedule. Returns a loss history."""
    torch.manual_seed(seed)
    device = device or get_device()
    model.to(device).train()
    n_img = model.cfg.num_image_tokens
    collate = make_collate(tokenizer.pad_id)
    dl = DataLoader(RobotVLMDataset(train_samples, tokenizer, n_img), batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    total = epochs * len(dl)
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    hist: Dict[str, List[float]] = {"step": [], "train_loss": [], "val_loss": []}
    step, t0 = 0, time.time()
    for ep in range(epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, loss = model(batch["input_ids"], batch["pixel_values"], batch["labels"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % log_every == 0:
                hist["step"].append(step)
                hist["train_loss"].append(loss.item())
            step += 1
        vl = eval_loss(model, tokenizer, val_samples, device=device) if val_samples else float("nan")
        hist["val_loss"].append(vl)
        print(f"epoch {ep + 1}/{epochs}  train_loss {loss.item():.4f}  val_loss {vl:.4f}  ({time.time() - t0:.0f}s)")
    model.eval()
    return hist


@torch.no_grad()
def eval_loss(model: TinyVLM, tokenizer: RobotTokenizer, samples: List[Sample], batch_size: int = 128, device=None) -> float:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    dl = DataLoader(
        RobotVLMDataset(samples, tokenizer, model.cfg.num_image_tokens),
        batch_size=batch_size,
        collate_fn=make_collate(tokenizer.pad_id),
    )
    tot, n = 0.0, 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, loss = model(batch["input_ids"], batch["pixel_values"], batch["labels"])
        tot += loss.item() * len(batch["input_ids"])
        n += len(batch["input_ids"])
    model.train(was_training)
    return tot / max(n, 1)


class VLMPolicy:
    """Wraps a trained TinyVLM so the robot agent can query it."""

    def __init__(self, model: TinyVLM, tokenizer: RobotTokenizer, device=None, max_new_tokens: int = 6):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def predict(self, image, prompt: str) -> str:
        ids = torch.tensor([self.tok.encode_prompt(prompt, self.model.cfg.num_image_tokens)], device=self.device)
        px = image_to_tensor(image).unsqueeze(0).to(self.device)
        dtype = next(self.model.parameters()).dtype
        if dtype in (torch.float16, torch.bfloat16):
            px = px.to(dtype)
        out = self.model.generate(ids, px, max_new_tokens=self.max_new_tokens, eos_token_id=self.tok.eos_id)
        return self.tok.decode(out.tolist())

    def act(self, state: AgentState) -> str:
        return self.predict(state.image, state.prompt)


def action_accuracy(policy: VLMPolicy, samples: List[Sample]) -> Dict[str, float]:
    """Exact-match accuracy of the predicted next skill, overall and per skill."""
    hits: Dict[str, List[int]] = {}
    for s in samples:
        ok = int(policy.predict(s.image, s.prompt) == s.action)
        hits.setdefault(s.action.split()[0], []).append(ok)
        hits.setdefault("overall", []).append(ok)
    return {k: sum(v) / len(v) for k, v in hits.items()}


# ----------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------
def save_checkpoint(model: TinyVLM, tokenizer: RobotTokenizer, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "vocab": tokenizer.itos, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Union[str, Path], device=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tok = RobotTokenizer(ckpt["vocab"])
    model = TinyVLM(TinyVLMConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device or "cpu").eval(), tok


def load_or_train(path: Union[str, Path], n_episodes: int = 3000, epochs: int = 6, device=None):
    """Load a checkpoint if present, otherwise train the default model and save it."""
    path = Path(path)
    if path.exists():
        return load_checkpoint(path, device)
    tok = RobotTokenizer.from_domain()
    model = build_model(tok)
    train(model, tok, generate_samples(n_episodes, seed=0), generate_samples(200, seed=10_000), epochs=epochs, device=device)
    save_checkpoint(model, tok, path)
    return model.to(device or "cpu").eval(), tok
