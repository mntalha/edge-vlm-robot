"""Synthetic dataset generation (imitation learning from an expert).

Each sample is one decision point of an episode:
    (camera image, text prompt with command + robot state) -> next skill text

We roll out the privileged expert in the simulator. With probability
``noise_prob`` we *execute* a random detour instead of the expert action (but
still record the expert's label). This "DART"-style noise shows the model
recovery states it would never see on perfect demonstrations - a cheap trick
against compounding errors in behaviour cloning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .actions import Action
from .agent import AgentState, expert_action
from .domain import SEARCH_ROOMS
from .sim.world import FetchWorld
from .tokenizer import IGNORE_INDEX, RobotTokenizer


@dataclass
class Sample:
    image: np.ndarray
    prompt: str
    action: str


def generate_samples(
    n_episodes: int,
    seed: int = 0,
    noise_prob: float = 0.2,
    max_steps: int = 10,
    image_size: int = 64,
) -> List[Sample]:
    world = FetchWorld(image_size=image_size, seed=seed)
    rng = np.random.default_rng(seed + 1)
    samples: List[Sample] = []
    for _ in range(n_episodes):
        world.reset()
        visited = [world.robot_room]
        for _ in range(max_steps):
            obs = world.observe()
            state = AgentState(obs.image, world.command, obs.room, obs.holding, list(visited), world.visible_objects())
            expert = expert_action(state)
            samples.append(Sample(obs.image, state.prompt, expert.to_text()))
            act = expert
            # only perturb exploration moves, never skip a visible target
            if expert.skill == "navigate" and rng.random() < noise_prob:
                choices = [r for r in SEARCH_ROOMS if r != obs.room and r != expert.target]
                if choices:
                    act = Action("navigate", str(rng.choice(choices)))
            res = world.step(act)
            if act.skill == "navigate" and act.target not in visited:
                visited.append(act.target)
            if res.done:
                break
    return samples


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 HWC [0,255] -> float CHW in [-1, 1] (SigLIP-style normalization)."""
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.5


class RobotVLMDataset(Dataset):
    def __init__(self, samples: List[Sample], tokenizer: RobotTokenizer, num_image_tokens: int):
        self.samples = samples
        self.tok = tokenizer
        self.n_img = num_image_tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        s = self.samples[i]
        ids, labels = self.tok.encode_example(s.prompt, s.action, self.n_img)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": image_to_tensor(s.image),
        }


def make_collate(pad_id: int):
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        T = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
        for i, b in enumerate(batch):  # right padding: causal attention never looks ahead
            n = len(b["input_ids"])
            ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
        return {"input_ids": ids, "labels": labels, "pixel_values": torch.stack([b["pixel_values"] for b in batch])}

    return collate


def action_distribution(samples: List[Sample]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in samples:
        k = s.action.split()[0]
        out[k] = out.get(k, 0) + 1
    return out

