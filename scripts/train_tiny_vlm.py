"""Train the TinyVLM fetch policy end-to-end from the command line.

    python scripts/train_tiny_vlm.py --episodes 3000 --epochs 6 --out checkpoints/tiny_robot_vlm.pt
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from edge_vlm.agent import evaluate_policy
from edge_vlm.data import action_distribution, generate_samples
from edge_vlm.tokenizer import RobotTokenizer
from edge_vlm.train import VLMPolicy, action_accuracy, build_model, get_device, save_checkpoint, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--val-episodes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--out", default="checkpoints/tiny_robot_vlm.pt")
    args = ap.parse_args()

    device = get_device(args.device)
    tok = RobotTokenizer.from_domain()
    t = time.time()
    train_s = generate_samples(args.episodes, seed=0)
    val_s = generate_samples(args.val_episodes, seed=10_000)  # different seed = unseen episodes
    print(f"{len(train_s)} train / {len(val_s)} val samples in {time.time() - t:.0f}s  {action_distribution(train_s)}")

    model = build_model(tok)
    print("params:", model.param_breakdown(), "device:", device)
    train(model, tok, train_s, val_s, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    save_checkpoint(model, tok, args.out)

    policy = VLMPolicy(model, tok, device=device)
    acc = action_accuracy(policy, val_s)
    closed_loop = evaluate_policy(policy, n_episodes=args.eval_episodes, seed=777)
    print("next-action accuracy:", json.dumps({k: round(v, 3) for k, v in acc.items()}))
    print("closed-loop:", closed_loop)
    print("saved", args.out)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
