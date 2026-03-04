from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.eval.evaluate import evaluate_policy
from src.models.nanovlm_policy import NanoVLMPolicy
from src.utils import ensure_dir, save_json, set_seed


def load_split(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "frames": data["frames"],
        "actions": data["actions"],
        "direct_prompts": data["direct_prompts"],
        "direct_targets": data["direct_targets"],
        "text_action_prompts": data["text_action_prompts"],
        "text_action_targets": data["text_action_targets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, default="artifacts/datasets/train.npz")
    parser.add_argument("--val_npz", type=str, default="artifacts/datasets/val.npz")
    parser.add_argument("--nanovlm_repo", type=str, default="external/nanoVLM")
    parser.add_argument("--model_source", type=str, default="lusxvr/nanoVLM-450M")
    parser.add_argument("--output_dir", type=str, default="artifacts/sft")
    parser.add_argument("--mode", type=str, choices=["action", "text_action"], default="action")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=100)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    train = load_split(args.train_npz)
    _val = load_split(args.val_npz)

    policy = NanoVLMPolicy(
        model_source=args.model_source,
        nanovlm_repo=args.nanovlm_repo,
        device="cuda",
    )

    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=args.lr, weight_decay=0.01)

    history = []

    for epoch in range(args.epochs):
        policy.set_mode(True)
        losses = []
        indices = np.random.permutation(len(train["actions"]))

        for idx in tqdm(indices, desc=f"sft epoch {epoch+1}/{args.epochs}"):
            frame = train["frames"][idx]
            if args.mode == "action":
                prompt = str(train["direct_prompts"][idx])
                target = str(train["direct_targets"][idx])
            else:
                prompt = str(train["text_action_prompts"][idx])
                target = str(train["text_action_targets"][idx])

            loss = policy.sft_loss(frame, prompt, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        eval_metrics = evaluate_policy(
            policy=policy,
            mode=args.mode,
            episodes=args.eval_episodes,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            **eval_metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(Path(args.output_dir) / "history.csv", index=False)

        ckpt_dir = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}"
        policy.save(str(ckpt_dir))

    final_dir = Path(args.output_dir) / "checkpoint_last"
    policy.save(str(final_dir))
    save_json({"mode": args.mode, "history": history}, Path(args.output_dir) / "summary.json")
    print(f"SFT finished. Last checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
