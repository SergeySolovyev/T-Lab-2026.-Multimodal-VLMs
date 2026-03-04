from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path: str, label: str):
    df = pd.read_csv(path)
    df["label"] = label
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=str, default="artifacts/sft/history.csv")
    parser.add_argument("--grpo_action", type=str, default="artifacts/grpo_action/history.csv")
    parser.add_argument("--grpo_text_action", type=str, default="artifacts/grpo_text_action/history.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft = read_csv(args.sft, "SFT")
    ga = read_csv(args.grpo_action, "GRPO-action")
    gt = read_csv(args.grpo_text_action, "GRPO-text+action")

    plt.figure(figsize=(8, 5))
    if "epoch" in sft.columns:
        plt.plot(sft["epoch"], sft["success_rate"], label="SFT")
    if "update" in ga.columns:
        plt.plot(ga["update"], ga["success_rate"], label="GRPO-action")
    if "update" in gt.columns:
        plt.plot(gt["update"], gt["success_rate"], label="GRPO-text+action")
    plt.xlabel("Training progress")
    plt.ylabel("Success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate.png", dpi=180)

    summary = pd.DataFrame(
        [
            {"method": "SFT", "final_success_rate": float(sft["success_rate"].iloc[-1]), "final_return": float(sft["avg_return"].iloc[-1])},
            {"method": "GRPO-action", "final_success_rate": float(ga["success_rate"].iloc[-1]), "final_return": float(ga["avg_return"].iloc[-1])},
            {"method": "GRPO-text+action", "final_success_rate": float(gt["success_rate"].iloc[-1]), "final_return": float(gt["avg_return"].iloc[-1])},
        ]
    )
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
