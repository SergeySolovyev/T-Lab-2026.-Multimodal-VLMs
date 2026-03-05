from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path: str, label: str):
    df = pd.read_csv(path)
    df["label"] = label
    return df


def infer_progress_col(df: pd.DataFrame) -> str:
    if "update" in df.columns:
        return "update"
    if "epoch" in df.columns:
        return "epoch"
    return df.columns[0]


def first_reach_progress(df: pd.DataFrame, threshold: float) -> float | None:
    if "success_rate" not in df.columns:
        return None
    progress_col = infer_progress_col(df)
    hit = df[df["success_rate"] >= threshold]
    if hit.empty:
        return None
    return float(hit.iloc[0][progress_col])


def plot_metric(
    sft: pd.DataFrame,
    ga: pd.DataFrame,
    gt: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)

    if metric_col in sft.columns:
        ax.plot(
            sft[infer_progress_col(sft)],
            sft[metric_col],
            label="SFT",
            linewidth=2.2,
            marker="o",
            markersize=3,
            alpha=0.95,
        )
    if metric_col in ga.columns:
        ax.plot(
            ga[infer_progress_col(ga)],
            ga[metric_col],
            label="GRPO-action",
            linewidth=2.2,
            marker="o",
            markersize=3,
            alpha=0.95,
        )
    if metric_col in gt.columns:
        ax.plot(
            gt[infer_progress_col(gt)],
            gt[metric_col],
            label="GRPO-text+action",
            linewidth=2.2,
            marker="o",
            markersize=3,
            alpha=0.95,
        )

    ax.set_xlabel("Training progress", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=str, default="artifacts/sft/history.csv")
    parser.add_argument("--grpo_action", type=str, default="artifacts/grpo_action/history.csv")
    parser.add_argument("--grpo_text_action", type=str, default="artifacts/grpo_text_action/history.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts/plots")
    parser.add_argument("--success_threshold", type=float, default=0.80)
    parser.add_argument("--episodes_per_update", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft = read_csv(args.sft, "SFT")
    ga = read_csv(args.grpo_action, "GRPO-action")
    gt = read_csv(args.grpo_text_action, "GRPO-text+action")

    plot_metric(
        sft=sft,
        ga=ga,
        gt=gt,
        metric_col="success_rate",
        ylabel="Success rate",
        title="Success rate during training",
        out_path=out_dir / "success_rate.png",
    )
    plot_metric(
        sft=sft,
        ga=ga,
        gt=gt,
        metric_col="avg_return",
        ylabel="Episode return",
        title="Average return during training",
        out_path=out_dir / "return.png",
    )

    summary = pd.DataFrame(
        [
            {
                "method": "SFT",
                "final_success": float(sft["success_rate"].iloc[-1]),
                "best_success": float(sft["success_rate"].max()),
                "final_return": float(sft["avg_return"].iloc[-1]),
                "best_return": float(sft["avg_return"].max()),
            },
            {
                "method": "GRPO-action",
                "final_success": float(ga["success_rate"].iloc[-1]),
                "best_success": float(ga["success_rate"].max()),
                "final_return": float(ga["avg_return"].iloc[-1]),
                "best_return": float(ga["avg_return"].max()),
            },
            {
                "method": "GRPO-text+action",
                "final_success": float(gt["success_rate"].iloc[-1]),
                "best_success": float(gt["success_rate"].max()),
                "final_return": float(gt["avg_return"].iloc[-1]),
                "best_return": float(gt["avg_return"].max()),
            },
        ]
    )
    summary.to_csv(out_dir / "summary_table.csv", index=False)

    sample_rows = []
    sample_rows.append(
        {
            "method": "SFT",
            "success_threshold": float(args.success_threshold),
            "first_progress": None,
            "episodes_to_threshold": None,
            "env_steps_to_threshold": None,
        }
    )
    for method, df in [("GRPO-action", ga), ("GRPO-text+action", gt)]:
        progress = first_reach_progress(df, args.success_threshold)
        if progress is None:
            episodes = None
            env_steps = None
        else:
            episodes = float(progress) * float(args.episodes_per_update) * float(args.group_size)
            env_steps = episodes * float(args.max_steps)
        sample_rows.append(
            {
                "method": method,
                "success_threshold": float(args.success_threshold),
                "first_progress": progress,
                "episodes_to_threshold": episodes,
                "env_steps_to_threshold": env_steps,
            }
        )
    sample_eff = pd.DataFrame(sample_rows)
    sample_eff.to_csv(out_dir / "sample_efficiency.csv", index=False)

    print("Saved:")
    print(f"- {out_dir / 'success_rate.png'}")
    print(f"- {out_dir / 'return.png'}")
    print(f"- {out_dir / 'summary_table.csv'}")
    print(f"- {out_dir / 'sample_efficiency.csv'}")
    print("\nSummary table:")
    print(summary)
    print("\nSample efficiency:")
    print(sample_eff)


if __name__ == "__main__":
    main()
