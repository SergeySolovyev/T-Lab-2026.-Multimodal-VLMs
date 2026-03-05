"""Generate interim comparison bar chart from available SFT & GRPO-action data."""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(ROOT, "artifacts", "plots")
os.makedirs(PLOTS, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────
sft = pd.read_csv(os.path.join(ROOT, "artifacts", "sft", "history.csv"))
grpo = pd.read_csv(os.path.join(ROOT, "artifacts", "grpo_action", "history.csv"))

labels = ["SFT (1 epoch)", "GRPO-action (10 upd)"]
success = [sft["success_rate"].iloc[-1], grpo["success_rate"].iloc[-1]]
returns = [sft["avg_return"].iloc[-1], grpo["avg_return"].iloc[-1]]
losses  = [sft["train_loss"].iloc[-1], grpo["train_loss"].iloc[-1]]

# ── plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
colors = ["#4C72B0", "#DD8452"]

for ax, metric, title, fmt in zip(
    axes,
    [success, returns, losses],
    ["Success Rate", "Avg Return", "Train Loss"],
    [".0%", ".4f", ".5f"],
):
    bars = ax.bar(labels, metric, color=colors, edgecolor="white", width=0.55)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(max(metric) * 1.35, 0.001))
    for bar, val in zip(bars, metric):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(metric) * 0.04,
            format(val, fmt),
            ha="center", va="bottom", fontsize=11,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Interim Results — MiniGrid-Empty-6×6  (NanoVLM-222M, MVP profile)",
    fontsize=14, fontweight="bold", y=1.02,
)
fig.tight_layout()
out = os.path.join(PLOTS, "interim_comparison.png")
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
