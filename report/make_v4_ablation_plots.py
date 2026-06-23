#!/usr/bin/env python3
"""Regenerate the DriveJudgeBench v4-routing ablation figures.

Two bar charts comparing the agent baseline against the v4k (no routing fix)
and v4m (strict routing) configurations, overall and per category. The numbers
are baked in from the analyzed v4 ablation run so the figures can be rebuilt
without the full results tree on hand.

Outputs (overwritten):
    report/figures/drivejudgebench_overall.png
    report/figures/drivejudgebench_category.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Bar colours: Baseline / v4k / v4m
COLORS = ["#4C72B0", "#55A868", "#DD8452"]

# Overall accuracy (%) per configuration.
OVERALL = {"Baseline": 42.0, "v4k": 45.6, "v4m": 46.9}

# Per-category accuracy (%): [Baseline, v4k, v4m].
CATEGORIES = {
    "Reality": [20, 47, 56],
    "Safety": [61, 52, 49],
    "Traffic\nLaws": [64, 46, 12],
    "Visual": [94, 59, 71],
    "Spatial-\nTemporal": [63, 46, 56],
    "Artifacts": [24, 15, 18],
}

LEGEND = [
    f"Baseline ({OVERALL['Baseline']:.1f}%)",
    f"v4k — no routing fix ({OVERALL['v4k']:.1f}%)",
    f"v4m — strict routing ({OVERALL['v4m']:.1f}%)",
]


def make_overall() -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7.5))

    labels = list(OVERALL.keys())
    values = list(OVERALL.values())
    label_with_pct = [f"{lbl}\n({val:.1f}%)" for lbl, val in OVERALL.items()]

    bars = ax.bar(labels, values, color=COLORS, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=20, fontweight="bold")

    ax.set_title("DriveJudgeBench — Overall Accuracy", fontsize=22, fontweight="bold", pad=18)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.set_ylim(0, 60)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(label_with_pct, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    sns.despine()

    out = FIG_DIR / "drivejudgebench_overall.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_category() -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 7.5))

    cats = list(CATEGORIES.keys())
    x = np.arange(len(cats))
    width = 0.27

    for i in range(3):
        vals = [CATEGORIES[c][i] for c in cats]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=COLORS[i], label=LEGEND[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val}%",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_title("DriveJudgeBench — Per-Category Accuracy", fontsize=22, fontweight="bold", pad=18)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(fontsize=13, loc="upper right")
    sns.despine()

    out = FIG_DIR / "drivejudgebench_category.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_overall()
    make_category()
