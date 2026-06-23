#!/usr/bin/env python3
"""Regenerate the accuracy-vs-inference-time scatter for the README.

Reads the committed `report/accuracy-vs-time.json` (per-model overall accuracy
and mean inference time) and draws base VLMs as circles, agentic configs as
squares. Output (overwritten):

    report/figures/scatter_accuracy_time.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA = HERE / "accuracy-vs-time.json"
OUT = HERE / "figures" / "scatter_accuracy_time.png"

# Internal agent stems -> the labels we want on the plot.
DISPLAY = {
    "Agent v4o": "Qwen3-VL + Tools",
    "Agent v4p": "Qwen3-VL + Tools + Hints",
}

COLORS = {
    "Qwen3-VL": "#E8820C",
    "LLaVA-OneVision": "#16A085",
    "InternVL3.5-8B": "#27AE60",
    "Cosmos-Reason": "#E74C3C",
    "InternVL3.5-30B": "#2E86DE",
    "Qwen-Omni": "#8E44AD",
    "Qwen3-VL + Tools": "#E8820C",
    "Qwen3-VL + Tools + Hints": "#8E44AD",
}

# label offset (points): dx, dy, ha, va, leader-line?
LABELS = {
    "Qwen3-VL + Tools": (14, 6, "left", "bottom", False),
    "Qwen3-VL + Tools + Hints": (14, -2, "left", "top", False),
    "InternVL3.5-30B": (14, -2, "left", "center", False),
    "Cosmos-Reason": (16, 6, "left", "bottom", False),
    "InternVL3.5-8B": (40, -26, "left", "top", True),
    "LLaVA-OneVision": (-120, -6, "right", "center", True),
    "Qwen3-VL": (-4, -20, "right", "top", False),
    "Qwen-Omni": (14, 0, "left", "center", False),
}


def main() -> None:
    points = json.loads(DATA.read_text())["points"]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 8))

    for p in points:
        name = DISPLAY.get(p["name"], p["name"])
        color = COLORS.get(name, "#555555")
        is_agent = p["category"] == "agent"
        ax.scatter(
            p["time"], p["accuracy"],
            s=320 if is_agent else 230,
            marker="s" if is_agent else "o",
            facecolor=color, edgecolor="black", linewidth=1.4, zorder=3,
        )

        dx, dy, ha, va, leader = LABELS.get(name, (12, 4, "left", "center", False))
        arrow = dict(arrowstyle="-", color=color, lw=1.3, alpha=0.6) if leader else None
        ax.annotate(
            name, (p["time"], p["accuracy"]),
            textcoords="offset points", xytext=(dx, dy),
            ha=ha, va=va, fontsize=13, fontweight="bold", color=color,
            arrowprops=arrow, zorder=2 if leader else 4,
        )

    # Legend: marker shape encodes base vs agentic (shape only, neutral colour).
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f",
                   markeredgecolor="black", markersize=13, label="Base VLMs"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#7f7f7f",
                   markeredgecolor="black", markersize=13, label="VLM + Tools (Agentic)"),
    ]
    ax.legend(handles=handles, fontsize=13, loc="center right", framealpha=0.95)

    ax.set_title("DriveJudgeBench: Accuracy vs Mean Inference Time",
                 fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Mean Inference Time (seconds)", fontsize=15)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.set_xlim(-0.8, 17.2)
    ax.margins(y=0.12)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
