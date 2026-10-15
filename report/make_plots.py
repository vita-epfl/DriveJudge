"""Generate two figures for the cross-model report:
  fig_overall_accuracy.pdf  — bar chart of overall accuracy (10 systems)
  fig_per_category.pdf      — heatmap of per-category accuracy (10 x 6)
All ten systems — the eight in-house ones plus GPT-5.4-mini and Gemini-3-Flash —
come from `report/all_models_accuracy_summary.json` (regenerate it with
`report/run_analysis.py`). Figures are written to `report/figures/`.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CATS = [
    "Visual understanding",
    "Spatial-temporal understanding",
    "Artifacts understanding",
    "Safety understanding",
    "Reality understanding",
    "Traffic laws understanding",
]
CAT_SHORT = ["Visual", "Spat-temp", "Artifacts", "Safety", "Reality", "Traffic"]

# Model ids that are closed-source (everything else is open-source / agentic)
CLOSED = {"GPT-5.4-mini", "Gemini-3-Flash"}


def kind_of(mid: str) -> str:
    if mid in CLOSED:
        return "closed"
    return "agent" if "agent" in mid else "baseline"


# Load all ten systems from the single cross-model summary
with open(OUT / "all_models_accuracy_summary.json") as f:
    s = json.load(f)

rows = []
for mid, d in s["models"].items():
    pc = {r["category"]: r["accuracy"] for r in d["per_category"]}
    rows.append({
        "model": mid,
        "overall": d["overall_accuracy"],
        "per_cat": [pc.get(c, float("nan")) for c in CATS],
        "kind": kind_of(mid),
    })

rows.sort(key=lambda r: r["overall"])

# ---------------- fig 1: overall accuracy bar ----------------
fig, ax = plt.subplots(figsize=(8.4, 4.0))
xs = np.arange(len(rows))
colors = {"baseline": "#7f8fa6", "agent": "#e1872c", "closed": "#2c5fa0"}
bar_colors = [colors[r["kind"]] for r in rows]
acc = [r["overall"] for r in rows]
labels = [r["model"] for r in rows]
ax.barh(xs, acc, color=bar_colors)
ax.set_yticks(xs)
ax.set_yticklabels(labels)
for x, a in zip(xs, acc):
    ax.text(a + 0.005, x, f"{a*100:.1f}", va="center", fontsize=9)
ax.set_xlim(0, 0.75)
ax.set_xlabel("Overall accuracy")
ax.set_title("Overall accuracy across 10 model configurations "
             "(open-source n = 7 371; closed-source n = 7 282)")
ax.grid(True, axis="x", linestyle=":", alpha=0.6)
handles = [
    mpatches.Patch(color=colors["baseline"], label="open-source baseline"),
    mpatches.Patch(color=colors["agent"], label="agentic Qwen3-VL"),
    mpatches.Patch(color=colors["closed"], label="closed-source"),
]
ax.legend(handles=handles, loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_overall_accuracy.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_overall_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------- fig 2: per-category heatmap ----------------
mat = np.array([r["per_cat"] for r in rows]) * 100
fig, ax = plt.subplots(figsize=(8.4, 4.4))
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_xticks(range(len(CAT_SHORT)))
ax.set_xticklabels(CAT_SHORT)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["model"] for r in rows])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                color="black" if 25 < v < 75 else "white", fontsize=9)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("Accuracy (%)")
ax.set_title("Per-category accuracy")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_per_category.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "fig_per_category.png", dpi=150, bbox_inches="tight")
plt.close()

print("Wrote:")
print(f"  {FIG_DIR}/fig_overall_accuracy.{{pdf,png}}")
print(f"  {FIG_DIR}/fig_per_category.{{pdf,png}}")
