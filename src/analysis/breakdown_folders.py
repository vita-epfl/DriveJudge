"""E11 -- the per-folder ("transportation") breakdown the paper quotes.

Every clip lives in one of five scenario folders

    negative, crossing_red_lights, crossing_stop, aggressive_takeover, artifacts

and the interesting behaviour of a system is very unevenly spread over them: a
model that answers "Yes, the ego car follows the rules" to everything scores
~100 % on ``negative`` and ~0 % on the three violation folders.  This script
persists that breakdown (with cluster-bootstrap CIs) instead of quoting it from
a notebook, and handles the Left/Right overtaking convention explicitly.

Outputs under ``--out`` (default ``results/stats/``):

    per_folder_question.parquet   model x folder x question accuracy + 95 % CI
    per_folder_category.parquet   model x folder x category accuracy + 95 % CI
    per_folder.tex                compact LaTeX: 3 violation folders x the
                                  traffic-relevant questions
    overtake_convention.parquet   raw vs Left<->Right-swapped accuracy

and ``report/figures/fig_folder_heatmap.{pdf,png}``.

Usage:
    .venv/bin/python src/analysis/breakdown_folders.py --out results/stats/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stats import (  # noqa: E402
    CATEGORY_ORDER,
    MODEL_ORDER,
    cluster_bootstrap,
    load_all_models,
    make_cluster_weights,
)
from run_stats import (  # noqa: E402
    FIG_DIR,
    KIND_COLOR,
    C_INK,
    C_INK2,
    kind_of,
    save,
    setup_style,
    show,
)

FOLDER_ORDER = [
    "negative",
    "crossing_red_lights",
    "crossing_stop",
    "aggressive_takeover",
    "artifacts",
]

VIOLATION_FOLDERS = ["crossing_red_lights", "crossing_stop", "aggressive_takeover"]

Q_FOLLOW = "Is the ego car following the traffic rules ?"
Q_LIGHT_CONSISTENT = "Is the ego car behavior consistent with visible cues of the traffic lights ?"
Q_LIGHTS_RED = "Are the traffic lights red ?"
Q_STOPPED = "Has the ego car stopped ?"
Q_OVERTAKE = "On which side is the ego car overtaking ?"
Q_SAFE = "Is it a safe way of driving for the ego car ?"

#: the questions the compact LaTeX table reports (paper's E11 table).
TRAFFIC_QUESTIONS = [
    Q_FOLLOW,
    Q_LIGHT_CONSISTENT,
    Q_LIGHTS_RED,
    Q_STOPPED,
    Q_OVERTAKE,
    Q_SAFE,
]

Q_SHORT = {
    "Is it a real video or generated ?": "real/gen",
    "Is it a safe way of driving for the ego car ?": "safe driving?",
    "Is the ego car following the traffic rules ?": "follows rules?",
    "Is the ego car on a highway ?": "highway?",
    "Is the ego car behavior consistent with visible cues of the traffic lights ?": "light-consistent?",
    "Are the traffic lights red ?": "lights red?",
    "Has the ego car stopped ?": "stopped?",
    "On which side is the ego car overtaking ?": "overtake side",
    "Do some objects change shape or appearance ?": "shape change?",
    "Does any object appear or disappear without continuous motion ?": "pop in/out?",
}


def short_q(q: str) -> str:
    if q in Q_SHORT:
        return Q_SHORT[q]
    if q.startswith("Overall, how realistic"):
        return "realism 1-3"
    if q.startswith("Overall, how safe"):
        return "safety 1-3"
    return q[:22]


# Sequential blue ramp (single hue, light -> dark) from the reference palette.
BLUE_RAMP = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dj_blue", BLUE_RAMP)


# --------------------------------------------------------------------------


def keep_frequent_questions(df: pl.DataFrame, min_n: int = 20) -> tuple[pl.DataFrame, list[str]]:
    """Drop the handful of one-off questions (n < ``min_n`` benchmark-wide)."""
    per_model = df["model_id"].n_unique()
    counts = df.group_by("question").agg((pl.len() // per_model).alias("n"))
    rare = counts.filter(pl.col("n") < min_n)["question"].to_list()
    return df.filter(~pl.col("question").is_in(rare)), rare


def breakdown(
    df: pl.DataFrame,
    group_cols: list[str],
    cluster_col: str = "scenario_id",
    n_boot: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """model x <group_cols> accuracy with cluster-bootstrap CIs."""
    w = make_cluster_weights(df[cluster_col].unique().to_numpy(), n_boot, seed)
    rows = []
    for m in [x for x in MODEL_ORDER if x in set(df["model_id"].unique())]:
        d = df.filter(pl.col("model_id") == m)
        res = cluster_bootstrap(
            d, cluster_col=cluster_col, weights=w, n_boot=n_boot, seed=seed,
            group_cols=group_cols,
        )
        for r in res.table.iter_rows(named=True):
            rows.append(
                {
                    "model": m,
                    "kind": kind_of(m),
                    **{c: r[c] for c in group_cols},
                    "n": int(r["n"]),
                    "n_clusters": int(r["n_clusters"]),
                    "acc": float(r["estimate"]),
                    "lo": float(r["lo"]),
                    "hi": float(r["hi"]),
                }
            )
    out = pl.DataFrame(rows)
    order = {f: i for i, f in enumerate(FOLDER_ORDER)}
    if "folder" in group_cols:
        out = out.with_columns(
            pl.col("folder").replace_strict(order, default=99, return_dtype=pl.Int64).alias("_o")
        ).sort(["model", "_o", *[c for c in group_cols if c != "folder"]]).drop("_o")
    return out


# ---------------------------------------------------------------- overtake --


def overtake_convention(df: pl.DataFrame) -> pl.DataFrame:
    """Raw vs Left<->Right-swapped accuracy on the overtaking-side question.

    The ground truth is "Right" for 127 of 133 items, so a model that has simply
    learned the mirrored convention (or reports the side of the *overtaken*
    vehicle) would look near-random.  Swapping the two labels in the prediction
    tells the two failure modes apart.
    """
    d = df.filter(pl.col("question") == Q_OVERTAKE)
    swap = {"Left": "Right", "Right": "Left"}
    rows = []
    for m in [x for x in MODEL_ORDER if x in set(d["model_id"].unique())]:
        s = d.filter(pl.col("model_id") == m)
        pred = s["vlm_answer"].to_list()
        gt = s["ground_truth"].to_list()
        swapped = [swap.get(p, p) for p in pred]
        raw = float(np.mean([p == g for p, g in zip(pred, gt)]))
        swp = float(np.mean([p == g for p, g in zip(swapped, gt)]))
        counts: dict[str, int] = {}
        for p in pred:
            counts[p] = counts.get(p, 0) + 1
        rows.append(
            {
                "model": m,
                "n": len(gt),
                "acc_raw": raw,
                "acc_swapped": swp,
                "delta_swap": swp - raw,
                "swap_helps": swp > raw,
                "pred_Left": counts.get("Left", 0),
                "pred_Right": counts.get("Right", 0),
                "pred_not_overtaking": counts.get("It's not overtaking", 0),
                "pred_other": len(gt)
                - counts.get("Left", 0)
                - counts.get("Right", 0)
                - counts.get("It's not overtaking", 0),
                "gt_Right": int(sum(g == "Right" for g in gt)),
                "gt_Left": int(sum(g == "Left" for g in gt)),
            }
        )
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ latex ---


def folder_latex(per_fq: pl.DataFrame) -> str:
    sub = per_fq.filter(
        pl.col("folder").is_in(VIOLATION_FOLDERS) & pl.col("question").is_in(TRAFFIC_QUESTIONS)
    )
    cols = [
        (f, q)
        for f in VIOLATION_FOLDERS
        for q in TRAFFIC_QUESTIONS
        if sub.filter((pl.col("folder") == f) & (pl.col("question") == q)).height
    ]
    lookup = {
        (r["model"], r["folder"], r["question"]): r for r in sub.iter_rows(named=True)
    }
    models = [m for m in MODEL_ORDER if m in set(sub["model"].unique())]

    head = "l" + "c" * len(cols)
    lines = [
        "% generated by src/analysis/breakdown_folders.py -- do not edit by hand",
        "\\begin{tabular}{" + head + "}",
        "\\toprule",
    ]
    # folder header (multicolumn)
    parts, i = [], 0
    while i < len(cols):
        f = cols[i][0]
        span = sum(1 for c in cols if c[0] == f)
        parts.append(
            "\\multicolumn{%d}{c}{\\texttt{%s}}" % (span, f.replace("_", "\\_"))
        )
        i += span
    lines.append("System & " + " & ".join(parts) + " \\\\")
    lines.append(
        "& "
        + " & ".join("\\rotatebox{90}{%s}" % short_q(q).replace("?", "") for _, q in cols)
        + " \\\\"
    )
    n_row = []
    for f, q in cols:
        r = lookup.get((models[0], f, q))
        n_row.append(str(r["n"]) if r else "--")
    lines.append("$n$ & " + " & ".join(n_row) + " \\\\")
    lines.append("\\midrule")
    for m in models:
        cells = []
        for f, q in cols:
            r = lookup.get((m, f, q))
            cells.append("--" if r is None else f"{100 * r['acc']:.1f}")
        name = m.replace("&", "\\&").replace("_", "\\_")
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% accuracy in %, per scenario folder and question.",
        "% 95 % scenario-cluster bootstrap CIs are in results/stats/per_folder_question.parquet.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures ---


def fig_heatmap(per_fq: pl.DataFrame, min_n: int = 20) -> list[Path]:
    import matplotlib.pyplot as plt

    d = per_fq.filter(pl.col("n") >= min_n)
    cols = []
    for f in FOLDER_ORDER:
        qs = (
            d.filter(pl.col("folder") == f)
            .group_by("question")
            .agg(pl.col("n").max().alias("n"))
            .sort("n", descending=True)
        )
        for r in qs.iter_rows(named=True):
            cols.append((f, r["question"], int(r["n"])))
    models = [m for m in MODEL_ORDER if m in set(d["model"].unique())]
    models = models[::-1]

    lookup = {(r["model"], r["folder"], r["question"]): r["acc"] for r in d.iter_rows(named=True)}
    mat = np.full((len(models), len(cols)), np.nan)
    for i, m in enumerate(models):
        for j, (f, q, _) in enumerate(cols):
            v = lookup.get((m, f, q))
            if v is not None:
                mat[i, j] = v

    fig, ax = plt.subplots(figsize=(0.52 * len(cols) + 3.2, 0.42 * len(models) + 2.6))
    im = ax.imshow(mat, cmap=blue_cmap(), vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(models)):
        for j in range(len(cols)):
            if np.isnan(mat[i, j]):
                continue
            ax.text(
                j, i, f"{100 * mat[i, j]:.0f}",
                ha="center", va="center", fontsize=6.5,
                color="white" if mat[i, j] > 0.55 else C_INK,
            )
    ax.set_xticks(
        range(len(cols)),
        [f"{short_q(q)}\nn={n}" for _, q, n in cols],
        rotation=90, fontsize=6.5,
    )
    ax.set_yticks(range(len(models)), models, fontsize=8)
    for lab, m in zip(ax.get_yticklabels(), models):
        lab.set_color(KIND_COLOR[kind_of(m)])
    # folder separators + headers
    bounds, start = [], 0
    for f in FOLDER_ORDER:
        span = sum(1 for c in cols if c[0] == f)
        if span:
            bounds.append((f, start, start + span))
            start += span
    for _, s, e in bounds[:-1]:
        ax.axvline(e - 0.5, color="white", linewidth=2.4)
    for f, s, e in bounds:
        ax.text(
            (s + e - 1) / 2, -0.95, f, ha="center", va="bottom", fontsize=8, color=C_INK
        )
    ax.set_ylim(len(models) - 0.5, -1.1)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cb.set_label("accuracy", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, color=C_INK2)
    ax.set_title(
        "Accuracy (%) per scenario folder and question", loc="left", pad=26
    )
    fig.tight_layout()
    return save(fig, "fig_folder_heatmap")


# -------------------------------------------------------------------- main --


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/stats/")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", default="scenario_id", choices=["scenario_id", "video_id"])
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_all_models(args.results_dir)
    kept, rare = keep_frequent_questions(df, args.min_n)
    print(f"folders: {sorted(set(df['folder'].unique()))}")
    print(f"dropped {len(rare)} question(s) with n < {args.min_n} benchmark-wide:")
    for q in rare:
        n = df.filter(pl.col("question") == q).height // df["model_id"].n_unique()
        print(f"    n={n:2d}  {q}")
    print(f"kept {kept.height} of {df.height} model-item rows "
          f"({kept.height // df['model_id'].n_unique()} items per system)")

    per_fq = breakdown(
        kept, ["folder", "question"], args.cluster, args.n_boot, args.seed
    )
    per_fc = breakdown(
        kept, ["folder", "category"], args.cluster, args.n_boot, args.seed
    )
    per_fq.write_parquet(out / "per_folder_question.parquet")
    per_fc.write_parquet(out / "per_folder_category.parquet")
    (out / "per_folder.tex").write_text(folder_latex(per_fq))

    oc = overtake_convention(df)
    oc.write_parquet(out / "overtake_convention.parquet")

    figs: list[Path] = []
    if not args.no_figures:
        setup_style()
        figs += fig_heatmap(per_fq, args.min_n)

    # ---------------- overtaking convention ----------------
    n_help = int(oc["swap_helps"].sum())
    print("\n" + "=" * 72)
    print(f"overtaking side ({Q_OVERTAKE!r}) -- GT is Right in "
          f"{int(oc['gt_Right'][0])}/{int(oc['n'][0])} items")
    print("=" * 72)
    print(show(oc.select(
        "model", "n", "acc_raw", "acc_swapped", "delta_swap",
        "pred_Left", "pred_Right", "pred_not_overtaking", "pred_other",
    )))
    print(
        f"\nSwapping Left<->Right in the predictions helps {n_help} of "
        f"{oc.height} models"
        + (
            f" (mean gain {100 * float(oc.filter(pl.col('swap_helps'))['delta_swap'].mean()):+.1f} pp"
            f" for those {n_help}); "
            if n_help
            else "; "
        )
        + "i.e. the near-zero raw scores are "
        + ("largely a mirrored-convention artefact." if n_help >= oc.height / 2
           else "not explained by a mirrored convention.")
    )

    # ---------------- sanity numbers quoted in report.tex ----------------
    print("\n" + "=" * 72)
    print("sanity numbers (accuracy %, 95 % scenario-cluster CI)")
    print("=" * 72)
    checks = [
        ("aggressive_takeover", Q_FOLLOW, "report.tex quotes 5.6 / 5.6 / 85.6 for Qwen3-VL / v4o / v4p"),
        ("crossing_red_lights", Q_FOLLOW, ""),
        ("crossing_red_lights", Q_LIGHT_CONSISTENT, ""),
        ("crossing_red_lights", Q_LIGHTS_RED, ""),
        ("crossing_stop", Q_FOLLOW, ""),
        ("crossing_stop", Q_STOPPED, ""),
        ("aggressive_takeover", Q_OVERTAKE, ""),
    ]
    for folder, q, note in checks:
        sub = per_fq.filter((pl.col("folder") == folder) & (pl.col("question") == q))
        if not sub.height:
            continue
        print(f"\n{folder} / {q}   (n={int(sub['n'][0])})"
              + (f"\n    [{note}]" if note else ""))
        for m in ["Qwen3-VL", "Qwen3-VL+agent", "Qwen3-VL+agent+hints",
                  "GPT-5.4-mini", "Gemini-3-Flash"]:
            r = sub.filter(pl.col("model") == m)
            if not r.height:
                continue
            r = r.row(0, named=True)
            tag = {"Qwen3-VL": "baseline", "Qwen3-VL+agent": "v4o",
                   "Qwen3-VL+agent+hints": "v4p"}.get(m, "")
            print(f"    {m:22s} {tag:9s} {100 * r['acc']:5.1f} "
                  f"[{100 * r['lo']:5.1f}, {100 * r['hi']:5.1f}]")

    print("\nper-folder overall accuracy (all questions, all categories):")
    per_f = breakdown(kept, ["folder"], args.cluster, args.n_boot, args.seed)
    piv = per_f.select("model", "folder", "acc").pivot(
        on="folder", index="model", values="acc"
    )
    cols = [c for c in FOLDER_ORDER if c in piv.columns]
    print(show(piv.select("model", *cols)))

    print("\nwrote:")
    for name in ["per_folder_question.parquet", "per_folder_category.parquet",
                 "per_folder.tex", "overtake_convention.parquet"]:
        print(f"  {out / name}")
    for p in figs:
        print(f"  {p}")
    print(f"\nruntime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
