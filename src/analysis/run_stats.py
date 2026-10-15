"""Cluster-bootstrap statistics for every DriveJudgeBench system.

Produces, under ``--out`` (default ``results/stats/``):

  overall_ci.parquet       micro + macro accuracy with scenario-cluster CIs,
                           plus video-cluster / i.i.d. / Wilson CIs (appendix)
  per_category_ci.parquet  model x category accuracy with cluster CIs
  per_source_ci.parquet    model x dataset accuracy with cluster CIs
  pairwise_diff_ci.parquet paired accuracy differences for all ordered pairs
  per_class_metrics.parquet per-question recall / precision / F1 / MCC / ...
  stats_summary.json       the same numbers, nested and human-readable
  tab_overall_ci.tex       LaTeX table (acc [lo, hi], macro [lo, hi])

and, under ``report/figures/`` (PDF + PNG):

  fig_overall_ci, fig_per_category_ci, fig_pairwise_vs_v4o

Usage:
    .venv/bin/python src/analysis/run_stats.py \
        --results-dir results --n-boot 2000 --seed 0 \
        --cluster scenario_id --out results/stats/
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

try:
    from stats import (  # type: ignore
        CATEGORY_ORDER,
        CATEGORY_SHORT,
        CLOSED_SOURCE,
        AGENT_MODELS,
        MODEL_ORDER,
        BootResult,
        cluster_bootstrap,
        iid_bootstrap_proportion,
        class_metrics,
        load_all_models,
        macro_accuracy,
        make_cluster_weights,
        paired_diff_ci,
        wilson_ci,
    )
except ImportError:  # running from the repo root
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stats import (  # type: ignore
        CATEGORY_ORDER,
        CATEGORY_SHORT,
        CLOSED_SOURCE,
        AGENT_MODELS,
        MODEL_ORDER,
        BootResult,
        cluster_bootstrap,
        iid_bootstrap_proportion,
        class_metrics,
        load_all_models,
        macro_accuracy,
        make_cluster_weights,
        paired_diff_ci,
        wilson_ci,
    )

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures"

REF_V4O = "Qwen3-VL+agent"
REF_GEMINI = "Gemini-3-Flash"

# ---------------------------------------------------------------- plotting --
# Colourblind-safe categorical slots 1-3 of the validated reference palette
# (validated all-pairs, light mode: worst CVD dE 9.2, worst normal-vision 24.0).
C_BASELINE = "#2a78d6"  # blue
C_AGENT = "#eb6834"  # orange
C_CLOSED = "#1baf7a"  # aqua  (below 3:1 on white -> always direct-labelled)
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#e2e1de"


def kind_of(model: str) -> str:
    if model in CLOSED_SOURCE:
        return "closed"
    return "agent" if model in AGENT_MODELS else "baseline"


KIND_COLOR = {"baseline": C_BASELINE, "agent": C_AGENT, "closed": C_CLOSED}
KIND_HATCH = {"baseline": "", "agent": "", "closed": "///"}
KIND_LABEL = {
    "baseline": "open-source VLM",
    "agent": "agentic (ours)",
    "closed": "closed-source (API)",
}


def setup_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": C_GRID,
            "axes.labelcolor": C_INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": C_GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "text.color": C_INK,
            "xtick.color": C_INK2,
            "ytick.color": C_INK2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "hatch.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def show(df: pl.DataFrame, max_rows: int = 60, width: int = 220) -> str:
    """Render a polars frame as a plain table (no pyarrow / pandas needed)."""
    with pl.Config(
        tbl_rows=max_rows,
        tbl_cols=30,
        tbl_width_chars=width,
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        float_precision=4,
        fmt_str_lengths=60,
    ):
        return str(df)


def save(fig, name: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p)
        out.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


# ------------------------------------------------------------------ tables --


def _row(res: BootResult) -> dict:
    r = res.table.row(0, named=True)
    return {"n": int(r["n"]), "n_clusters": int(r["n_clusters"]), "acc": float(r["estimate"]),
            "lo": float(r["lo"]), "hi": float(r["hi"])}


def compute_all(
    df: pl.DataFrame,
    cluster_col: str = "scenario_id",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    models = [m for m in MODEL_ORDER if m in set(df["model_id"].unique())]

    # One shared weight draw per clustering, reused by every model / group so
    # that all replicates describe the same resampled worlds.
    w_scn = make_cluster_weights(df[cluster_col].unique().to_numpy(), n_boot, seed)
    w_vid = make_cluster_weights(df["video_id"].unique().to_numpy(), n_boot, seed)

    overall_rows, cat_rows, src_rows = [], [], []
    for m in models:
        d = df.filter(pl.col("model_id") == m)
        k, n = int(d["correct"].sum()), d.height

        ov = cluster_bootstrap(d, cluster_col=cluster_col, weights=w_scn, n_boot=n_boot, seed=seed)
        per_cat, macro = macro_accuracy(
            d, cluster_col=cluster_col, weights=w_scn, n_boot=n_boot, seed=seed
        )
        ov_vid = cluster_bootstrap(d, cluster_col="video_id", weights=w_vid, n_boot=n_boot, seed=seed)
        _, ilo, ihi, _ = iid_bootstrap_proportion(k, n, n_boot=n_boot, seed=seed)
        wlo, whi = wilson_ci(k, n)
        per_src = cluster_bootstrap(
            d, cluster_col=cluster_col, weights=w_scn, n_boot=n_boot, seed=seed,
            group_cols=["source"],
        )

        o = _row(ov)
        overall_rows.append(
            {
                "model": m,
                "kind": kind_of(m),
                "n": o["n"],
                "n_correct": k,
                "acc": o["acc"],
                "lo": o["lo"],
                "hi": o["hi"],
                "ci_width": o["hi"] - o["lo"],
                "n_clusters_scenario": o["n_clusters"],
                "macro_acc": macro.estimate,
                "macro_lo": macro.lo,
                "macro_hi": macro.hi,
                # appendix: alternative uncertainty models
                "lo_video": _row(ov_vid)["lo"],
                "hi_video": _row(ov_vid)["hi"],
                "ci_width_video": _row(ov_vid)["hi"] - _row(ov_vid)["lo"],
                "n_clusters_video": _row(ov_vid)["n_clusters"],
                "lo_iid": ilo,
                "hi_iid": ihi,
                "ci_width_iid": ihi - ilo,
                "lo_wilson": wlo,
                "hi_wilson": whi,
                "ci_width_wilson": whi - wlo,
                # design effect: how much wider the cluster CI is than the
                # i.i.d. one, on the variance scale
                "deff_scenario": ((o["hi"] - o["lo"]) / (ihi - ilo)) ** 2,
                "deff_video": ((_row(ov_vid)["hi"] - _row(ov_vid)["lo"]) / (ihi - ilo)) ** 2,
            }
        )
        for r in per_cat.table.iter_rows(named=True):
            cat_rows.append(
                {
                    "model": m,
                    "kind": kind_of(m),
                    "category": r["category"],
                    "n": int(r["n"]),
                    "n_clusters": int(r["n_clusters"]),
                    "acc": float(r["estimate"]),
                    "lo": float(r["lo"]),
                    "hi": float(r["hi"]),
                }
            )
        for r in per_src.table.iter_rows(named=True):
            src_rows.append(
                {
                    "model": m,
                    "kind": kind_of(m),
                    "source": r["source"],
                    "n": int(r["n"]),
                    "n_clusters": int(r["n_clusters"]),
                    "acc": float(r["estimate"]),
                    "lo": float(r["lo"]),
                    "hi": float(r["hi"]),
                }
            )

    overall = pl.DataFrame(overall_rows)
    per_category = pl.DataFrame(cat_rows)
    per_source = pl.DataFrame(src_rows)

    # ---- pairwise ----
    pair_rows = []
    for a in models:
        for b in models:
            if a == b:
                continue
            r = paired_diff_ci(
                df, a, b, cluster_col=cluster_col, weights=w_scn, n_boot=n_boot, seed=seed
            )
            r["is_vs_v4o"] = b == REF_V4O
            r["is_vs_gemini"] = b == REF_GEMINI
            r["significant"] = (r["lo"] > 0) or (r["hi"] < 0)
            pair_rows.append(r)
    pairwise = pl.DataFrame(pair_rows)

    return {
        "overall": overall,
        "per_category": per_category,
        "per_source": per_source,
        "pairwise": pairwise,
        "weights": {"scenario": w_scn, "video": w_vid},
        "models": models,
    }


def latex_table(overall: pl.DataFrame) -> str:
    d = overall.sort("acc", descending=True)
    lines = [
        "% generated by src/analysis/run_stats.py -- do not edit by hand",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "System & $n$ & Accuracy [95\\% CI] & Macro accuracy [95\\% CI] \\\\",
        "\\midrule",
    ]
    for r in d.iter_rows(named=True):
        name = r["model"].replace("&", "\\&").replace("_", "\\_")
        if r["kind"] == "closed":
            name += "$^{\\dagger}$"
        lines.append(
            f"{name} & {r['n']} & "
            f"{r['acc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] & "
            f"{r['macro_acc']:.3f} [{r['macro_lo']:.3f}, {r['macro_hi']:.3f}] \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% CIs: percentile cluster bootstrap (2000 replicates) resampling base",
        "% scenarios (weather re-renderings of one scenario form one cluster).",
        "% $^{\\dagger}$ closed-source, evaluated on the 7282-item shared subset.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- figures --


def fig_overall(overall: pl.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = overall.sort("acc")
    y = np.arange(d.height)
    acc = d["acc"].to_numpy()
    err = np.vstack([acc - d["lo"].to_numpy(), d["hi"].to_numpy() - acc])

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    for i, r in enumerate(d.iter_rows(named=True)):
        ax.barh(
            i,
            r["acc"],
            height=0.66,
            color=KIND_COLOR[r["kind"]],
            hatch=KIND_HATCH[r["kind"]],
            edgecolor="white",
            linewidth=0.0,
        )
    ax.errorbar(
        acc, y, xerr=err, fmt="none", ecolor=C_INK, elinewidth=1.1, capsize=3, capthick=1.1
    )
    for i, (a, h) in enumerate(zip(acc, d["hi"].to_numpy())):
        ax.text(h + 0.012, i, f"{a:.3f}", va="center", ha="left", fontsize=8, color=C_INK2)
    ax.set_yticks(y, d["model"].to_list())
    ax.set_xlabel("Overall accuracy (95 % cluster-bootstrap CI)")
    ax.set_xlim(0, max(0.75, float(d["hi"].max()) + 0.09))
    ax.set_title("DriveJudgeBench: accuracy with scenario-cluster confidence intervals", pad=10)
    handles = [
        Patch(facecolor=KIND_COLOR[k], hatch=KIND_HATCH[k], edgecolor="white", label=KIND_LABEL[k])
        for k in ("baseline", "agent", "closed")
    ]
    ax.legend(handles=handles, loc="lower right", ncols=1)
    fig.tight_layout()
    return save(fig, "fig_overall_ci")


def fig_per_category(per_category: pl.DataFrame, overall: pl.DataFrame) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = overall.sort("acc")["model"].to_list()
    kinds = {r["model"]: r["kind"] for r in overall.iter_rows(named=True)}

    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.4), sharex=True)
    for ax, cat in zip(axes.ravel(), CATEGORY_ORDER):
        sub = per_category.filter(pl.col("category") == cat)
        m = {r["model"]: r for r in sub.iter_rows(named=True)}
        y = np.arange(len(order))
        acc = np.array([m[k]["acc"] if k in m else np.nan for k in order])
        lo = np.array([m[k]["lo"] if k in m else np.nan for k in order])
        hi = np.array([m[k]["hi"] if k in m else np.nan for k in order])
        ax.set_axisbelow(True)
        ax.xaxis.grid(True)
        for i, k in enumerate(order):
            if k not in m:
                continue
            ax.barh(
                i, m[k]["acc"], height=0.66, color=KIND_COLOR[kinds[k]],
                hatch=KIND_HATCH[kinds[k]], edgecolor="white", linewidth=0.0,
            )
        ax.errorbar(
            acc, y, xerr=np.vstack([acc - lo, hi - acc]), fmt="none",
            ecolor=C_INK, elinewidth=0.9, capsize=2, capthick=0.9,
        )
        n = int(sub["n"].max()) if sub.height else 0
        ax.set_title(f"{CATEGORY_SHORT[cat]}  (n={n})", loc="left")
        ax.set_yticks(y, order if ax in axes[:, 0] else [""] * len(order))
        ax.set_xlim(0, 1.0)
    for ax in axes[1, :]:
        ax.set_xlabel("Accuracy")
    handles = [
        Patch(facecolor=KIND_COLOR[k], hatch=KIND_HATCH[k], edgecolor="white", label=KIND_LABEL[k])
        for k in ("baseline", "agent", "closed")
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Per-category accuracy with 95 % scenario-cluster confidence intervals", y=1.0
    )
    fig.tight_layout()
    return save(fig, "fig_per_category_ci")


def fig_forest(pairwise: pl.DataFrame, reference: str = REF_V4O) -> list[Path]:
    import matplotlib.pyplot as plt

    d = pairwise.filter(pl.col("B") == reference).sort("diff")
    y = np.arange(d.height)
    diff = d["diff"].to_numpy()
    lo, hi = d["lo"].to_numpy(), d["hi"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.axvline(0.0, color=C_INK, linewidth=1.0, zorder=1)
    colors = [KIND_COLOR[kind_of(m)] for m in d["A"].to_list()]
    ax.errorbar(
        diff, y, xerr=np.vstack([diff - lo, hi - diff]), fmt="none",
        ecolor=C_INK2, elinewidth=1.1, capsize=3, capthick=1.1, zorder=2,
    )
    ax.scatter(diff, y, s=44, c=colors, edgecolors="white", linewidths=1.2, zorder=3)
    for i, r in enumerate(d.iter_rows(named=True)):
        ax.text(
            r["hi"] + 0.012, i,
            f"{r['diff']:+.3f}" + ("" if r["significant"] else " (n.s.)"),
            va="center", ha="left", fontsize=8, color=C_INK2,
        )
    ax.set_yticks(y, d["A"].to_list())
    ax.set_xlabel(f"$\\Delta$ accuracy vs {reference}  (paired, 95 % cluster CI)")
    ax.set_xlim(float(lo.min()) - 0.06, float(hi.max()) + 0.14)
    ax.set_title(f"Paired accuracy difference against {reference}", pad=8)
    fig.tight_layout()
    return save(fig, "fig_pairwise_vs_v4o")


# -------------------------------------------------------------------- main --


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", default="scenario_id", choices=["scenario_id", "video_id"])
    ap.add_argument("--out", default="results/stats/")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_all_models(args.results_dir)
    print(f"loaded {df.height} rows / {df['model_id'].n_unique()} systems "
          f"from {args.results_dir}")

    # ---- data checks -----------------------------------------------------
    keys = ["source", "video", "question"]
    sizes = df.group_by("model_id").agg(pl.len().alias("n")).sort("n")
    ref = df.filter(pl.col("model_id") == "Qwen3-VL").select(keys)
    print("join coverage against the 7371-item in-house grid:")
    for m in sorted(set(df["model_id"].unique())):
        s = df.filter(pl.col("model_id") == m).select(keys)
        shared = s.join(ref, on=keys, how="inner").height
        print(f"  {m:24s} n={s.height:5d}  shared with Qwen3-VL={shared:5d}"
              f"  unmatched={s.height - shared}")
    bad = df.filter(pl.col("question").str.contains("answer is one of the following"))
    print(f"question strings still carrying the option suffix: {bad.height}")
    print(f"item-key duplicates: "
          f"{df.height - df.select(['model_id', *keys]).n_unique()}")
    print(f"scenario clusters: {df['scenario_id'].n_unique()} "
          f"(videos: {df['video_id'].n_unique()})")
    cats = sorted(set(df["category"].unique()))
    print(f"categories ({len(cats)}): {cats}")
    print(f"sizes: {sizes.to_dicts()}")

    # ---- statistics ------------------------------------------------------
    res = compute_all(df, cluster_col=args.cluster, n_boot=args.n_boot, seed=args.seed)
    overall = res["overall"]
    per_category = res["per_category"]
    per_source = res["per_source"]
    pairwise = res["pairwise"]

    cm = class_metrics(df, group_cols=["model_id"])

    overall.write_parquet(out / "overall_ci.parquet")
    per_category.write_parquet(out / "per_category_ci.parquet")
    per_source.write_parquet(out / "per_source_ci.parquet")
    pairwise.write_parquet(out / "pairwise_diff_ci.parquet")
    cm.write_parquet(out / "per_class_metrics.parquet")
    (out / "tab_overall_ci.tex").write_text(latex_table(overall))

    summary = {
        "config": {
            "results_dir": str(args.results_dir),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "cluster": args.cluster,
            "n_scenario_clusters": int(df["scenario_id"].n_unique()),
            "n_video_clusters": int(df["video_id"].n_unique()),
            "category_order": CATEGORY_ORDER,
        },
        "models": {},
        "pairwise_vs_reference": {
            REF_V4O: [
                r for r in pairwise.filter(pl.col("B") == REF_V4O).sort("diff").to_dicts()
            ],
            REF_GEMINI: [
                r for r in pairwise.filter(pl.col("B") == REF_GEMINI).sort("diff").to_dicts()
            ],
        },
    }
    for r in overall.sort("acc", descending=True).iter_rows(named=True):
        m = r["model"]
        summary["models"][m] = {
            **{k: v for k, v in r.items() if k != "model"},
            "per_category": per_category.filter(pl.col("model") == m)
            .drop("model", "kind")
            .to_dicts(),
            "per_source": per_source.filter(pl.col("model") == m)
            .drop("model", "kind")
            .to_dicts(),
        }
    (out / "stats_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figures ---------------------------------------------------------
    figs: list[Path] = []
    if not args.no_figures:
        setup_style()
        figs += fig_overall(overall)
        figs += fig_per_category(per_category, overall)
        figs += fig_forest(pairwise, REF_V4O)

    # ---- console report --------------------------------------------------
    print("\noverall accuracy with 95 % scenario-cluster CIs")
    print(
        show(
            overall.sort("acc", descending=True).select(
                "model", "n", "acc", "lo", "hi", "macro_acc", "macro_lo", "macro_hi"
            )
        )
    )

    v = overall.filter(pl.col("model") == REF_V4O).row(0, named=True)
    print(
        f"\n{REF_V4O} CI widths -- "
        f"iid {v['ci_width_iid']:.4f} "
        f"(Wilson {v['ci_width_wilson']:.4f}) <= "
        f"video-cluster {v['ci_width_video']:.4f} <= "
        f"scenario-cluster {v['ci_width']:.4f}"
    )
    print("\n95 % CI width under three independence assumptions (+ design effect):")
    print(
        show(
            overall.sort("ci_width", descending=True).select(
                "model",
                pl.col("ci_width_iid").alias("w_iid"),
                pl.col("ci_width_video").alias("w_video"),
                pl.col("ci_width").alias("w_scenario"),
                "deff_video",
                pl.col("deff_scenario").alias("deff_scen"),
            )
        )
    )
    n_mono = int(
        ((overall["ci_width_iid"] <= overall["ci_width_video"])
         & (overall["ci_width_video"] <= overall["ci_width"])).sum()
    )
    print(
        f"iid <= video <= scenario holds for {n_mono}/{overall.height} systems; "
        f"scenario >= video for "
        f"{int((overall['ci_width'] >= overall['ci_width_video']).sum())}/{overall.height}, "
        f"scenario >= iid for "
        f"{int((overall['ci_width'] >= overall['ci_width_iid']).sum())}/{overall.height}."
    )

    d = pairwise.filter((pl.col("A") == REF_V4O) & (pl.col("B") == "GPT-5.4-mini")).row(
        0, named=True
    )
    print(
        f"{REF_V4O} - GPT-5.4-mini: diff={d['diff']:+.4f} "
        f"CI=[{d['lo']:+.4f}, {d['hi']:+.4f}] p_boot={d['p_boot']:.4g} "
        f"mcnemar_p={d['mcnemar_p']:.3g} n_shared={d['n_shared']} "
        f"excludes 0: {d['lo'] > 0 or d['hi'] < 0}"
    )

    ns = pairwise.filter(~pl.col("significant"))
    print(f"\nordered pairs whose 95 % CI covers 0: {ns.height}/{pairwise.height}")
    if ns.height:
        print(
            show(
                ns.filter(pl.col("A") < pl.col("B")).select(
                    "A", "B", "diff", "lo", "hi", "p_boot", "mcnemar_p"
                )
            )
        )

    up = (
        cm.filter(pl.col("class") == "UNPARSED")
        .group_by("model_id")
        .agg(pl.col("n_predicted").sum().alias("unparsed"), pl.col("n").sum().alias("n_scored"))
        .sort("unparsed", descending=True)
    )
    print("\nunparsed predictions (scored questions only):")
    print(show(up))

    print("\nwrote:")
    for p in sorted(out.glob("*")):
        print(f"  {p}")
    for p in figs:
        print(f"  {p}")
    print(f"\nruntime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
