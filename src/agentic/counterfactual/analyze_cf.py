"""E10 counterfactual analysis: does the agent's answer depend on its tools?

    .venv/bin/python src/agentic/counterfactual/analyze_cf.py \
        --runs-dir results/counterfactual --n-boot 2000

Inputs
------
``--runs-dir/*.json``   result files written by ``main_counterfactual.py``
    (agent shape ``{video, question, evaluation, answer, time_seconds,
    history, cf_modes, cf_tool_text}``).  A run whose ``cf_modes`` are all
    ``real`` is a *baseline*; with two baselines the second one measures how
    much the agent flips **on its own** at temperature 0.3.
``subsets/cf_subsets_gt.json``  ground truth, category, folder, scenario_id and
    the v4o outcome for every subset item (written by ``build_cf_subset.py``).

Runs are joined on ``(video, question)``.

Metrics (all with ``stats.cluster_bootstrap``, cluster = ``scenario_id``, so the
eight weather re-renderings of one scene count as one independent unit)
-----------------------------------------------------------------------------
``accuracy``              accuracy per run, overall and per subset.
``flip_rate``             P(answer != baseline answer).
``self_flip``             the same between the two baseline runs -- the noise
                          floor that any flip rate has to beat.
``net_flip``              flip_rate - self_flip, bootstrapped *per item* so the
                          CI accounts for the pairing.
``verdict_following``     P(answer == the FFT verdict hint actually shown), on
                          the "real or generated" question; comparing ``real``
                          with ``reverse`` says whether the agent follows the
                          number or the pixels.
``tool_citation_rate``    P(the ``evaluation`` mentions a tool at all).
``cond_flip_cites`` /     P(flip | tool reversed and the explanation cites the
``cond_flip_no_cites``    tool) vs P(flip | reversed and it does not), and their
``cond_flip_diff``        difference -- the conditional-faithfulness read-out.

Outputs
-------
``results/counterfactual/cf_metrics.parquet``  every estimate, long format
``results/counterfactual/cf_summary.json``     the same, nested
``results/counterfactual/cf_table.tex``        LaTeX table for the appendix
``report/figures/fig_counterfactual_flips.{pdf,png}``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(_HERE))

import cf_tools  # noqa: E402
from analysis.run_stats import C_AGENT, C_BASELINE, C_CLOSED, C_INK, C_INK2, setup_style  # noqa: E402
from analysis.stats import (  # noqa: E402
    OPTION_SUFFIX_RE,
    cluster_bootstrap,
    make_cluster_weights,
)

DEFAULT_RUNS = REPO / "results" / "counterfactual"
DEFAULT_GT = _HERE / "subsets" / "cf_subsets_gt.json"
DEFAULT_FIGDIR = REPO / "report" / "figures"

VERDICT_RE = re.compile(r"VERDICT HINT:\s*(GENERATED|REAL)", re.IGNORECASE)
Q_REALGEN = "Is it a real video or generated ?"

CHANNELS = ("fft", "sam", "raft")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def norm_video(video: str) -> tuple[str, str]:
    """absolute pod path -> (source, '<folder>/<file>.mp4')."""
    rel = video.split("final_dataset/", 1)[-1].lstrip("/")
    parts = rel.split("/")
    if parts and parts[0] in ("cosmos-drive-dreams", "cosmos-predict1", "cosmos-predict2"):
        return parts[0], "/".join(parts[1:])
    return "unknown", rel


def norm_question(text: str) -> str:
    return re.sub(OPTION_SUFFIX_RE, "", text).strip()


def fft_hint(record: dict) -> str | None:
    """The FFT verdict hint the VLM actually saw ('Generated' / 'Real' / None)."""
    texts = list((record.get("cf_tool_text") or {}).get("get_frequency_analysis") or [])
    if not texts:
        for h in record.get("history") or []:
            if h.get("name") == "get_frequency_analysis" and h.get("output"):
                texts.append(h["output"])
    for text in reversed(texts):
        m = VERDICT_RE.search(text or "")
        if m:
            return m.group(1).capitalize()
    return None


def load_runs(runs_dir: Path, gt: dict) -> pl.DataFrame:
    """One long frame: a row per (run, item)."""
    rows = []
    files = sorted(p for p in runs_dir.glob("*.json")
                   if p.name not in {"cf_summary.json"})
    if not files:
        raise SystemExit(f"no run files under {runs_dir}")
    for path in files:
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict) or "answer" not in rec:
                continue
            modes = rec.get("cf_modes") or {}
            source, video = norm_video(rec["video"])
            question = norm_question(rec["question"])
            key = f"{source}/{video}||{question}"
            meta = gt.get(key)
            if meta is None:
                continue
            evaluation = rec.get("evaluation") or ""
            answer = (rec.get("answer") or "").strip()
            hint = fft_hint(rec)
            rows.append({
                "run": modes.get("run_label") or path.stem,
                "file": path.name,
                "fft_mode": modes.get("fft", "real"),
                "sam_mode": modes.get("sam", "real"),
                "raft_mode": modes.get("raft", "real"),
                "key": key,
                "source": source,
                "video": video,
                "question": question,
                "subset": meta["subset"],
                "category": meta["category"],
                "folder": meta["folder"],
                "scenario_id": meta["scenario_id"],
                "ground_truth": meta["ground_truth"],
                "v4o_correct": meta["v4o_correct"],
                "answer": answer,
                "correct": answer == meta["ground_truth"],
                "cites_tool": bool(cf_tools.TOOL_CITATION_RE.search(evaluation)),
                "fft_hint": hint,
                "follows_hint": (hint is not None and question == Q_REALGEN
                                 and answer == hint),
                "hint_applicable": hint is not None and question == Q_REALGEN,
                "time_seconds": float(rec.get("time_seconds") or 0.0),
            })
    if not rows:
        raise SystemExit("no records joined against the ground truth -- wrong --gt file?")
    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("fft_mode").eq("real") & pl.col("sam_mode").eq("real")
         & pl.col("raft_mode").eq("real")).alias("is_baseline")
    )
    return df


def run_descriptor(df: pl.DataFrame) -> pl.DataFrame:
    """One row per run: its modes, the perturbed channel, size."""
    d = (df.group_by("run")
           .agg(pl.col("fft_mode").first(), pl.col("sam_mode").first(),
                pl.col("raft_mode").first(), pl.col("is_baseline").first(),
                pl.col("file").first(), pl.len().alias("n"),
                pl.col("subset").n_unique().alias("n_subsets"))
           .sort("run"))
    channel, mode = [], []
    for r in d.iter_rows(named=True):
        perturbed = [(c, r[f"{c}_mode"]) for c in CHANNELS if r[f"{c}_mode"] != "real"]
        if not perturbed:
            channel.append("none")
            mode.append("real")
        else:
            channel.append("+".join(c for c, _ in perturbed))
            mode.append("+".join(m for _, m in perturbed))
    return d.with_columns(pl.Series("channel", channel), pl.Series("mode", mode))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def boot_rows(res, metric: str, group_cols: list[str], extra: dict) -> list[dict]:
    out = []
    for r in res.table.iter_rows(named=True):
        row = {"metric": metric, **extra}
        for c in group_cols:
            row[c] = r[c]
        row.update({"n": int(r["n"]), "n_clusters": int(r["n_clusters"]),
                    "estimate": float(r["estimate"]), "lo": float(r["lo"]),
                    "hi": float(r["hi"])})
        out.append(row)
    return out


def diff_ci(res, i: int, j: int, conf: float = 0.95) -> tuple[float, float, float]:
    """estimate / lo / hi of group ``i`` minus group ``j`` from shared replicates."""
    reps = res.replicates[:, i] - res.replicates[:, j]
    a = (1 - conf) / 2 * 100
    est = float(res.table["estimate"][i] - res.table["estimate"][j])
    return est, float(np.nanpercentile(reps, a)), float(np.nanpercentile(reps, 100 - a))


def analyse(df: pl.DataFrame, n_boot: int, seed: int) -> tuple[pl.DataFrame, dict]:
    runs = run_descriptor(df)
    weights = make_cluster_weights(df["scenario_id"].unique().to_numpy(), n_boot, seed)
    rows: list[dict] = []
    summary: dict = {"runs": runs.to_dicts(), "metrics": {}}

    kw = dict(cluster_col="scenario_id", weights=weights, n_boot=n_boot, seed=seed)

    # ---- accuracy, citation rate, verdict following ------------------------
    for metric, value_col, frame in (
        ("accuracy", "correct", df),
        ("tool_citation_rate", "cites_tool", df),
        ("verdict_following", "follows_hint", df.filter(pl.col("hint_applicable"))),
    ):
        if frame.height == 0:
            continue
        res = cluster_bootstrap(frame, value_col=value_col, group_cols=["run"], **kw)
        rows += boot_rows(res, metric, ["run"], {"subset": "all"})
        for subset in sorted(frame["subset"].unique().to_list()):
            sub = frame.filter(pl.col("subset") == subset)
            if sub.height == 0:
                continue
            res_s = cluster_bootstrap(sub, value_col=value_col, group_cols=["run"], **kw)
            rows += boot_rows(res_s, metric, ["run"], {"subset": subset})

    # ---- flips against the reference baseline ------------------------------
    baselines = runs.filter(pl.col("is_baseline"))["run"].to_list()
    if not baselines:
        print("WARNING: no baseline run found -- flip metrics skipped")
        return pl.DataFrame(rows), summary

    ref = baselines[0]
    ref_answers = (df.filter(pl.col("run") == ref)
                     .select("key", pl.col("answer").alias("answer_ref")))
    j = df.join(ref_answers, on="key", how="inner")
    j = j.with_columns((pl.col("answer") != pl.col("answer_ref")).alias("flip"))
    summary["reference_baseline"] = ref

    res = cluster_bootstrap(j.filter(pl.col("run") != ref), value_col="flip",
                            group_cols=["run"], **kw)
    rows += boot_rows(res, "flip_rate", ["run"], {"subset": "all"})
    flip_tbl = {r["run"]: r for r in res.table.iter_rows(named=True)}

    self_flip = None
    if len(baselines) > 1:
        alt = baselines[1]
        self_flip = flip_tbl.get(alt)
        summary["self_flip_baseline"] = alt
        # net flip: (flip in the cf run) - (flip of the second baseline), paired
        alt_flip = (j.filter(pl.col("run") == alt)
                      .select("key", pl.col("flip").alias("flip_self")))
        paired = (j.filter(~pl.col("run").is_in([ref, alt]))
                   .join(alt_flip, on="key", how="inner")
                   .with_columns((pl.col("flip").cast(pl.Int8)
                                  - pl.col("flip_self").cast(pl.Int8))
                                 .cast(pl.Float64).alias("net_flip")))
        if paired.height:
            res_net = cluster_bootstrap(paired, value_col="net_flip",
                                        group_cols=["run"], **kw)
            rows += boot_rows(res_net, "net_flip", ["run"], {"subset": "all"})

    # ---- conditional faithfulness on the reversed runs ---------------------
    reversed_runs = runs.filter(
        pl.col("is_baseline").not_()
        & (pl.col("fft_mode").eq("reverse") | pl.col("raft_mode").eq("reverse"))
    )["run"].to_list()
    cond: dict[str, dict] = {}
    for run in reversed_runs:
        sub = j.filter(pl.col("run") == run)
        if sub["cites_tool"].n_unique() < 2:
            continue
        res_c = cluster_bootstrap(sub, value_col="flip", group_cols=["cites_tool"], **kw)
        tbl = res_c.table
        idx = {bool(r["cites_tool"]): i for i, r in enumerate(tbl.iter_rows(named=True))}
        if True not in idx or False not in idx:
            continue
        for cites, metric in ((True, "cond_flip_cites"), (False, "cond_flip_no_cites")):
            r = tbl.row(idx[cites], named=True)
            rows.append({"metric": metric, "run": run, "subset": "all",
                         "n": int(r["n"]), "n_clusters": int(r["n_clusters"]),
                         "estimate": float(r["estimate"]), "lo": float(r["lo"]),
                         "hi": float(r["hi"])})
        est, lo, hi = diff_ci(res_c, idx[True], idx[False])
        rows.append({"metric": "cond_flip_diff", "run": run, "subset": "all",
                     "n": sub.height, "n_clusters": int(tbl["n_clusters"].max()),
                     "estimate": est, "lo": lo, "hi": hi})
        cond[run] = {"cites": float(tbl["estimate"][idx[True]]),
                     "no_cites": float(tbl["estimate"][idx[False]]),
                     "diff": est, "lo": lo, "hi": hi}

    metrics = pl.DataFrame(rows)
    metrics = metrics.join(runs.select("run", "channel", "mode", "is_baseline"),
                           on="run", how="left")
    metrics = metrics.select(["metric", "run", "channel", "mode", "is_baseline",
                              "subset", "n", "n_clusters", "estimate", "lo", "hi"])

    summary["metrics"] = {
        m: metrics.filter(pl.col("metric") == m).to_dicts()
        for m in metrics["metric"].unique().to_list()
    }
    summary["conditional_faithfulness"] = cond
    if self_flip is not None:
        summary["self_flip_rate"] = {"estimate": float(self_flip["estimate"]),
                                     "lo": float(self_flip["lo"]),
                                     "hi": float(self_flip["hi"])}
    return metrics, summary


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


def to_tex(metrics: pl.DataFrame, summary: dict, n_boot: int = 2000) -> str:
    acc = {r["run"]: r for r in metrics.filter((pl.col("metric") == "accuracy")
                                               & (pl.col("subset") == "all")).iter_rows(named=True)}
    flip = {r["run"]: r for r in metrics.filter(pl.col("metric") == "flip_rate").iter_rows(named=True)}
    net = {r["run"]: r for r in metrics.filter(pl.col("metric") == "net_flip").iter_rows(named=True)}
    cite = {r["run"]: r for r in metrics.filter(pl.col("metric") == "tool_citation_rate").iter_rows(named=True)}
    foll = {r["run"]: r for r in metrics.filter(pl.col("metric") == "verdict_following").iter_rows(named=True)}

    def cell(d, run, fmt="{:.3f}"):
        r = d.get(run)
        if r is None:
            return "--"
        return (f"{fmt.format(r['estimate'])} "
                f"[{fmt.format(r['lo'])}, {fmt.format(r['hi'])}]")

    def esc(s: str) -> str:
        return str(s).replace("_", "\\_").replace("&", "\\&")

    lines = [
        "% generated by src/agentic/counterfactual/analyze_cf.py -- do not edit by hand",
        "\\begin{tabular}{llrcccc}",
        "\\toprule",
        "Run & Perturbed & $n$ & Accuracy & Flip vs base & Net flip & Cites tool \\\\",
        "\\midrule",
    ]
    order = sorted(acc, key=lambda r: (not acc[r]["is_baseline"], acc[r]["channel"], r))
    for run in order:
        a = acc[run]
        what = "--" if a["is_baseline"] else f"{a['channel']}: {a['mode']}"
        lines.append(
            f"{esc(run)} & {esc(what)} & {a['n']} & {cell(acc, run)} & "
            f"{cell(flip, run)} & {cell(net, run)} & {cell(cite, run)} \\\\"
        )
    if foll:
        lines += ["\\midrule",
                  "\\multicolumn{7}{l}{\\emph{FFT verdict following} "
                  "-- P(answer $=$ the verdict hint actually shown), "
                  "real-or-generated items} \\\\"]
        for run in order:
            if run in foll:
                a = acc[run]
                what = "--" if a["is_baseline"] else f"{a['channel']}: {a['mode']}"
                lines.append(f"{esc(run)} & {esc(what)} & {foll[run]['n']} & "
                             f"\\multicolumn{{4}}{{l}}{{{cell(foll, run)}}} \\\\")
    cond = summary.get("conditional_faithfulness") or {}
    if cond:
        lines += ["\\midrule",
                  "\\multicolumn{7}{l}{\\emph{Conditional faithfulness} "
                  "-- P(flip $\\mid$ cites tool) $-$ P(flip $\\mid$ does not)} \\\\"]
        for run, c in sorted(cond.items()):
            lines.append(
                f"{esc(run)} & cites / not & -- & {c['cites']:.3f} & "
                f"{c['no_cites']:.3f} & "
                f"{c['diff']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] & \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"% 95\\% percentile cluster bootstrap ({n_boot} replicates) resampling base",
        "% scenarios (the weather re-renderings of one scenario form one cluster).",
        "% \"Flip\" = the answer differs from the baseline run on the same item;",
        "% \"net flip\" subtracts the baseline-vs-baseline self-flip rate, i.e. the",
        "% agent's own sampling noise at temperature 0.3.",
    ]
    return "\n".join(lines) + "\n"


def make_figure(metrics: pl.DataFrame, summary: dict, fig_dir: Path) -> list[Path]:
    setup_style()
    import matplotlib.pyplot as plt

    flip = metrics.filter(pl.col("metric") == "flip_rate").sort("estimate")
    cond_c = {r["run"]: r for r in metrics.filter(pl.col("metric") == "cond_flip_cites").iter_rows(named=True)}
    cond_n = {r["run"]: r for r in metrics.filter(pl.col("metric") == "cond_flip_no_cites").iter_rows(named=True)}

    has_cond = bool(cond_c)
    fig, axes = plt.subplots(1, 2 if has_cond else 1,
                             figsize=(10.4 if has_cond else 6.6, 4.0),
                             squeeze=False)
    ax = axes[0][0]

    labels = [f"{r['channel']}: {r['mode']}" for r in flip.iter_rows(named=True)]
    est = flip["estimate"].to_numpy()
    err = np.vstack([est - flip["lo"].to_numpy(), flip["hi"].to_numpy() - est])
    y = np.arange(len(labels))
    colors = [C_AGENT if r["is_baseline"] else C_BASELINE for r in flip.iter_rows(named=True)]

    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.barh(y, est, height=0.66, color=colors, edgecolor="white", linewidth=0.0)
    ax.errorbar(est, y, xerr=err, fmt="none", ecolor=C_INK, elinewidth=1.1,
                capsize=3, capthick=1.1)
    for i, (e, h) in enumerate(zip(est, flip["hi"].to_numpy())):
        ax.text(h + 0.012, i, f"{e:.3f}", va="center", ha="left", fontsize=8, color=C_INK2)
    sf = summary.get("self_flip_rate")
    if sf:
        ax.axvline(sf["estimate"], color=C_CLOSED, ls="--", lw=1.2, zorder=3,
                   label=f"baseline self-flip ({sf['estimate']:.3f})")
        ax.axvspan(sf["lo"], sf["hi"], color=C_CLOSED, alpha=0.12, zorder=0)
        ax.legend(loc="lower right")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(0.35, float(flip["hi"].max()) + 0.08))
    ax.set_xlabel("P(answer differs from the baseline run)")
    ax.set_title("Counterfactual tool output: answer flip rate", loc="left", pad=10)

    if has_cond:
        ax2 = axes[0][1]
        runs = sorted(cond_c)
        yy = np.arange(len(runs))
        h = 0.34
        for off, d, color, lab in ((+h / 2, cond_c, C_BASELINE, "explanation cites the tool"),
                                   (-h / 2, cond_n, C_CLOSED, "does not cite it")):
            e = np.array([d[r]["estimate"] for r in runs])
            lo = np.array([d[r]["lo"] for r in runs])
            hi = np.array([d[r]["hi"] for r in runs])
            ax2.barh(yy + off, e, height=h, color=color, edgecolor="white",
                     linewidth=0.0, label=lab)
            ax2.errorbar(e, yy + off, xerr=np.vstack([e - lo, hi - e]), fmt="none",
                         ecolor=C_INK, elinewidth=0.9, capsize=2, capthick=0.9)
        ax2.set_axisbelow(True)
        ax2.xaxis.grid(True)
        ax2.set_yticks(yy, runs)
        ax2.set_xlabel("P(flip | tool reversed)")
        ax2.set_title("Conditional faithfulness", loc="left", pad=10)
        ax2.legend(loc="lower right")

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        p = fig_dir / f"fig_counterfactual_flips.{ext}"
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGDIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text())["items"]
    df = load_runs(args.runs_dir, gt)

    runs = run_descriptor(df)
    with pl.Config(tbl_rows=40, fmt_str_lengths=40, tbl_hide_dataframe_shape=True):
        print("=== runs ===")
        print(runs.select("run", "channel", "mode", "is_baseline", "n", "file"))

    metrics, summary = analyse(df, args.n_boot, args.seed)

    with pl.Config(tbl_rows=200, fmt_str_lengths=32, tbl_hide_dataframe_shape=True,
                   float_precision=4):
        print("\n=== metrics ===")
        print(metrics.filter(pl.col("subset") == "all")
                     .sort(["metric", "run"])
                     .select("metric", "run", "mode", "n", "n_clusters",
                             "estimate", "lo", "hi"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.write_parquet(args.out_dir / "cf_metrics.parquet")
    (args.out_dir / "cf_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "cf_table.tex").write_text(to_tex(metrics, summary, args.n_boot))
    print(f"\nwrote {args.out_dir / 'cf_metrics.parquet'}")
    print(f"wrote {args.out_dir / 'cf_summary.json'}")
    print(f"wrote {args.out_dir / 'cf_table.tex'}")

    if not args.no_figure:
        for p in make_figure(metrics, summary, args.figure_dir):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
