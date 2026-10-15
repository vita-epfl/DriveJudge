"""
How much of the FFT anisotropy signal is the video, and how much is the container?

Reads every ``results/fft/features_*.parquet`` that the robustness grid has
produced so far (``src/fft/run_grid.sh``), restricts each to the 600 grid clips
(``src/fft/grid_sample.py``) and scores the *same clips* under every
perturbation: compression, rescaling, frame-rate, codec, clip length, how many
frames the tool averages and which frames it lands on -- plus the two
fingerprint swaps, where a generated clip is re-encoded with the real pool's
mpeg4 profile and a real clip with the generated pool's h264 profile.

Conditions whose parquet does not exist yet are listed as missing and skipped,
so this is safe to run against a half-finished grid.

Usage
-----
    .venv/bin/python src/fft/robustness.py \\
        --grid results/fft/grid_sample.parquet \\
        --features-dir results/fft --out results/fft/robustness/

Outputs
-------
``robustness_summary.parquet`` / ``.json``   one row per condition
``weather.parquet``                          the weather analysis
``tab_fft_robustness.tex``                   the paper table
``report/figures/fft_robustness_{crf,resolution,fps,codec,frames,weather,swap}.pdf``

Thresholds
----------
Everything is scored twice: at the published ``0.048`` and at the threshold
calibrated on the *normalised* clips, read from
``results/fft/calib_n10/calibration_summary.json`` (``t_youden``).  If that file
is absent only ``0.048`` is used, and the summary says so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import cluster_bootstrap, make_cluster_weights  # noqa: E402
from fft import metrics as M  # noqa: E402
from fft.calibrate import COLORS, N_BOOT, SEED, _save  # noqa: E402
from fft.common import DEFAULT_MANIFEST, RESULTS_FFT_DIR, rel_to_repo  # noqa: E402

FEATURE_RE = re.compile(r"^features_(?P<cond>.+)_f(?P<frames>\d+)(?:_o(?P<off>\d+))?\.parquet$")

GRID_FRAMES = 20  # the frame budget run_grid.sh uses for the condition sweep

#: figure -> (x label, [(condition, max_frames, frame_offset, x value, tick label)])
def figure_groups(frames: int = GRID_FRAMES) -> dict:
    return {
        "crf": (
            "x264 CRF (quality knob; higher = more compression)",
            [("crf18", frames, 0, 18, "18"), ("n10", frames, 0, 23, "23\n(baseline)"),
             ("crf28", frames, 0, 28, "28"), ("crf35", frames, 0, 35, "35"),
             ("crf45", frames, 0, 45, "45")],
        ),
        "resolution": (
            "frame width (px)",
            [("res480", frames, 0, 480, "480"), ("res640", frames, 0, 640, "640"),
             ("res960", frames, 0, 960, "960"), ("n10", frames, 0, 1280, "1280\n(baseline)")],
        ),
        "fps": (
            "frame rate (fps)",
            [("fps5", frames, 0, 5, "5"), ("n10", frames, 0, 10, "10\n(baseline)"),
             ("fps24", frames, 0, 24, "24")],
        ),
        "codec": (
            "video codec",
            [("n10", frames, 0, 0, "x264\n(baseline)"), ("codec_mpeg4", frames, 0, 1, "mpeg4"),
             ("codec_hevc", frames, 0, 2, "hevc"), ("codec_vp9", frames, 0, 3, "vp9")],
        ),
        "frames": (
            "frames averaged by the tool",
            [("n10", 5, 0, 5, "5"), ("n10", 10, 0, 10, "10"), ("n10", 20, 0, 20, "20"),
             ("n10", 60, 0, 60, "60\n(agent default)"), ("n10", 20, 1, 21, "20\noffset 1"),
             ("n10", 60, 1, 61, "60\noffset 1")],
        ),
        "swap": (
            "container profile",
            [("raw", frames, 0, 0, "raw\n(as published)"), ("n10", frames, 0, 1, "n10\n(normalised)"),
             ("swap", frames, 0, 2, "swapped\nfingerprints")],
        ),
    }


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover(features_dir: Path) -> dict[tuple[str, int, int], Path]:
    """``(condition, max_frames, frame_offset) -> parquet`` for what exists."""
    found: dict[tuple[str, int, int], Path] = {}
    for p in sorted(features_dir.glob("features_*.parquet")):
        m = FEATURE_RE.match(p.name)
        if not m:
            print(f"  ignoring unparsable file name: {p.name}")
            continue
        key = (m["cond"], int(m["frames"]), int(m["off"] or 0))
        found[key] = p
    return found


def load_condition(path: Path, cond: str, frames: int, off: int, grid: pl.DataFrame) -> pl.DataFrame:
    f = pl.read_parquet(path).filter(
        (pl.col("condition") == cond)
        & (pl.col("max_frames") == frames)
        & (pl.col("frame_offset") == off)
        & pl.col("error").is_null()
        & pl.col("aniso_mean").is_not_null()
    )
    f = f.sort(["skip_periodic", "impl"]).unique(subset=["clip_id"], keep="first")
    return f.join(grid, on="clip_id", how="inner").with_columns(
        (pl.col("label") == "generated").cast(pl.Int8).alias("y")
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(sub: pl.DataFrame, thresholds: dict[str, float], label: str) -> dict:
    """AUC + per-threshold metrics + class medians for one condition."""
    y = sub["y"].to_numpy().astype(float)
    s = sub["aniso_mean"].to_numpy().astype(float)
    clusters = sub["scenario_id"].to_numpy()
    both = bool(y.sum() > 0 and (1 - y).sum() > 0)
    cw = M.weights_for(clusters, N_BOOT, SEED)
    w = M.row_weights(cw, clusters)

    row: dict = {
        "condition": label,
        "n": int(len(y)),
        "n_gen": int(y.sum()),
        "n_real": int((1 - y).sum()),
        "n_clusters": int(len(np.unique(clusters))),
        "two_class": both,
    }
    tnames = list(thresholds)
    tvals = np.array([thresholds[k] for k in tnames])
    pt = M.metrics_at(y, s, tvals, M.unit_weights(len(y)))
    bt = M.metrics_at(y, s, tvals, w)
    for ti, tname in enumerate(tnames):
        for met in ("bal_acc", "acc", "tpr", "fpr", "mcc"):
            if not both and met not in ("tpr", "fpr"):
                continue
            r = M.est_ci(pt[met][0, ti], bt[met][:, ti])
            row[f"{met}_{tname}"] = r["estimate"]
            row[f"{met}_{tname}_lo"] = r["lo"]
            row[f"{met}_{tname}_hi"] = r["hi"]
    if both:
        ra_p = M.roc_analysis(y, s, M.unit_weights(len(y)))
        ra_b = M.roc_analysis(y, s, w)
        for key in ("auc", "ap"):
            r = M.est_ci(ra_p[key][0], ra_b[key])
            row[key] = r["estimate"]
            row[f"{key}_lo"] = r["lo"]
            row[f"{key}_hi"] = r["hi"]
        yj = M.est_ci(ra_p["thr_youden"][0], ra_b["thr_youden"])
        row["t_youden_here"] = yj["estimate"]
        row["t_youden_here_lo"] = yj["lo"]
        row["t_youden_here_hi"] = yj["hi"]

    for g in ("real", "dd", "p1", "dd_unlabelled", "p1_unlabelled"):
        part = sub.filter(pl.col("pool_generator") == g)
        row[f"median_{g}"] = float(part["aniso_mean"].median()) if part.height else None
        row[f"n_{g}"] = part.height
    return row


def swap_frame(loaded: dict, frames: int) -> pl.DataFrame | None:
    """Generated clips wearing the real profile + real clips wearing the generated one."""
    gen = loaded.get(("swap_gen2real", frames, 0))
    real = loaded.get(("swap_real2gen", frames, 0))
    if gen is None or real is None:
        return None
    gen = gen.filter(pl.col("y") == 1)
    real = real.filter(pl.col("y") == 0)
    if gen.is_empty() or real.is_empty():
        return None
    return pl.concat([gen, real.select(gen.columns)])


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------


def weather_analysis(features_dir: Path, manifest: Path, splits_hint: str = "features_raw_f60.parquet") -> dict:
    """Within-scenario anisotropy by weather, and per-weather AUC against real.

    Deliberately run on *all* clips of ``features_raw_f60.parquet``, not on the
    grid: the grid keeps at most one clip per scenario, so a within-scenario
    contrast is impossible there.  The within-scenario value is
    ``aniso_mean - mean(aniso_mean over that scenario)``, which removes the
    scene and leaves the weather.
    """
    path = features_dir / splits_hint
    if not path.exists():
        return {"skipped": f"missing {rel_to_repo(path)}"}
    f = pl.read_parquet(path).filter(
        (pl.col("condition") == "raw") & (pl.col("max_frames") == 60) & (pl.col("frame_offset") == 0)
        & pl.col("error").is_null() & pl.col("aniso_mean").is_not_null()
    )
    man = pl.read_parquet(manifest).select("clip_id", "label", "generator", "weather", "scenario_id", "folder")
    d = f.join(man, on="clip_id", how="inner")
    gen = d.filter(pl.col("label") == "generated")
    real = d.filter(pl.col("label") == "real")
    out: dict = {"source": rel_to_repo(path), "n_generated": gen.height, "n_real": real.height}
    if gen.height < 20:
        out["skipped"] = f"only {gen.height} generated clips with features"
        return out

    # multi-clip scenarios only: a singleton scenario carries no within contrast
    counts = gen.group_by("scenario_id").agg(pl.len().alias("k"))
    multi = gen.join(counts.filter(pl.col("k") >= 2), on="scenario_id", how="inner")
    within = (
        multi.with_columns(
            (pl.col("aniso_mean") - pl.col("aniso_mean").mean().over("scenario_id")).alias("within")
        )
        .filter(pl.col("weather").is_not_null())
        .group_by("weather")
        .agg(pl.len().alias("n"), pl.col("within").mean().alias("mean_within"),
             pl.col("within").std().alias("sd_within"),
             pl.col("aniso_mean").mean().alias("mean_aniso"))
        .sort("mean_within")
    )
    out["n_scenarios_multi"] = int(multi["scenario_id"].n_unique())
    out["within_scenario"] = within.to_dicts()

    per_weather = []
    if real.height >= 20:
        for w in sorted([x for x in gen["weather"].unique().to_list() if x is not None]):
            part = gen.filter(pl.col("weather") == w)
            if part.height < 5:
                continue
            sub = pl.concat([part.select("aniso_mean", "scenario_id", "label"),
                             real.select("aniso_mean", "scenario_id", "label")])
            y = (sub["label"] == "generated").cast(pl.Int8).to_numpy().astype(float)
            s = sub["aniso_mean"].to_numpy().astype(float)
            cl = sub["scenario_id"].to_numpy()
            cw = M.weights_for(cl, N_BOOT, SEED)
            a_p = M.roc_analysis(y, s, M.unit_weights(len(y)))["auc"][0]
            a_b = M.roc_analysis(y, s, M.row_weights(cw, cl))["auc"]
            tpr = float((part["aniso_mean"] >= M.PUBLISHED_THRESHOLD).mean())
            per_weather.append({"weather": w, "n_gen": part.height, "n_real": real.height,
                                "tpr_at_0.048": tpr, **M.est_ci(a_p, a_b)})
    out["per_weather_auc"] = per_weather

    if per_weather:
        # TPR per weather with cluster CIs, via the project's own bootstrap
        part = gen.filter(pl.col("weather").is_not_null()).with_columns(
            (pl.col("aniso_mean") >= M.PUBLISHED_THRESHOLD).cast(pl.Int8).alias("said_generated")
        )
        w = make_cluster_weights(part["scenario_id"].unique().to_numpy(), N_BOOT, SEED)
        br = cluster_bootstrap(part, value_col="said_generated", cluster_col="scenario_id",
                               group_cols=["weather"], weights=w)
        out["per_weather_tpr"] = br.table.to_dicts()
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_group(name: str, xlabel: str, points: list, table: dict, missing: list) -> list[str]:
    import matplotlib.pyplot as plt

    have = [(x, lab, table[k]) for k, x, lab in points if k in table]
    if len(have) < 2:
        return []
    have.sort(key=lambda r: r[0])
    xs = [h[0] for h in have]
    ticks = [h[1] for h in have]
    fig, ax = plt.subplots(figsize=(1.15 * len(have) + 2.4, 3.4))
    for metric, colour, lab in (("auc", COLORS["real"], "ROC-AUC"),
                                ("bal_acc_t0.048", COLORS["fpr"], "balanced acc @ 0.048"),
                                ("bal_acc_t_n10", COLORS["tpr"], "balanced acc @ calibrated t")):
        if not any(metric in h[2] and h[2].get(metric) is not None for h in have):
            continue
        v = [h[2].get(metric) for h in have]
        lo = [h[2].get(f"{metric}_lo") for h in have]
        hi = [h[2].get(f"{metric}_hi") for h in have]
        ax.plot(xs, v, "o-", color=colour, lw=1.6, ms=4, label=lab)
        if all(x is not None for x in lo):
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.16, lw=0)
    ax.axhline(0.5, color="#999999", ls=":", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    title = f"FFT detector robustness: {name}"
    if missing:
        title += f"  (missing: {', '.join(missing)})"
    ax.set_title(title, fontsize=9)
    return _save(fig, f"fft_robustness_{name}")


def fig_weather(w: dict) -> list[str]:
    import matplotlib.pyplot as plt

    within = w.get("within_scenario") or []
    per = w.get("per_weather_auc") or []
    if not within and not per:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    if within:
        labs = [r["weather"] for r in within]
        vals = [r["mean_within"] for r in within]
        err = [(r["sd_within"] or 0) / max(1, np.sqrt(r["n"])) for r in within]
        axes[0].bar(range(len(labs)), vals, yerr=err, color=COLORS["gen"], capsize=2,
                    error_kw={"lw": 0.8, "ecolor": "#333333"})
        axes[0].axhline(0, color="#333333", lw=0.8)
        axes[0].set_xticks(range(len(labs)))
        axes[0].set_xticklabels(labs, rotation=30, ha="right", fontsize=7.5)
        axes[0].set_ylabel("anisotropy - scenario mean")
        axes[0].set_title(f"within-scenario weather effect (n={sum(r['n'] for r in within)})", fontsize=9)
        axes[0].grid(alpha=0.25, lw=0.5, axis="y")
    if per:
        labs = [r["weather"] for r in per]
        vals = [r["estimate"] for r in per]
        lo = [v - (r["lo"] or v) for v, r in zip(vals, per)]
        hi = [(r["hi"] or v) - v for v, r in zip(vals, per)]
        axes[1].bar(range(len(labs)), vals, yerr=[lo, hi], color=COLORS["real"], capsize=2,
                    error_kw={"lw": 0.8, "ecolor": "#333333"})
        axes[1].axhline(0.5, color="#999999", ls=":", lw=0.9)
        axes[1].set_xticks(range(len(labs)))
        axes[1].set_xticklabels(labs, rotation=30, ha="right", fontsize=7.5)
        axes[1].set_ylabel("AUC vs all real clips")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_title("separability per weather (raw clips, 60 frames)", fontsize=9)
        axes[1].grid(alpha=0.25, lw=0.5, axis="y")
    fig.tight_layout()
    return _save(fig, "fft_robustness_weather")


def fig_swap(table: dict, frames: int) -> list[str]:
    import matplotlib.pyplot as plt

    keys = [(("raw", frames, 0), "raw\n(as published)"), (("n10", frames, 0), "n10\n(normalised)"),
            (("swap", frames, 0), "swapped\nfingerprints")]
    have = [(lab, table[k]) for k, lab in keys if k in table]
    if len(have) < 2:
        return []
    fig, ax = plt.subplots(figsize=(1.7 * len(have) + 2.2, 3.4))
    x = np.arange(len(have))
    width = 0.26
    for i, (metric, colour, lab) in enumerate((("tpr_t0.048", COLORS["tpr"], "TPR (generated found)"),
                                               ("fpr_t0.048", COLORS["fpr"], "FPR (real mislabelled)"),
                                               ("bal_acc_t0.048", COLORS["real"], "balanced accuracy"))):
        v = [h[1].get(metric) for h in have]
        lo = [(h[1].get(metric) or 0) - (h[1].get(f"{metric}_lo") or h[1].get(metric) or 0) for h in have]
        hi = [(h[1].get(f"{metric}_hi") or h[1].get(metric) or 0) - (h[1].get(metric) or 0) for h in have]
        ax.bar(x + (i - 1) * width, v, width * 0.92, color=colour, label=lab, yerr=[lo, hi],
               capsize=2, error_kw={"lw": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels([h[0] for h in have], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate @ t = 0.048")
    ax.set_title("Fingerprint swap control (same clips, same threshold)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    return _save(fig, "fft_robustness_swap")


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------


def latex(rows: list[dict], thresholds: dict) -> str:
    def f(row, key, digits=3):
        v = row.get(key)
        if v is None:
            return "--"
        lo, hi = row.get(f"{key}_lo"), row.get(f"{key}_hi")
        if lo is None:
            return f"{v:.{digits}f}"
        return f"{v:.{digits}f} \\tiny[{lo:.{digits}f},{hi:.{digits}f}]"

    tcal = [k for k in thresholds if k != "t0.048"]
    head = "condition & $n$ & AUC & bal.\\,acc @0.048 & TPR & FPR"
    if tcal:
        head += " & bal.\\,acc @$t^{*}$"
    lines = [
        "% generated by src/fft/robustness.py -- do not edit by hand",
        "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lr" + "r" * (4 + len(tcal)) + "}", "\\toprule", head + " \\\\", "\\midrule",
    ]
    for r in rows:
        cells = [r["condition"].replace("_", "\\_"), str(r["n"]), f(r, "auc"),
                 f(r, "bal_acc_t0.048"), f(r, "tpr_t0.048"), f(r, "fpr_t0.048")]
        if tcal:
            cells.append(f(r, f"bal_acc_{tcal[0]}"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{FFT anisotropy detector under container perturbations, on the "
        "600-clip robustness grid (300 real / 150 dd / 150 p1). 95\\% cluster-bootstrap "
        "intervals over \\texttt{scenario\\_id}"
        + (f", $t^{{*}}={thresholds[tcal[0]]:.4f}$ calibrated on normalised clips" if tcal else "")
        + ".}",
        "\\label{tab:fft-robustness}", "\\end{table}", "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=Path, default=RESULTS_FFT_DIR / "grid_sample.parquet")
    ap.add_argument("--features-dir", type=Path, default=RESULTS_FFT_DIR)
    ap.add_argument("--out", type=Path, default=RESULTS_FFT_DIR / "robustness")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--calib-n10", type=Path, default=RESULTS_FFT_DIR / "calib_n10" / "calibration_summary.json")
    ap.add_argument("--grid-frames", type=int, default=GRID_FRAMES)
    args = ap.parse_args()

    if not args.grid.exists():
        raise SystemExit(f"grid sample not found: {args.grid}  (run src/fft/grid_sample.py)")
    grid = pl.read_parquet(args.grid).select(
        "clip_id", "label", "generator", "pool_generator", "folder", "weather", "scenario_id",
        "split", "codec", "width", "height", "fps"
    )
    print(f"grid: {rel_to_repo(args.grid)}  ({grid.height} clips, "
          f"{int((grid['label'] == 'real').sum())} real / {int((grid['label'] == 'generated').sum())} generated)")
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- thresholds ------------------------------------------------------
    thresholds = {"t0.048": M.PUBLISHED_THRESHOLD}
    if args.calib_n10.exists():
        cal = json.loads(args.calib_n10.read_text())
        t = (cal.get("threshold_values") or {}).get("t_youden")
        if t is not None:
            thresholds["t_n10"] = float(t)
            print(f"calibrated threshold from {rel_to_repo(args.calib_n10)}: t_n10 = {t:.5f}"
                  f"  (design={cal.get('design', 'splits')}, "
                  f"max_frames={cal.get('max_frames', '?')}"
                  + ("; the grid is scored at "
                     f"{args.grid_frames} frames, so this is a cross-budget threshold"
                     if cal.get("max_frames") not in (None, args.grid_frames) else "")
                  + ")")
        else:
            print(f"NOTE: {rel_to_repo(args.calib_n10)} has no calibrated threshold "
                  f"(the n10 calibration fold was probably empty); using 0.048 only")
    else:
        print(f"NOTE: {rel_to_repo(args.calib_n10)} does not exist yet -- run "
              f"calibrate.py --condition n10 first; using 0.048 only")

    # ---- discovery -------------------------------------------------------
    print("\n=== available feature files ===")
    found = discover(args.features_dir)
    loaded: dict[tuple[str, int, int], pl.DataFrame] = {}
    for key, path in found.items():
        sub = load_condition(path, *key, grid)
        if sub.is_empty():
            print(f"  {str(key):<34} {path.name}: no grid clips yet -> skipped")
            continue
        loaded[key] = sub
        print(f"  {str(key):<34} {path.name}: {sub.height}/{grid.height} grid clips "
              f"({int(sub['y'].sum())} gen / {int((1 - sub['y']).sum())} real)")
    if not loaded:
        print("\nnothing to analyse yet -- run src/fft/run_grid.sh")
        return 0

    sw = swap_frame(loaded, args.grid_frames)
    if sw is not None:
        loaded[("swap", args.grid_frames, 0)] = sw
        print(f"  {'swap (gen2real + real2gen)':<34} synthesised: {sw.height} clips")

    # ---- score -----------------------------------------------------------
    print("\n=== per-condition scores ===")
    table: dict[tuple[str, int, int], dict] = {}
    rows = []
    for key in sorted(loaded, key=lambda k: (k[0], k[1], k[2])):
        sub = loaded[key]
        cond, frames, off = key
        label = f"{cond}_f{frames}" + (f"_o{off}" if off else "")
        r = score(sub, thresholds, label)
        r.update({"cond": cond, "max_frames": frames, "frame_offset": off})
        table[key] = r
        rows.append(r)
        auc = f"{r['auc']:.4f}[{r['auc_lo']:.4f},{r['auc_hi']:.4f}]" if r.get("auc") is not None else "--"
        bal = r.get("bal_acc_t0.048")
        print(f"  {label:<26} n={r['n']:<4} AUC {auc:<26} "
              f"bal@0.048 {'--' if bal is None else f'{bal:.3f}'}  "
              f"TPR {_o(r.get('tpr_t0.048'))}  FPR {_o(r.get('fpr_t0.048'))}  "
              f"median real {_o(r.get('median_real'), 4)} / dd {_o(r.get('median_dd'), 4)} / p1 {_o(r.get('median_p1'), 4)}")

    summary = pl.DataFrame(rows, infer_schema_length=None)
    summary.write_parquet(args.out / "robustness_summary.parquet")

    # ---- weather ---------------------------------------------------------
    print("\n=== weather analysis (all clips of features_raw_f60.parquet) ===")
    w = weather_analysis(args.features_dir, args.manifest)
    if "skipped" in w:
        print(f"  SKIPPED: {w['skipped']}")
    else:
        print(f"  {w['n_generated']} generated clips, {w.get('n_scenarios_multi', 0)} multi-clip scenarios")
        for r in w.get("within_scenario", []):
            print(f"    {r['weather']:<12} n={r['n']:<5} within-scenario mean {r['mean_within']:+.5f}"
                  f"  absolute {r['mean_aniso']:.5f}")
        for r in w.get("per_weather_auc", []):
            print(f"    {r['weather']:<12} AUC vs real {r['estimate']:.4f} "
                  f"[{r['lo']:.4f}, {r['hi']:.4f}]  TPR@0.048 {r['tpr_at_0.048']:.3f}")
        pl.DataFrame(w.get("within_scenario") or [], infer_schema_length=None).write_parquet(
            args.out / "weather.parquet"
        )

    # ---- figures ---------------------------------------------------------
    print("\n=== figures ===")
    figs = []
    for name, (xlabel, points) in figure_groups(args.grid_frames).items():
        keys = [((c, fr, of), x, lab) for c, fr, of, x, lab in points]
        missing = [_key_label(k) for k, _, _ in keys if k not in table]
        if name == "swap":
            got = fig_swap(table, args.grid_frames)
        else:
            got = fig_group(name, xlabel, keys, table, missing)
        if got:
            figs += got
            print(f"  {name}: {got[0]}" + (f"   (missing conditions: {', '.join(missing)})" if missing else ""))
        else:
            print(f"  {name}: SKIPPED -- need at least two of "
                  f"{[_key_label(k) for k, _, _ in keys]}, missing {missing}")
    got = fig_weather(w)
    figs += got
    print(f"  weather: {got[0] if got else 'SKIPPED (no weather data yet)'}")

    # ---- outputs ---------------------------------------------------------
    (args.out / "tab_fft_robustness.tex").write_text(latex(rows, thresholds))
    payload = {
        "grid": rel_to_repo(args.grid),
        "features_dir": rel_to_repo(args.features_dir),
        "thresholds": thresholds,
        "n_grid_clips": grid.height,
        "conditions_found": [f"{c}_f{f}" + (f"_o{o}" if o else "") for c, f, o in sorted(found)],
        "conditions_scored": [r["condition"] for r in rows],
        "conditions_missing": sorted(
            {_key_label((c, fr, of)) for _, (_, pts) in figure_groups(args.grid_frames).items()
             for c, fr, of, _, _ in pts if (c, fr, of) not in table}
        ),
        "per_condition": rows,
        "weather": w,
        "figures": figs,
    }
    (args.out / "robustness_summary.json").write_text(json.dumps(_j(payload), indent=2))
    print(f"\nwrote {rel_to_repo(args.out)}/: robustness_summary.parquet, robustness_summary.json, "
          f"weather.parquet, tab_fft_robustness.tex")
    if payload["conditions_missing"]:
        print(f"\nstill missing (run src/fft/run_grid.sh): {', '.join(payload['conditions_missing'])}")
    return 0


def _key_label(key: tuple[str, int, int]) -> str:
    cond, frames, off = key
    return f"{cond}_f{frames}" + (f"_o{off}" if off else "")


def _o(v, digits: int = 3) -> str:
    return "--" if v is None else f"{v:.{digits}f}"


def _j(obj):
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
