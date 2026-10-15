"""
Calibrate and stress-test the agent's FFT anisotropy detector.

The agent's ``get_frequency_analysis`` tool declares a clip *Generated* when its
mean anisotropy is >= 0.048 -- a threshold eyeballed on 18 real + 18 generated
clips whose real half is mpeg4 1280x720@10fps and whose generated half is h264
1280x704@24fps.  This script re-derives the threshold on a scenario-disjoint
calibration fold (``src/fft/splits.py``), reports what it costs to keep 0.048,
and measures how much of the separation survives when the container fingerprint
is normalised away (``--condition n10``).

Usage
-----
    .venv/bin/python src/fft/calibrate.py \\
        --features results/fft/features_raw_f60.parquet --condition raw \\
        --out results/fft/calib_raw/

    .venv/bin/python src/fft/calibrate.py \\
        --features results/fft/features_n10_f60.parquet --condition n10 \\
        --out results/fft/calib_n10/

    # side-by-side, on the E6 clips both conditions have
    .venv/bin/python src/fft/calibrate.py --compare raw n10

Two designs
-----------
``--mode splits`` (used for ``raw``, which covers the whole 5510-clip manifest)
fits the threshold on the external, scenario-disjoint ``calib`` fold and tests
on ``holdout`` / ``e6_test`` / ``extra_gen``.

``--mode e6cv`` (used for ``n10``, whose re-encode only exists for the 2000 E6
clips listed in ``results/fft/e6_clip_ids.txt``) has no external calibration
fold, so it cross-validates the threshold *inside* the E6 set: 5-fold, grouped
by ``scenario_id`` and stratified on the label, giving an out-of-fold operating
point plus the spread of the five per-fold thresholds.  The AUC is
threshold-free and therefore not fold-dependent -- it is reported over the whole
set, per fold, and per generator.

``auto`` (the default) picks ``e6cv`` whenever the calibration fold is unusable
and at least 100 E6 clips have features.

Every number is a cluster bootstrap over ``scenario_id`` (B=2000, seed 0) -- the
eight weather re-renderings of a scene are near duplicates, so the clip is not
the unit of independence.  Partial feature files are fine: the script reports
the row count of every fold and simply skips what it cannot compute.

Outputs (under ``--out``)
-------------------------
``calibration_summary.json``  everything, machine-readable
``fold_metrics.parquet``      one row per (fold, subset, threshold, metric)
``threshold_sweep.parquet``   acc / TPR / FPR vs t on holdout and e6_test
``roc_points.parquet``        ROC bands (TPR at a fixed FPR grid) per fold
``lr_coefficients.parquet``   standardised logistic-regression coefficients
``tab_fft_calibration.tex``   the paper table
and figures ``report/figures/fft_{roc,threshold_sweep,aniso_hist,heldout_generators}_<cond>.pdf``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import cluster_bootstrap, make_cluster_weights, wilson_ci  # noqa: E402
from fft import metrics as M  # noqa: E402
from fft.common import DEFAULT_MANIFEST, REPO, RESULTS_FFT_DIR, rel_to_repo  # noqa: E402
from fft.splits import DEFAULT_SPLITS  # noqa: E402

FIG_DIR = REPO / "report" / "figures"

#: the seven features the multi-feature comparison uses.
LR_FEATURES = [
    "aniso_mean",
    "aniso_std",
    "peak_ratio_mean",
    "axis_ratio_mean",
    "hf_ratio_mean",
    "spectral_slope_mean",
    "anomaly_ratio_mean",
]

#: a feature is dropped from the multi-feature model when more than this
#: fraction of clips lack it.  ``extract_features.py --fast`` skips
#: ``detect_periodic_artifacts``, so a feature file assembled from a mix of
#: ``--fast`` and non-``--fast`` runs has ``anomaly_ratio_*`` on only some rows;
#: dropping those *rows* instead of the *column* would silently throw away most
#: of the dataset.
MAX_FEATURE_NULL_FRACTION = 0.20

#: folds with both classes -> full metric set; generated-only folds -> TPR only.
TWO_CLASS_FOLDS = ["calib", "holdout", "e6_test", "calib18"]
GEN_ONLY_FOLDS = ["extra_gen", "e6_sibling"]

SWEEP_LO, SWEEP_HI, SWEEP_STEP = 0.030, 0.082, 0.001
FPR_GRID = np.linspace(0.0, 1.0, 201)
N_BOOT = 2000
SEED = 0

# Okabe-Ito, colourblind safe.
COLORS = {
    "real": "#0072B2",
    "gen": "#D55E00",
    "dd": "#D55E00",
    "p1": "#E69F00",
    "dd_unlabelled": "#B25000",
    "p1_unlabelled": "#8C6D00",
    "acc": "#000000",
    "tpr": "#009E73",
    "fpr": "#D55E00",
    "raw": "#0072B2",
    "n10": "#009E73",
    "published": "#666666",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_joined(
    features: Path,
    condition: str,
    manifest: Path = DEFAULT_MANIFEST,
    splits: Path = DEFAULT_SPLITS,
    max_frames: int = 60,
    frame_offset: int = 0,
    quiet: bool = False,
) -> pl.DataFrame:
    """features x manifest x splits, restricted to one extraction configuration."""
    if not features.exists():
        hint = (f"\n  .venv/bin/python src/fft/extract_features.py --condition {condition} "
                f"--max-frames {max_frames} --impl fast --workers 4"
                + ("  --filter \"$(cat results/fft/e6_filter.txt)\"" if condition != "raw" else ""))
        raise SystemExit(f"features not found: {rel_to_repo(features)}{hint}")
    for p in (manifest, splits):
        if not p.exists():
            raise SystemExit(f"missing {p}  (run src/fft/manifest.py / src/fft/splits.py)")

    f = pl.read_parquet(features)
    n_all = f.height
    f = f.filter(
        (pl.col("condition") == condition)
        & (pl.col("max_frames") == max_frames)
        & (pl.col("frame_offset") == frame_offset)
    )
    n_cfg = f.height
    n_err = int(f.filter(pl.col("error").is_not_null()).height)
    f = f.filter(pl.col("error").is_null() & pl.col("aniso_mean").is_not_null())
    # A clip can appear twice when part of the file was produced with
    # ``--fast`` (which skips detect_periodic_artifacts) and part without.  The
    # anisotropy is identical either way; keep the richer row.
    mix = f.group_by("skip_periodic").agg(pl.len().alias("n")).sort("skip_periodic").to_dicts()
    f = f.sort(["skip_periodic", "impl"]).unique(subset=["clip_id"], keep="first")

    man = pl.read_parquet(manifest).select(
        "clip_id", "label", "generator", "folder", "weather", "gen_size", "scenario_id",
        "in_benchmark", "category", "codec", "width", "height", "fps",
    ).rename({"width": "src_width", "height": "src_height"})
    spl = pl.read_parquet(splits).select("clip_id", "split", "pool_generator", "in_e6", "in_calib18")

    df = f.join(man, on="clip_id", how="inner").join(spl, on="clip_id", how="inner")
    df = df.with_columns(
        (pl.col("label") == "generated").cast(pl.Int8).alias("y"),
        (pl.col("src_width").cast(pl.Utf8) + "x" + pl.col("src_height").cast(pl.Utf8)).alias("resolution"),
    )
    if not quiet:
        print(f"features : {rel_to_repo(features)}  ({n_all} rows total)")
        print(
            f"  condition={condition} max_frames={max_frames} frame_offset={frame_offset}: "
            f"{n_cfg} rows, {n_err} errors, {df.height} usable after the manifest/splits join"
        )
        if len(mix) > 1:
            counts = ", ".join(f"skip_periodic={m['skip_periodic']}: {m['n']}" for m in mix)
            print(f"  mixed extraction settings ({counts}) -- keeping the row that carries "
                  f"anomaly_ratio where both exist; the anisotropy is identical either way")
        with pl.Config(tbl_rows=20, tbl_width_chars=160):
            print(
                df.group_by("split")
                .agg(pl.len().alias("n"), (pl.col("y") == 1).sum().alias("gen"), (pl.col("y") == 0).sum().alias("real"),
                     pl.col("scenario_id").n_unique().alias("scen"))
                .sort("split")
            )
    return df


def fold(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Named subsets.  ``e6_*`` names live inside the 2000-clip E6 set."""
    if name.startswith("gen:"):
        return df.filter(pl.col("generator") == name[len("gen:"):])
    if name == "dd_unlabelled":
        return df.filter(pl.col("pool_generator") == "dd_unlabelled")
    if name == "e6_all":
        return df.filter(pl.col("in_e6"))
    if name.startswith("e6_") and name.endswith("_vs_real"):
        gen = name[len("e6_"):-len("_vs_real")]
        return df.filter(pl.col("in_e6") & ((pl.col("generator") == gen) | (pl.col("label") == "real")))
    return df.filter(pl.col("split") == name)


def usable(sub: pl.DataFrame, need_both: bool = True, min_n: int = 4) -> bool:
    if sub.height < min_n:
        return False
    if need_both:
        return bool(sub["y"].sum() > 0 and (1 - sub["y"]).sum() > 0)
    return bool(sub["y"].sum() > 0)


# --------------------------------------------------------------------------
# Metric emission
# --------------------------------------------------------------------------


class Rows:
    """Accumulator for ``fold_metrics.parquet``."""

    def __init__(self, condition: str):
        self.condition = condition
        self.rows: list[dict] = []

    def add(self, fold: str, subset: str, subset_value: str, threshold_name: str,
            threshold: float | None, metric: str, res: dict, n: int, n_pos: int,
            n_neg: int, n_clusters: int, extra: dict | None = None) -> None:
        self.rows.append(
            {
                "condition": self.condition,
                "fold": fold,
                "subset": subset,
                "subset_value": subset_value,
                "threshold_name": threshold_name,
                "threshold": threshold,
                "metric": metric,
                "n": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_clusters": n_clusters,
                "estimate": res.get("estimate"),
                "lo": res.get("lo"),
                "hi": res.get("hi"),
                "se": res.get("se"),
                **(extra or {}),
            }
        )

    def frame(self) -> pl.DataFrame:
        if not self.rows:
            return pl.DataFrame(
                schema={
                    "condition": pl.Utf8, "fold": pl.Utf8, "subset": pl.Utf8, "subset_value": pl.Utf8,
                    "threshold_name": pl.Utf8, "threshold": pl.Float64, "metric": pl.Utf8,
                    "n": pl.Int64, "n_pos": pl.Int64, "n_neg": pl.Int64, "n_clusters": pl.Int64,
                    "estimate": pl.Float64, "lo": pl.Float64, "hi": pl.Float64, "se": pl.Float64,
                }
            )
        return pl.DataFrame(self.rows)


def eval_fold(rows: Rows, name: str, sub: pl.DataFrame, thresholds: dict[str, float],
              score_col: str = "aniso_mean", tag: str = "aniso") -> dict:
    """Full metric set at every named threshold + AUC/PR-AUC, with cluster CIs."""
    y = sub["y"].to_numpy().astype(float)
    score = sub[score_col].to_numpy().astype(float)
    clusters = sub["scenario_id"].to_numpy()
    both = bool(y.sum() > 0 and (1 - y).sum() > 0)
    cw = M.weights_for(clusters, N_BOOT, SEED)
    w_boot = M.row_weights(cw, clusters)
    n, npos, nneg, ncl = int(len(y)), int(y.sum()), int((1 - y).sum()), int(len(np.unique(clusters)))

    tnames = list(thresholds)
    tvals = np.array([thresholds[k] for k in tnames], dtype=float)
    point = M.metrics_at(y, score, tvals, M.unit_weights(len(y)))
    reps = M.metrics_at(y, score, tvals, w_boot)

    wanted = M.METRIC_NAMES if both else ("tpr",)
    out: dict = {"n": n, "n_pos": npos, "n_neg": nneg, "n_clusters": ncl, "two_class": both,
                 "thresholds": {}}
    for ti, tname in enumerate(tnames):
        entry = {}
        for m in wanted:
            res = M.est_ci(point[m][0, ti], reps[m][:, ti])
            entry[m] = res
            rows.add(name, "all", "all", tname, float(tvals[ti]), f"{tag}_{m}", res, n, npos, nneg, ncl)
        # i.i.d. reference for accuracy (Wilson), to show how much the clustering costs
        if both:
            k = int(round(point["acc"][0, ti] * n))
            lo, hi = wilson_ci(k, n)
            entry["acc_wilson"] = {"estimate": point["acc"][0, ti], "lo": float(lo), "hi": float(hi)}
        out["thresholds"][tname] = entry

    if both:
        ra_pt = M.roc_analysis(y, score, M.unit_weights(len(y)), fpr_grid=FPR_GRID)
        ra_bt = M.roc_analysis(y, score, w_boot, fpr_grid=FPR_GRID)
        for key in ("auc", "ap"):
            res = M.est_ci(ra_pt[key][0], ra_bt[key])
            out[key] = res
            rows.add(name, "all", "all", "-", None, f"{tag}_{key}", res, n, npos, nneg, ncl)
        lo, hi = M.ci(ra_bt["tpr_at_fpr"])
        out["_roc"] = {"fpr": FPR_GRID, "tpr": ra_pt["tpr_at_fpr"][0], "lo": lo, "hi": hi,
                       "curve_fpr": ra_pt["curve_fpr"], "curve_tpr": ra_pt["curve_tpr"]}
    return out


def breakdown(rows: Rows, name: str, sub: pl.DataFrame, t: float, tname: str,
              by: str, side: str = "gen") -> list[dict]:
    """Per-``by`` rate of "called Generated" at a fixed threshold.

    ``side='gen'`` restricts to generated clips, so the rate *is* the TPR;
    ``side='real'`` restricts to real clips, so it is the FPR.  Uses
    :func:`analysis.stats.cluster_bootstrap` -- these are plain means, so the
    project's own bootstrap applies directly, with one shared weight matrix per
    fold so the groups are coherent.
    """
    part = sub.filter(pl.col("y") == (1 if side == "gen" else 0))
    part = part.filter(pl.col(by).is_not_null())
    if part.is_empty():
        return []
    part = part.with_columns((pl.col("aniso_mean") >= t).cast(pl.Int8).alias("said_generated"))
    w = make_cluster_weights(part["scenario_id"].unique().to_numpy(), N_BOOT, SEED)
    br = cluster_bootstrap(part, value_col="said_generated", cluster_col="scenario_id",
                           group_cols=[by], weights=w)
    metric = "tpr" if side == "gen" else "fpr"
    out = []
    for r in br.table.iter_rows(named=True):
        res = {"estimate": r["estimate"], "lo": r["lo"], "hi": r["hi"], "se": None}
        n = int(r["n"])
        rows.add(name, by, str(r[by]), tname, t, f"aniso_{metric}", res,
                 n, n if side == "gen" else 0, 0 if side == "gen" else n, int(r["n_clusters"]))
        out.append({by: r[by], "n": n, "n_clusters": int(r["n_clusters"]), "side": side, **res})
    return out


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------


def select_thresholds(calib: pl.DataFrame, score_col: str = "aniso_mean") -> dict:
    """Youden-J / balanced-accuracy / EER / max-accuracy thresholds + cluster CIs."""
    y = calib["y"].to_numpy().astype(float)
    score = calib[score_col].to_numpy().astype(float)
    clusters = calib["scenario_id"].to_numpy()
    cw = M.weights_for(clusters, N_BOOT, SEED)
    pt = M.roc_analysis(y, score, M.unit_weights(len(y)))
    bt = M.roc_analysis(y, score, M.row_weights(cw, clusters))

    out = {"n": int(len(y)), "n_gen": int(y.sum()), "n_real": int((1 - y).sum()),
           "n_clusters": int(len(np.unique(clusters))), "criteria": {}}
    for crit in M.CRITERIA:
        thr = float(pt[f"thr_{crit}"][0])
        reps = bt[f"thr_{crit}"]
        entry = M.est_ci(thr, reps)
        entry["objective"] = M._f(pt[f"value_{crit}"][0])
        entry["objective_ci"] = M.est_ci(pt[f"value_{crit}"][0], bt[f"value_{crit}"])
        entry["published_0.048_in_ci"] = bool(
            entry["lo"] is not None and entry["lo"] <= M.PUBLISHED_THRESHOLD <= entry["hi"]
        )
        entry["published_percentile"] = float((reps <= M.PUBLISHED_THRESHOLD).mean())
        out["criteria"][crit] = entry
    out["auc"] = M.est_ci(pt["auc"][0], bt["auc"])
    out["ap"] = M.est_ci(pt["ap"][0], bt["ap"])
    out["youden_equals_bal_acc"] = bool(
        np.allclose(bt["thr_youden"], bt["thr_bal_acc"], equal_nan=True)
    )
    return out


# --------------------------------------------------------------------------
# Cross-validated threshold inside the E6 set
# --------------------------------------------------------------------------


def _point_thresholds(y, score) -> dict:
    """Threshold per criterion on one training split, no bootstrap (cheap)."""
    ra = M.roc_analysis(np.asarray(y, float), np.asarray(score, float), M.unit_weights(len(y)))
    return {c: float(ra[f"thr_{c}"][0]) for c in M.CRITERIA}


def cv_thresholds(sub: pl.DataFrame, n_splits: int = 5, seed: int = SEED,
                  score_col: str = "aniso_mean") -> dict:
    """Grouped 5-fold CV of the *threshold* inside one clip set.

    Why this exists: once the n10 re-encode only covers the 2000 E6 clips there
    is no external calibration fold left, so the threshold has to be fitted and
    tested inside that set.  Folds are grouped by ``scenario_id`` (real clips are
    singleton groups), stratified on the label where scikit-learn offers it, so
    a scene's weather variants never straddle the train/test boundary.

    The score itself (``aniso_mean``) is not fitted -- only the threshold is --
    so the *AUC* is a property of the whole set and does not change out of fold;
    what the CV buys is an honest out-of-fold operating point.  Both are
    reported, along with the per-fold thresholds, which is the real question:
    how stable is 0.048?

    Returns per-fold thresholds, the out-of-fold decision margin per criterion
    (``score - t_of_its_test_fold``, so thresholding it at 0 reproduces the
    out-of-fold decisions) and the whole-set threshold with a bootstrap CI.
    """
    y = sub["y"].to_numpy().astype(float)
    s = sub[score_col].to_numpy().astype(float)
    g = sub["scenario_id"].to_numpy()
    out: dict = {"n": int(len(y)), "n_gen": int(y.sum()), "n_real": int((1 - y).sum()),
                 "n_scenarios": int(len(np.unique(g))), "n_splits": n_splits, "seed": seed,
                 "score": score_col}
    if len(np.unique(g)) < n_splits or y.sum() < n_splits or (1 - y).sum() < n_splits:
        out["skipped"] = (f"need >= {n_splits} scenarios and >= {n_splits} clips of each class, "
                          f"have {len(np.unique(g))} scenarios / {int(y.sum())} gen / "
                          f"{int((1 - y).sum())} real")
        return out

    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(s.reshape(-1, 1), y.astype(int), groups=g))
        out["splitter"] = f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True, random_state={seed})"
    except Exception as exc:  # noqa: BLE001 -- very old sklearn
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(s.reshape(-1, 1), y.astype(int), groups=g))
        out["splitter"] = f"GroupKFold(n_splits={n_splits})  [{type(exc).__name__}]"

    margins = {c: np.full(len(y), np.nan) for c in M.CRITERIA}
    folds = []
    for k, (tr, te) in enumerate(splits):
        rec = {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
               "n_train_gen": int(y[tr].sum()), "n_test_gen": int(y[te].sum()),
               "n_test_real": int((1 - y[te]).sum())}
        if y[tr].sum() == 0 or (1 - y[tr]).sum() == 0:
            rec["degenerate"] = "training split has one class; fell back to 0.048"
            for c in M.CRITERIA:
                rec[f"t_{c}"] = M.PUBLISHED_THRESHOLD
                margins[c][te] = s[te] - M.PUBLISHED_THRESHOLD
        else:
            ths = _point_thresholds(y[tr], s[tr])
            for c, t in ths.items():
                rec[f"t_{c}"] = t
                margins[c][te] = s[te] - t
        # what that threshold does on its own held-out fold
        for c in M.CRITERIA:
            m = M.metrics_at(y[te], s[te], [rec[f"t_{c}"]], M.unit_weights(len(te)))
            rec[f"bal_acc_{c}"] = M._f(m["bal_acc"][0, 0])
            rec[f"tpr_{c}"] = M._f(m["tpr"][0, 0])
            rec[f"fpr_{c}"] = M._f(m["fpr"][0, 0])
        folds.append(rec)
    out["folds"] = folds
    for c in M.CRITERIA:
        ts = np.array([f[f"t_{c}"] for f in folds])
        out.setdefault("threshold_spread", {})[c] = {
            "mean": float(ts.mean()), "std": float(ts.std(ddof=1)) if len(ts) > 1 else 0.0,
            "min": float(ts.min()), "max": float(ts.max()),
            "published_inside_range": bool(ts.min() <= M.PUBLISHED_THRESHOLD <= ts.max()),
        }

    # out-of-fold operating points, with cluster-bootstrap CIs
    cw = M.weights_for(g, N_BOOT, SEED)
    w = M.row_weights(cw, g)
    out["oof"] = {}
    for c in M.CRITERIA:
        pt = M.metrics_at(y, margins[c], [0.0], M.unit_weights(len(y)))
        bt = M.metrics_at(y, margins[c], [0.0], w)
        out["oof"][c] = {m: M.est_ci(pt[m][0, 0], bt[m][:, 0]) for m in M.METRIC_NAMES}
    # the published threshold on the same clips, for the paired comparison
    pt = M.metrics_at(y, s, [M.PUBLISHED_THRESHOLD], M.unit_weights(len(y)))
    bt = M.metrics_at(y, s, [M.PUBLISHED_THRESHOLD], w)
    out["at_published"] = {m: M.est_ci(pt[m][0, 0], bt[m][:, 0]) for m in M.METRIC_NAMES}

    # AUC of the raw score over the whole set (threshold-free, so not fold-dependent)
    ra_p = M.roc_analysis(y, s, M.unit_weights(len(y)))
    ra_b = M.roc_analysis(y, s, w)
    out["auc"] = M.est_ci(ra_p["auc"][0], ra_b["auc"])
    out["ap"] = M.est_ci(ra_p["ap"][0], ra_b["ap"])
    out["auc_per_fold"] = []
    for k, (_, te) in enumerate(splits):
        if y[te].sum() == 0 or (1 - y[te]).sum() == 0:
            out["auc_per_fold"].append(None)
            continue
        out["auc_per_fold"].append(
            float(M.roc_analysis(y[te], s[te], M.unit_weights(len(te)))["auc"][0])
        )
    # whole-set threshold + CI: what you would ship if you refitted on everything
    whole = select_thresholds(sub, score_col=score_col)
    out["whole_set"] = whole["criteria"]
    out["margins"] = margins  # numpy, stripped before JSON
    return out


def print_cv(cv: dict, title: str) -> None:
    if "skipped" in cv:
        print(f"  {title}: SKIPPED ({cv['skipped']})")
        return
    print(f"  {title}: {cv['n']} clips ({cv['n_gen']} generated / {cv['n_real']} real), "
          f"{cv['n_scenarios']} scenarios, {cv['splitter']}")
    for c in ("youden", "eer", "acc_max"):
        sp = cv["threshold_spread"][c]
        ws = cv["whole_set"][c]
        oof = cv["oof"][c]
        print(f"    {c:<8} per-fold t {sp['mean']:.5f} +- {sp['std']:.5f} "
              f"[{sp['min']:.5f}, {sp['max']:.5f}]   whole-set t {ws['estimate']:.5f} "
              f"[{ws['lo']:.5f}, {ws['hi']:.5f}]  0.048 in CI: "
              f"{'yes' if ws['published_0.048_in_ci'] else 'NO'}")
        print(f"    {'':<8} out-of-fold: bal {_p(oof['bal_acc'])}  acc {_p(oof['acc'])}  "
              f"TPR {_p(oof['tpr'])}  FPR {_p(oof['fpr'])}  MCC {_p(oof['mcc'])}")
    ap = cv["at_published"]
    print(f"    {'0.048':<8} in-sample : bal {_p(ap['bal_acc'])}  acc {_p(ap['acc'])}  "
          f"TPR {_p(ap['tpr'])}  FPR {_p(ap['fpr'])}  MCC {_p(ap['mcc'])}")
    per_fold = [a for a in cv["auc_per_fold"] if a is not None]
    print(f"    AUC {_p(cv['auc'], 4)}  PR-AUC {_p(cv['ap'], 4)}"
          + (f"   per-fold AUC {np.mean(per_fold):.4f} +- {np.std(per_fold, ddof=1):.4f}"
             if len(per_fold) > 1 else ""))


# --------------------------------------------------------------------------
# Leave-one-generator-out
# --------------------------------------------------------------------------


def split_real_halves(real: pl.DataFrame, seed: int = SEED) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic 50/50 split of a real-clip frame (singleton scenarios)."""
    if real.is_empty():
        return real, real
    rng = np.random.default_rng(seed + 7)
    order = rng.permutation(real.height)
    half = real.height // 2
    return real[np.sort(order[:half])], real[np.sort(order[half:])]


def logo(df: pl.DataFrame, rows: Rows, real_tr: pl.DataFrame | None = None,
         real_te: pl.DataFrame | None = None) -> dict:
    """Calibrate on {one generator + real}, test on the other generators.

    Scenario ids are namespaced by generator (``dd/...`` vs ``p1/...``), so
    training on *every* clip of one generator cannot leak into a test fold made
    of another generator.  Real clips still have to be split: by default
    ``calib`` real trains and ``holdout`` + ``e6_test`` real is the test-side
    negative class; inside the E6-only set the caller passes two disjoint halves
    instead.

    The generator pairs are read off the frame, so the analysis follows whatever
    generators the manifest holds (currently dd <-> p1).
    """
    if real_tr is None:
        real_tr = df.filter((pl.col("label") == "real") & (pl.col("split") == "calib"))
    if real_te is None:
        real_te = df.filter((pl.col("label") == "real") & pl.col("split").is_in(["holdout", "e6_test"]))
    gens = sorted(df.filter(pl.col("label") == "generated")["generator"].unique().to_list())
    out: dict = {}
    for train_gen in gens:
        test_gens = [g for g in gens if g != train_gen]
        gen_tr = df.filter(pl.col("generator") == train_gen)
        if gen_tr.height < 10 or real_tr.height < 10:
            out[train_gen] = {
                "skipped": f"{gen_tr.height} {train_gen} clips and {real_tr.height} calib-real "
                f"clips have features (need >= 10 of each)"
            }
            continue
        tr = pl.concat([real_tr, gen_tr])
        sel = select_thresholds(tr)
        entry = {"train": {"generator": train_gen, "n_gen": gen_tr.height, "n_real": real_tr.height,
                           "thresholds": sel["criteria"], "auc": sel["auc"]}, "test": {}}
        ths = {"t0.048": M.PUBLISHED_THRESHOLD,
               "t_youden_logo": sel["criteria"]["youden"]["estimate"],
               "t_eer_logo": sel["criteria"]["eer"]["estimate"]}
        for tg in test_gens:
            gen_te = df.filter(pl.col("generator") == tg)
            if gen_te.height < 4 or real_te.height < 4:
                entry["test"][tg] = {"skipped": f"{gen_te.height} gen / {real_te.height} real rows"}
                continue
            sub = pl.concat([real_te, gen_te])
            entry["test"][tg] = eval_fold(rows, f"logo_train_{train_gen}_test_{tg}", sub, ths)
            entry["test"][tg].pop("_roc", None)
        out[train_gen] = entry
    return out


# --------------------------------------------------------------------------
# Multi-feature comparison
# --------------------------------------------------------------------------


def usable_features(df: pl.DataFrame) -> tuple[list[str], dict]:
    """Which of ``LR_FEATURES`` are present often enough to model with.

    Returns ``(features, {dropped feature: reason})``.  See
    :data:`MAX_FEATURE_NULL_FRACTION` -- the point is to drop a sparsely
    populated *column* rather than every *row* that lacks it.
    """
    feats, dropped = [], {}
    n = max(1, df.height)
    for c in LR_FEATURES:
        if c not in df.columns:
            dropped[c] = "column absent from the feature file"
            continue
        frac = df[c].null_count() / n
        if frac > MAX_FEATURE_NULL_FRACTION:
            dropped[c] = (f"null for {100 * frac:.1f}% of clips "
                          f"(extract_features.py --fast skips it); column dropped, rows kept")
            continue
        feats.append(c)
    return feats, dropped


def logistic_cv(df: pl.DataFrame, rows: Rows, out_dir: Path, n_splits: int = 5,
                seed: int = SEED) -> dict:
    """Same multi-feature comparison, but cross-validated inside the E6 set.

    Used when the condition only exists for the E6 clips, so there is no
    external calibration fold to fit on.  The model *is* fitted here (unlike the
    single-feature threshold), so the out-of-fold decision function is the only
    honest score: a model fitted and scored on the same 2000 clips would look
    far better than it is.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats, dropped = usable_features(fold(df, "e6_all"))
    d = fold(df, "e6_all").drop_nulls(subset=feats)
    info: dict = {"mode": "e6cv", "features": feats, "dropped_features": dropped,
                  "n_rows_after_dropna": d.height, "n_splits": n_splits}
    for c, why in dropped.items():
        print(f"  dropped feature {c}: {why}")
    if not usable(d) or d["y"].sum() < n_splits or (1 - d["y"]).sum() < n_splits:
        info["skipped"] = (f"{d.height} E6 rows with all features "
                           f"({int(d['y'].sum())} gen / {int((1 - d['y']).sum())} real)")
        pl.DataFrame(schema={"feature": pl.Utf8, "coef_standardised": pl.Float64}).write_parquet(
            out_dir / "lr_coefficients.parquet")
        return info

    from sklearn.model_selection import StratifiedGroupKFold

    x = d.select(feats).to_numpy()
    y = d["y"].to_numpy().astype(int)
    g = d["scenario_id"].to_numpy()
    splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed).split(x, y, groups=g))
    oof = np.full(len(y), np.nan)
    coefs = []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        pipe.fit(x[tr], y[tr])
        oof[te] = pipe.decision_function(x[te])
        coefs.append(pipe.named_steps["logisticregression"].coef_[0])
    if not coefs or np.isnan(oof).all():
        info["skipped"] = "every CV split was degenerate"
        return info

    full = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(x, y)
    coef_full = full.named_steps["logisticregression"].coef_[0]
    coef_arr = np.vstack(coefs)
    uni = []
    for i, f in enumerate(feats):
        s = d[f].to_numpy().astype(float)
        a = float(M.roc_analysis(y.astype(float), s, M.unit_weights(len(y)))["auc"][0])
        uni.append({"feature": f, "coef_standardised": float(coef_full[i]),
                    "coef_cv_mean": float(coef_arr[:, i].mean()),
                    "coef_cv_std": float(coef_arr[:, i].std(ddof=1)) if len(coefs) > 1 else 0.0,
                    "auc_e6_all": a, "auc_e6_all_oriented": max(a, 1.0 - a)})
    pl.DataFrame(uni).write_parquet(out_dir / "lr_coefficients.parquet")
    info["coefficients"] = uni

    keep = ~np.isnan(oof)
    sub = d.filter(pl.Series(keep)).with_columns(pl.Series("lr_score", oof[keep]))
    info["n_scored_out_of_fold"] = int(keep.sum())
    ev = eval_fold(rows, "e6_all_oof", sub, {"lr_t0": 0.0}, score_col="lr_score", tag="lr")
    ev.pop("_roc", None)
    info["out_of_fold"] = ev
    # in-sample AUC of the refitted model, purely to show the optimism gap
    ins = float(M.roc_analysis(y.astype(float), full.decision_function(x),
                               M.unit_weights(len(y)))["auc"][0])
    info["in_sample_auc_refit"] = ins
    return info


def logistic(df: pl.DataFrame, rows: Rows, out_dir: Path) -> dict:
    """Standardised logistic regression on the seven FFT features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats, dropped = usable_features(df)
    d = df.drop_nulls(subset=feats)
    calib = fold(d, "calib")
    info: dict = {"features": feats, "dropped_features": dropped,
                  "n_rows_after_dropna": d.height, "n_calib": calib.height}
    for c, why in dropped.items():
        print(f"  dropped feature {c}: {why}")
    if not usable(calib) or calib["y"].sum() < 5 or (1 - calib["y"]).sum() < 5:
        info["skipped"] = (
            f"calibration fold has {calib.height} rows with all seven features "
            f"({int(calib['y'].sum())} generated / {int((1 - calib['y']).sum())} real); need >= 5 of each"
        )
        pl.DataFrame(schema={"feature": pl.Utf8, "coef_standardised": pl.Float64}).write_parquet(
            out_dir / "lr_coefficients.parquet"
        )
        return info

    x = calib.select(feats).to_numpy()
    y = calib["y"].to_numpy().astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    pipe.fit(x, y)
    lr = pipe.named_steps["logisticregression"]
    scaler = pipe.named_steps["standardscaler"]
    coefs = lr.coef_[0]

    # univariate AUC of every feature, on calib and (if present) on e6_test
    uni = []
    for i, f in enumerate(feats):
        row = {"feature": f, "coef_standardised": float(coefs[i]),
               "mean_calib": float(scaler.mean_[i]), "scale_calib": float(scaler.scale_[i])}
        for fname in ("calib", "e6_test", "holdout"):
            sub = fold(d, fname)
            if not usable(sub):
                row[f"auc_{fname}"] = None
                continue
            s = sub[f].to_numpy().astype(float)
            yy = sub["y"].to_numpy().astype(float)
            a = float(M.roc_analysis(yy, s, M.unit_weights(len(yy)))["auc"][0])
            row[f"auc_{fname}"] = a
            row[f"auc_{fname}_oriented"] = max(a, 1.0 - a)
        uni.append(row)
    coef_df = pl.DataFrame(uni)
    coef_df.write_parquet(out_dir / "lr_coefficients.parquet")
    info["coefficients"] = coef_df.to_dicts()

    # decision function as the score; threshold picked on calib by Youden
    d = d.with_columns(pl.Series("lr_score", pipe.decision_function(d.select(feats).to_numpy())))
    calib = fold(d, "calib")
    sel = select_thresholds(calib, score_col="lr_score")
    info["calib_thresholds"] = sel["criteria"]
    info["calib_auc"] = sel["auc"]
    ths = {"lr_t0": 0.0, "lr_t_youden": sel["criteria"]["youden"]["estimate"]}
    info["folds"] = {}
    for fname in TWO_CLASS_FOLDS + GEN_ONLY_FOLDS:
        sub = fold(d, fname)
        if not usable(sub, need_both=fname in TWO_CLASS_FOLDS):
            info["folds"][fname] = {"skipped": f"{sub.height} rows"}
            continue
        r = eval_fold(rows, fname, sub, ths, score_col="lr_score", tag="lr")
        r.pop("_roc", None)
        info["folds"][fname] = r
    return info


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------


def sweep(df: pl.DataFrame, folds=("holdout", "e6_test", "calib")) -> pl.DataFrame:
    ts = np.round(np.arange(SWEEP_LO, SWEEP_HI + SWEEP_STEP / 2, SWEEP_STEP), 4)
    out = []
    for name in folds:
        sub = fold(df, name)
        if not usable(sub):
            continue
        y = sub["y"].to_numpy().astype(float)
        score = sub["aniso_mean"].to_numpy().astype(float)
        clusters = sub["scenario_id"].to_numpy()
        cw = M.weights_for(clusters, N_BOOT, SEED)
        pt = M.metrics_at(y, score, ts, M.unit_weights(len(y)))
        bt = M.metrics_at(y, score, ts, M.row_weights(cw, clusters))
        for ti, t in enumerate(ts):
            row = {"fold": name, "threshold": float(t), "n": int(len(y)),
                   "n_gen": int(y.sum()), "n_real": int((1 - y).sum())}
            for m in ("acc", "bal_acc", "tpr", "fpr", "f1", "mcc"):
                lo, hi = M.ci(bt[m][:, ti][:, None])
                row[m] = M._f(pt[m][0, ti])
                row[f"{m}_lo"] = M._f(lo[0])
                row[f"{m}_hi"] = M._f(hi[0])
            out.append(row)
    return pl.DataFrame(out) if out else pl.DataFrame(schema={"fold": pl.Utf8, "threshold": pl.Float64})


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _save(fig, stem: str) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        paths.append(rel_to_repo(p))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def fig_roc(evals: dict, condition: str) -> list[str]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    plotted = 0
    for i, (name, ev) in enumerate(evals.items()):
        roc = ev.get("_roc")
        if roc is None:
            continue
        c = palette[i % len(palette)]
        auc = ev.get("auc", {})
        lab = f"{name} (n={ev['n']}, AUC={auc.get('estimate', float('nan')):.3f}"
        if auc.get("lo") is not None:
            lab += f" [{auc['lo']:.3f},{auc['hi']:.3f}]"
        ax.plot(roc["fpr"], roc["tpr"], color=c, lw=1.6, label=lab + ")")
        ax.fill_between(roc["fpr"], roc["lo"], roc["hi"], color=c, alpha=0.18, lw=0)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return []
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("False positive rate (real called Generated)")
    ax.set_ylabel("True positive rate (generated found)")
    ax.set_title(f"FFT anisotropy ROC - condition '{condition}'", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, f"fft_roc_{condition}")


def fig_sweep(sw: pl.DataFrame, thresholds: dict, condition: str) -> list[str]:
    import matplotlib.pyplot as plt

    if not sw.height:
        return []
    present = set(sw["fold"])
    # prefer the test folds; fall back to whatever the design produced
    folds = [f for f in ("holdout", "e6_test", "e6_all") if f in present]
    if not folds:
        folds = sorted(present)[:2]
    fig, axes = plt.subplots(1, len(folds), figsize=(4.4 * len(folds), 3.6), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], folds):
        s = sw.filter(pl.col("fold") == name).sort("threshold")
        t = s["threshold"].to_numpy()
        for m, lab in (("acc", "accuracy"), ("tpr", "TPR (generated)"), ("fpr", "FPR (real)")):
            ax.plot(t, s[m].to_numpy(), color=COLORS[m if m in COLORS else "acc"], lw=1.5, label=lab)
            if f"{m}_lo" in s.columns:
                ax.fill_between(t, s[f"{m}_lo"].to_numpy(), s[f"{m}_hi"].to_numpy(),
                                color=COLORS[m if m in COLORS else "acc"], alpha=0.15, lw=0)
        # one vline per distinct threshold value; criteria that coincide share a label
        merged: dict[float, list[str]] = {}
        for label, tv in thresholds.items():
            if tv is None or not (SWEEP_LO <= tv <= SWEEP_HI):
                continue
            merged.setdefault(round(float(tv), 4), []).append(
                "published 0.048" if label == "t0.048" else label.removeprefix("t_")
            )
        for i, (tv, names) in enumerate(sorted(merged.items())):
            published = abs(tv - M.PUBLISHED_THRESHOLD) < 5e-4
            colour = COLORS["published"] if published else "#56B4E9"
            ax.axvline(tv, color=colour, ls="--" if published else "-.", lw=1.0)
            ax.text(tv, 1.015 + 0.055 * (i % 3), "+".join(names),
                    ha="center", fontsize=6.5, color=colour)
        ax.set_title(f"{name} (n={int(s['n'][0])})", fontsize=10, pad=22)
        ax.set_xlabel("anisotropy threshold t")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(0, 1.18)
    axes[0][0].set_ylabel("rate")
    axes[0][-1].legend(fontsize=7, loc="center right", frameon=False)
    fig.suptitle(f"Threshold sensitivity - condition '{condition}'", fontsize=10, y=1.12)
    return _save(fig, f"fft_threshold_sweep_{condition}")


def fig_hist(df: pl.DataFrame, thresholds: dict, condition: str) -> list[str]:
    import matplotlib.pyplot as plt

    groups = [("real", "real (CarCrash)"), ("dd", "generated: cosmos-drive-dreams"),
              ("dd_unlabelled", "generated: dd (unlabelled folders)"),
              ("p1", "generated: cosmos-predict1"), ("p1_unlabelled", "generated: p1 (unlabelled)")]
    present = [(g, lab) for g, lab in groups if df.filter(pl.col("pool_generator") == g).height >= 3]
    if not present:
        return []
    lo = float(df["aniso_mean"].min())
    hi = float(df["aniso_mean"].max())
    bins = np.linspace(lo, hi, 60)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for g, lab in present:
        v = df.filter(pl.col("pool_generator") == g)["aniso_mean"].to_numpy()
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6,
                color=COLORS.get(g, "#444444"), label=f"{lab} (n={len(v)})")
    ax.axvline(M.PUBLISHED_THRESHOLD, color=COLORS["published"], ls="--", lw=1.2, label="published t=0.048")
    for label, tv in thresholds.items():
        if label == "t0.048" or tv is None:
            continue
        ax.axvline(tv, color="#56B4E9", ls="-.", lw=1.0, label=f"{label}={tv:.4f}")
    ax.set_xlabel("mean anisotropy")
    ax.set_ylabel("density")
    ax.set_title(f"Anisotropy by pool - condition '{condition}'", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, f"fft_aniso_hist_{condition}")


def fig_heldout(gen_tpr: dict, condition: str) -> list[str]:
    import matplotlib.pyplot as plt

    keys, series, sizes = [], {}, {}
    for tname, entries in gen_tpr.items():
        for e in entries:
            k = (e["fold"], e["pool_generator"])
            if k not in keys:
                keys.append(k)
            series.setdefault(tname, {})[k] = e
            sizes[k] = e["n"]
    if not keys:
        return []
    keys.sort(key=lambda k: (k[1], k[0]))
    labels = [f"{g}\n{f}\nn={sizes[(f, g)]}" for f, g in keys]
    fig, ax = plt.subplots(figsize=(1.35 * max(4, len(keys)) + 1.5, 3.6))
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(series))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for i, (tname, per_gen) in enumerate(series.items()):
        vals = [per_gen.get(k, {}).get("estimate", np.nan) for k in keys]
        lo = [v - (per_gen.get(k, {}).get("lo") or v) for v, k in zip(vals, keys)]
        hi = [(per_gen.get(k, {}).get("hi") or v) - v for v, k in zip(vals, keys)]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92, label=tname,
               color=palette[i % len(palette)], yerr=[lo, hi], capsize=2,
               error_kw={"lw": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("TPR (fraction called Generated)")
    ax.set_title(f"Recall per generator pool - condition '{condition}'", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    return _save(fig, f"fft_heldout_generators_{condition}")


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------


def _fmt(res: dict | None, digits: int = 3) -> str:
    if not res or res.get("estimate") is None:
        return "--"
    e = res["estimate"]
    if res.get("lo") is None:
        return f"{e:.{digits}f}"
    return f"{e:.{digits}f} \\tiny[{res['lo']:.{digits}f},{res['hi']:.{digits}f}]"


def latex_table(summary: dict, condition: str, partial: bool) -> str:
    lines = [
        "% generated by src/fft/calibrate.py -- do not edit by hand",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lllrrrrrr}",
        "\\toprule",
        "fold & criterion & $t$ & $n$ & acc & bal.\\,acc & TPR & FPR & MCC \\\\",
        "\\midrule",
    ]
    for name, ev in summary.get("folds", {}).items():
        if "thresholds" not in ev:
            continue
        for tname, met in ev["thresholds"].items():
            t = summary["threshold_values"].get(tname)
            lines.append(
                " & ".join(
                    [
                        name.replace("_", "\\_"),
                        ("published" if tname == "t0.048" else tname.removeprefix("t_")).replace("_", "\\_"),
                        "--" if t is None else f"{t:.4f}",
                        str(ev["n"]),
                        _fmt(met.get("acc")),
                        _fmt(met.get("bal_acc")),
                        _fmt(met.get("tpr")),
                        _fmt(met.get("fpr")),
                        _fmt(met.get("mcc")),
                    ]
                )
                + " \\\\"
            )
        lines.append("\\addlinespace[2pt]")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{FFT anisotropy detector, condition \\texttt{"
        + condition
        + "}. Thresholds "
        + ("cross-validated inside the E6 set (grouped by \\texttt{scenario\\_id}); "
           if summary.get("design") == "e6cv"
           else "calibrated on the scenario-disjoint \\texttt{calib} fold; ")
        + "95\\% cluster-bootstrap intervals over \\texttt{scenario\\_id} (B="
        + str(N_BOOT)
        + ")."
        + (" \\textbf{Partial feature extraction -- numbers are not final.}" if partial else "")
        + "}",
        "\\label{tab:fft-calibration-" + condition + "}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------


def run(args) -> dict:
    out_dir: Path = args.out or RESULTS_FFT_DIR / f"calib_{args.condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_joined(args.features, args.condition, args.manifest, args.splits,
                     args.max_frames, args.frame_offset)

    spl = pl.read_parquet(args.splits)
    expected = spl.height
    expected_e6 = int(spl["in_e6"].sum())
    n_e6 = int(df["in_e6"].sum())
    e6_all = fold(df, "e6_all")
    calib = fold(df, "calib")
    calib_ok = bool(usable(calib) and calib["y"].sum() >= 3 and (1 - calib["y"]).sum() >= 3)

    # Which design applies?  'splits' needs an external calibration fold; when the
    # condition only exists for the 2000 E6 clips (which is how the n10 re-encode
    # is produced) there is none, so the threshold is cross-validated inside E6.
    mode = args.mode
    if mode == "auto":
        e6_ok = usable(e6_all) and e6_all["y"].sum() >= 20 and (1 - e6_all["y"]).sum() >= 20
        mode = "splits" if calib_ok else ("e6cv" if e6_ok and n_e6 >= 100 else "splits")
    print(f"\ndesign  : {mode}"
          + ("  (external scenario-disjoint calibration fold)" if mode == "splits"
             else f"  ({args.cv_folds}-fold grouped CV inside the {e6_all.height} E6 clips "
                  f"present -- no external calibration fold for this condition)"))
    if mode == "e6cv" and calib_ok:
        print("  note: a usable calib fold exists but --mode e6cv was requested")
    if mode == "splits" and not calib_ok:
        print(f"  note: the calib fold is not usable yet ({calib.height} rows, "
              f"{int(calib['y'].sum())} generated / {int((1 - calib['y']).sum())} real) and there "
              f"are only {n_e6} E6 rows, so thresholds cannot be fitted at all")

    partial = (n_e6 < expected_e6) if mode == "e6cv" else (df.height < expected)
    if partial:
        have, target = (n_e6, f"{expected_e6} E6") if mode == "e6cv" else (df.height, str(expected))
        print(f"\n!! PARTIAL: {have} of {target} clips have features for this "
              f"configuration; every number below is provisional.")

    rows = Rows(args.condition)
    summary: dict = {
        "condition": args.condition,
        "features": rel_to_repo(args.features),
        "max_frames": args.max_frames,
        "frame_offset": args.frame_offset,
        "n_boot": N_BOOT,
        "seed": SEED,
        "published_threshold": M.PUBLISHED_THRESHOLD,
        "partial": partial,
        "n_rows": df.height,
        "n_expected": expected,
        "rows_per_split": df.group_by("split").agg(
            pl.len().alias("n"), (pl.col("y") == 1).sum().alias("gen"),
            (pl.col("y") == 0).sum().alias("real")).sort("split").to_dicts(),
    }

    summary["design"] = mode
    summary["n_e6_rows"] = n_e6
    summary["n_e6_expected"] = expected_e6

    # ---- 1. threshold selection -----------------------------------------
    if mode == "e6cv":
        print(f"\n=== cross-validated threshold inside the E6 set ===")
        cv = cv_thresholds(e6_all, n_splits=args.cv_folds, seed=SEED)
        print_cv(cv, "e6_all")
        summary["cv"] = {k: v for k, v in cv.items() if k != "margins"}
        summary["calibration"] = {"skipped": "no external calibration fold for this condition; "
                                             "see 'cv' (grouped CV inside the E6 set)"}
        if "skipped" in cv:
            thresholds = {"t0.048": M.PUBLISHED_THRESHOLD}
        else:
            # 't_youden' keeps the same key the splits design uses, so robustness.py
            # and any downstream reader work unchanged.
            thresholds = {
                "t0.048": M.PUBLISHED_THRESHOLD,
                "t_youden": cv["whole_set"]["youden"]["estimate"],
                "t_eer": cv["whole_set"]["eer"]["estimate"],
                "t_acc_max": cv["whole_set"]["acc_max"]["estimate"],
                "t_cv_mean_youden": cv["threshold_spread"]["youden"]["mean"],
            }
        summary["threshold_values"] = thresholds
        two_class_folds = ["e6_all", "e6_dd_vs_real", "e6_p1_vs_real"]
        gen_only_folds = ["e6_sibling", "extra_gen", "dd_unlabelled"]
        breakdown_folds = ["e6_all"]
        sweep_folds = ("e6_all",)
    else:
        two_class_folds = list(TWO_CLASS_FOLDS)
        gen_only_folds = GEN_ONLY_FOLDS + ["dd_unlabelled"]
        breakdown_folds = ["e6_test", "holdout", "extra_gen", "e6_sibling"]
        sweep_folds = ("holdout", "e6_test", "calib")
        thresholds = _calib_thresholds(df, calib, calib_ok, summary)
    return _analyse(args, df, rows, summary, thresholds, mode, out_dir, partial,
                    two_class_folds, gen_only_folds, breakdown_folds, sweep_folds)


def _calib_thresholds(df: pl.DataFrame, calib: pl.DataFrame, calib_ok: bool, summary: dict) -> dict:
    """Section 1 of the splits design: fit the threshold on the calib fold."""
    print(f"\n=== threshold selection on 'calib' ({calib.height} clips, "
          f"{int(calib['y'].sum())} generated / {int((1 - calib['y']).sum())} real) ===")
    if calib_ok:
        sel = select_thresholds(calib)
        summary["calibration"] = sel
        for crit, e in sel["criteria"].items():
            inside = "yes" if e["published_0.048_in_ci"] else "NO"
            print(f"  {crit:<8} t = {e['estimate']:.5f}  95% CI [{e['lo']:.5f}, {e['hi']:.5f}]"
                  f"   objective {e['objective']:.4f}   0.048 inside CI: {inside}"
                  f"   (0.048 is at the {100 * e['published_percentile']:.1f}th pct of the replicates)")
        print(f"  calib AUC {sel['auc']['estimate']:.4f} "
              f"[{sel['auc']['lo']:.4f}, {sel['auc']['hi']:.4f}]")
        print(f"  Youden and balanced-accuracy thresholds identical: {sel['youden_equals_bal_acc']}")
        thresholds = {
            "t0.048": M.PUBLISHED_THRESHOLD,
            "t_youden": sel["criteria"]["youden"]["estimate"],
            "t_eer": sel["criteria"]["eer"]["estimate"],
            "t_acc_max": sel["criteria"]["acc_max"]["estimate"],
        }
    else:
        print("  SKIPPED: the calibration fold does not yet have both classes.")
        summary["calibration"] = {"skipped": "calib fold lacks both classes"}
        thresholds = {"t0.048": M.PUBLISHED_THRESHOLD}
    summary["threshold_values"] = thresholds
    return thresholds


def _analyse(args, df: pl.DataFrame, rows: Rows, summary: dict, thresholds: dict, mode: str,
             out_dir: Path, partial: bool, two_class_folds: list, gen_only_folds: list,
             breakdown_folds: list, sweep_folds: tuple) -> dict:
    """Sections 2-7, shared by the 'splits' and 'e6cv' designs."""
    # ---- 2. folds --------------------------------------------------------
    print("\n=== fold metrics ===")
    evals: dict = {}
    summary["folds"] = {}
    for name in two_class_folds + gen_only_folds:
        sub = fold(df, name)
        need_both = name in two_class_folds
        if not usable(sub, need_both=need_both):
            print(f"  {name:<14} SKIPPED ({sub.height} rows"
                  f"{', one class only' if sub.height else ''})")
            summary["folds"][name] = {"skipped": f"{sub.height} rows"}
            continue
        ev = eval_fold(rows, name, sub, thresholds)
        evals[name] = ev
        summary["folds"][name] = {k: v for k, v in ev.items() if k != "_roc"}
        for tname, met in ev["thresholds"].items():
            bits = [f"TPR {_p(met.get('tpr'))}"]
            if ev["two_class"]:
                bits = [f"acc {_p(met.get('acc'))}", f"bal {_p(met.get('bal_acc'))}",
                        f"TPR {_p(met.get('tpr'))}", f"FPR {_p(met.get('fpr'))}",
                        f"MCC {_p(met.get('mcc'))}"]
            print(f"  {name:<14} n={ev['n']:<5} {tname:<10} t={thresholds[tname]:.4f}  " + "  ".join(bits))
        if "auc" in ev:
            print(f"  {name:<14} AUC {_p(ev['auc'], 4)}   PR-AUC {_p(ev['ap'], 4)}")

    # ---- 3. breakdowns at the fixed thresholds ---------------------------
    print("\n=== breakdowns (fingerprint checks) ===")
    summary["breakdowns"] = {}
    gen_tpr: dict = {}
    for tname, tv in thresholds.items():
        per_gen = []
        for fname in breakdown_folds:
            sub = fold(df, fname)
            if sub.filter(pl.col("y") == 1).height < 3:
                continue
            for e in breakdown(rows, fname, sub, tv, tname, "pool_generator"):
                per_gen.append({**e, "fold": fname})
        gen_tpr[tname] = per_gen
        summary["breakdowns"][tname] = {"per_generator": per_gen}

    ref = "t_youden" if "t_youden" in thresholds else "t0.048"
    for by, side in (("weather", "gen"), ("resolution", "gen"), ("resolution", "real"),
                     ("folder", "gen"), ("codec", "gen"), ("codec", "real")):
        key = f"{by}_{side}"
        agg = []
        for fname in breakdown_folds:
            sub = fold(df, fname)
            for e in breakdown(rows, fname, sub, thresholds[ref], ref, by, side):
                agg.append({**e, "fold": fname})
        summary["breakdowns"].setdefault(ref, {})[key] = agg
        if agg:
            metric = "TPR" if side == "gen" else "FPR"
            print(f"  {key} @ {ref} ({metric}): {len(agg)} groups")
            for e in sorted(agg, key=lambda r: -r["n"])[:8]:
                print(f"    {str(e[by]):<12} {e['fold']:<10} n={e['n']:<5} "
                      f"{metric} {e['estimate']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]")

    # ---- 4. leave-one-generator-out -------------------------------------
    print("\n=== leave-one-generator-out ===")
    if mode == "e6cv":
        # No external calibration fold: split the E6 real clips (singleton
        # scenarios) into two disjoint halves, one to fit with, one to test on.
        e6 = fold(df, "e6_all")
        real_tr, real_te = split_real_halves(e6.filter(pl.col("label") == "real"))
        print(f"  (E6-internal: {real_tr.height} real clips train / {real_te.height} test)")
        summary["logo"] = logo(e6, rows, real_tr, real_te)
    else:
        summary["logo"] = logo(df, rows)
    for train_gen, entry in summary["logo"].items():
        if "skipped" in entry:
            print(f"  train={train_gen}: SKIPPED ({entry['skipped']})")
            continue
        yj = entry["train"]["thresholds"]["youden"]
        print(f"  train={train_gen} (+real): t_youden={yj['estimate']:.5f} "
              f"[{yj['lo']:.5f}, {yj['hi']:.5f}]  AUC {_p(entry['train']['auc'], 4)}")
        for tg, ev in entry["test"].items():
            if "skipped" in ev:
                print(f"    test={tg}: SKIPPED ({ev['skipped']})")
                continue
            print(f"    test={tg:<3} n={ev['n']:<5} AUC {_p(ev.get('auc'), 4)}  "
                  f"bal@0.048 {_p(ev['thresholds']['t0.048'].get('bal_acc'))}  "
                  f"bal@t_youden_logo {_p(ev['thresholds']['t_youden_logo'].get('bal_acc'))}")

    # ---- 5. multi-feature ------------------------------------------------
    print("\n=== multi-feature logistic regression ===")
    lr = logistic_cv(df, rows, out_dir, args.cv_folds) if mode == "e6cv" else logistic(df, rows, out_dir)
    summary["logistic"] = lr
    if "skipped" in lr:
        print(f"  SKIPPED: {lr['skipped']}")
    elif mode == "e6cv":
        for c in lr["coefficients"]:
            print(f"  {c['feature']:<22} coef {c['coef_standardised']:+.3f} "
                  f"(CV {c['coef_cv_mean']:+.3f} +- {c['coef_cv_std']:.3f})   "
                  f"univariate AUC {_n(c.get('auc_e6_all'))}")
        oof = lr.get("out_of_fold", {})
        print(f"  LR out-of-fold  n={oof.get('n', 0)}  AUC {_p(oof.get('auc'), 4)}  "
              f"bal@0 {_p((oof.get('thresholds') or {}).get('lr_t0', {}).get('bal_acc'))}   "
              f"(in-sample refit AUC {_n(lr.get('in_sample_auc_refit'), 4)} -- the optimism gap)")
    else:
        for c in lr["coefficients"]:
            print(f"  {c['feature']:<22} coef {c['coef_standardised']:+.3f}   "
                  f"univariate AUC calib {_n(c.get('auc_calib'))}  e6_test {_n(c.get('auc_e6_test'))}")
        for fname, ev in lr["folds"].items():
            if "skipped" in ev:
                continue
            met = ev["thresholds"]["lr_t_youden"]
            tail = (f"bal@youden {_p(met.get('bal_acc'))}" if ev["two_class"]
                    else f"TPR@youden {_p(met.get('tpr'))} (generated-only fold)")
            print(f"  LR {fname:<12} n={ev['n']:<5} AUC {_p(ev.get('auc'), 4)}  " + tail)

    # ---- 6. sweep --------------------------------------------------------
    sw = sweep(df, sweep_folds)
    sw.write_parquet(out_dir / "threshold_sweep.parquet")
    print(f"\n=== threshold sweep === {sw.height} rows "
          f"({', '.join(sorted(set(sw['fold']))) if sw.height else 'nothing to sweep'})")

    # ---- 7. artefacts ----------------------------------------------------
    roc_rows = []
    for name, ev in evals.items():
        roc = ev.get("_roc")
        if roc is None:
            continue
        roc_rows.append(pl.DataFrame({
            "condition": args.condition, "fold": name, "fpr": roc["fpr"],
            "tpr": roc["tpr"], "tpr_lo": roc["lo"], "tpr_hi": roc["hi"],
        }))
    roc_df = pl.concat(roc_rows) if roc_rows else pl.DataFrame(
        schema={"condition": pl.Utf8, "fold": pl.Utf8, "fpr": pl.Float64, "tpr": pl.Float64,
                "tpr_lo": pl.Float64, "tpr_hi": pl.Float64})
    roc_df.write_parquet(out_dir / "roc_points.parquet")
    rows.frame().write_parquet(out_dir / "fold_metrics.parquet")

    figs = []
    figs += fig_roc(evals, args.condition)
    figs += fig_sweep(sw, thresholds, args.condition)
    figs += fig_hist(df, thresholds, args.condition)
    figs += fig_heldout(gen_tpr, args.condition)
    summary["figures"] = figs

    (out_dir / "tab_fft_calibration.tex").write_text(latex_table(summary, args.condition, partial))
    (out_dir / "calibration_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))

    print(f"\nwrote {rel_to_repo(out_dir)}/: calibration_summary.json, fold_metrics.parquet, "
          f"threshold_sweep.parquet, roc_points.parquet, lr_coefficients.parquet, tab_fft_calibration.tex")
    for f in figs:
        print(f"  figure {f}")
    if partial:
        print("\n!! reminder: the feature file is incomplete, these numbers are provisional.")
    return summary


def _p(res: dict | None, digits: int = 3) -> str:
    if not res or res.get("estimate") is None:
        return "--"
    if res.get("lo") is None:
        return f"{res['estimate']:.{digits}f}"
    return f"{res['estimate']:.{digits}f}[{res['lo']:.{digits}f},{res['hi']:.{digits}f}]"


def _n(x, digits: int = 3) -> str:
    return "--" if x is None else f"{x:.{digits}f}"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "_roc"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------
# raw vs n10 comparison
# --------------------------------------------------------------------------


def compare(args) -> int:
    """Same clips, two conditions, side by side -- inside the E6 set.

    The n10 re-encode only exists for the 2000 E6 clips, so the comparison is
    restricted to the clips *both* conditions have **and** that are in E6.  Each
    condition gets its own grouped 5-fold CV threshold (there is no external
    calibration fold available for n10), and both are also scored at the
    published 0.048, which is the number the paper has to defend.
    """
    conds = args.compare
    features_dir = args.features_dir or RESULTS_FFT_DIR
    frames, missing = {}, []
    for c in conds:
        path = features_dir / f"features_{c}_f{args.max_frames}.parquet"
        if not path.exists():
            print(f"MISSING: {rel_to_repo(path)} -- run "
                  f"`extract_features.py --condition {c} --max-frames {args.max_frames}` first "
                  f"(for n10, restricted to results/fft/e6_clip_ids.txt)")
            missing.append(c)
            continue
        frames[c] = load_joined(path, c, args.manifest, args.splits, args.max_frames, 0, quiet=True)
        print(f"{c}: {frames[c].height} clips with features "
              f"({int(frames[c]['in_e6'].sum())} of them in the E6 set)")
    if missing:
        print(f"\ncannot compare: {', '.join(missing)} not available yet")
        return 1

    common = set(frames[conds[0]]["clip_id"])
    for c in conds[1:]:
        common &= set(frames[c]["clip_id"])
    frames = {c: d.filter(pl.col("clip_id").is_in(list(common)) & pl.col("in_e6"))
              for c, d in frames.items()}
    n_common = frames[conds[0]].height
    print(f"clips present in all of {conds} and in the E6 set: {n_common}")
    if n_common < 40:
        print("too few shared E6 clips to compare")
        return 1

    out: dict = {"conditions": list(conds), "n_common_e6_clips": n_common,
                 "design": "e6cv", "cv_folds": args.cv_folds, "n_boot": N_BOOT, "seed": SEED,
                 "published_threshold": M.PUBLISHED_THRESHOLD, "per_condition": {}}
    rows = Rows("compare")
    for c, d in frames.items():
        entry: dict = {"n": d.height, "n_gen": int(d["y"].sum()), "n_real": int((1 - d["y"]).sum())}
        cv = cv_thresholds(d, n_splits=args.cv_folds, seed=SEED)
        print(f"\n--- {c} ---")
        print_cv(cv, "shared E6 clips")
        entry["cv"] = {k: v for k, v in cv.items() if k != "margins"}
        ths = {"t0.048": M.PUBLISHED_THRESHOLD}
        if "skipped" not in cv:
            ths["t_youden"] = cv["whole_set"]["youden"]["estimate"]
            ths["t_eer"] = cv["whole_set"]["eer"]["estimate"]
        entry["folds"] = {}
        for fname in ("e6_all", "e6_dd_vs_real", "e6_p1_vs_real"):
            sub = fold(d, fname)
            if not usable(sub, need_both=True):
                entry["folds"][fname] = {"skipped": f"{sub.height} rows"}
                continue
            ev = eval_fold(rows, f"{c}:{fname}", sub, ths)
            ev.pop("_roc", None)
            entry["folds"][fname] = ev
        entry["per_generator_tpr"] = {
            tname: breakdown(rows, f"{c}:e6_all", d, tv, tname, "pool_generator")
            for tname, tv in ths.items()
        }
        entry["per_resolution"] = {
            "gen": breakdown(rows, f"{c}:e6_all", d, ths["t0.048"], "t0.048", "resolution", "gen"),
            "real": breakdown(rows, f"{c}:e6_all", d, ths["t0.048"], "t0.048", "resolution", "real"),
        }
        entry["class_medians"] = (
            d.group_by("pool_generator")
            .agg(pl.len().alias("n"), pl.col("aniso_mean").median().alias("median_aniso"),
                 pl.col("aniso_mean").quantile(0.25).alias("q25"),
                 pl.col("aniso_mean").quantile(0.75).alias("q75"))
            .sort("pool_generator")
            .to_dicts()
        )
        out["per_condition"][c] = entry
    dest_dir = args.out if args.out else RESULTS_FFT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows.frame().write_parquet(dest_dir / "compare_fold_metrics.parquet")

    fig = fig_compare_hist(frames, out, conds)
    out["figures"] = fig
    dest = (args.out if args.out else RESULTS_FFT_DIR) / "calib_compare.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_jsonable(out), indent=2))
    print(f"\nwrote {rel_to_repo(dest)}")
    for f in fig:
        print(f"  figure {f}")

    print("\n=== summary (same clips, both conditions) ===")
    for c, e in out["per_condition"].items():
        cv = e.get("cv", {})
        ws = (cv.get("whole_set") or {}).get("youden")
        oof = (cv.get("oof") or {}).get("youden", {})
        print(f"  {c}: refit t_youden={_n(ws['estimate'], 5) if ws else '--'}"
              + (f" [{ws['lo']:.5f},{ws['hi']:.5f}]" if ws else "")
              + f"  AUC {_p(cv.get('auc'), 4)}  out-of-fold bal {_p(oof.get('bal_acc'))}")
        for fname, ev in e["folds"].items():
            if "skipped" in ev:
                print(f"    {fname:<14} SKIPPED ({ev['skipped']})")
                continue
            print(f"    {fname:<14} AUC {_p(ev.get('auc'), 4)}  "
                  f"bal@0.048 {_p(ev['thresholds']['t0.048'].get('bal_acc'))}  "
                  f"TPR@0.048 {_p(ev['thresholds']['t0.048'].get('tpr'))}  "
                  f"FPR@0.048 {_p(ev['thresholds']['t0.048'].get('fpr'))}")
        for e2 in e["per_generator_tpr"].get("t0.048", []):
            print(f"      TPR@0.048 {e2['pool_generator']:<14} n={e2['n']:<5} "
                  f"{e2['estimate']:.3f} [{e2['lo']:.3f}, {e2['hi']:.3f}]")
        for e2 in e["per_resolution"]["real"]:
            print(f"      FPR@0.048 {str(e2['resolution']):<14} n={e2['n']:<5} "
                  f"{e2['estimate']:.3f} [{e2['lo']:.3f}, {e2['hi']:.3f}]")
    return 0


def fig_compare_hist(frames: dict, out: dict, conds) -> list[str]:
    import matplotlib.pyplot as plt

    groups = ["real", "dd", "dd_unlabelled", "p1"]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 3.4), sharex=True, sharey=True,
                             squeeze=False)
    allv = np.concatenate([d["aniso_mean"].to_numpy() for d in frames.values()])
    bins = np.linspace(float(allv.min()), float(allv.max()), 60)
    for ax, c in zip(axes[0], conds):
        d = frames[c]
        for g in groups:
            v = d.filter(pl.col("pool_generator") == g)["aniso_mean"].to_numpy()
            if len(v) < 3:
                continue
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.5,
                    color=COLORS.get(g, "#444444"), label=f"{g} (n={len(v)})")
        ax.axvline(M.PUBLISHED_THRESHOLD, color=COLORS["published"], ls="--", lw=1.1,
                   label="published 0.048" if c == conds[0] else None)
        cv = out["per_condition"][c].get("cv", {})
        t = (cv.get("whole_set") or {}).get("youden", {}).get("estimate")
        auc = (cv.get("auc") or {}).get("estimate")
        if t is not None:
            ax.axvline(t, color="#56B4E9", ls="-.", lw=1.1,
                       label="refit t" if c == conds[0] else None)
        ax.set_title(f"condition '{c}'" + (f"   AUC {auc:.3f}" if auc is not None else ""), fontsize=10)
        ax.set_xlabel("mean anisotropy")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("density")
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Anisotropy before and after codec/fps normalisation (same E6 clips)", fontsize=10)
    return _save(fig, "fft_aniso_hist_" + "_vs_".join(conds))


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--condition", default="raw")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--frame-offset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--features-dir", type=Path, default=None,
                    help="where --compare looks for features_<cond>_f<N>.parquet (default results/fft)")
    ap.add_argument("--compare", nargs="+", default=None, metavar="COND",
                    help="compare two conditions on their shared E6 clips, e.g. --compare raw n10")
    ap.add_argument("--mode", choices=("auto", "splits", "e6cv"), default="auto",
                    help="'splits' fits the threshold on the external calib fold; 'e6cv' "
                         "cross-validates it inside the 2000 E6 clips (what the n10 condition "
                         "needs, since it only exists for those); 'auto' (default) picks e6cv "
                         "when there is no usable calib fold")
    ap.add_argument("--cv-folds", type=int, default=5, help="folds for the e6cv design (default 5)")
    args = ap.parse_args()

    if args.compare:
        return compare(args)
    if args.features is None:
        args.features = RESULTS_FFT_DIR / f"features_{args.condition}_f{args.max_frames}.parquet"
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
