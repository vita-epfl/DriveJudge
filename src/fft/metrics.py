"""
Cluster-bootstrapped detector metrics for the FFT study.

``src/analysis/stats.py`` already provides the project's cluster bootstrap, but
it bootstraps a *mean* (accuracy over rows).  Threshold selection, ROC-AUC,
PR-AUC, MCC and friends are not means, so this module reuses that module's
:func:`~analysis.stats.make_cluster_weights` -- the very same multinomial
``(n_boot, n_clusters)`` weight matrix, hence the very same resampled worlds --
and evaluates arbitrary confusion-matrix functionals under it.

Conventions
-----------
* ``y``      1 = *Generated* (the positive class), 0 = *Real*.
* ``score``  higher = more "generated" (``aniso_mean``, or a model's decision
  function).  A clip is declared Generated when ``score >= t``.
* Every replicate weights whole *clusters* (``scenario_id``), because the
  weather re-renderings of one scene are near duplicates.

Everything is vectorised over the ``B`` bootstrap replicates and chunked so the
intermediate ``(B, n)`` arrays stay small.

Equivalence note
----------------
Maximising Youden's J (``TPR - FPR``) and maximising balanced accuracy
(``(TPR + TNR) / 2``) are the *same* optimisation: ``BA = (J + 1) / 2``.  Both
criteria are reported because the reviewer asked for both, and they are expected
to coincide exactly; ``acc_max`` (plain accuracy, which does depend on class
prevalence) is the one that genuinely differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.stats import ClusterWeights, make_cluster_weights  # noqa: E402

#: the threshold hard-coded in ``src/agentic/fft/compute_fft.py``.
PUBLISHED_THRESHOLD = 0.048

#: threshold-selection criteria understood by :func:`roc_analysis`.
CRITERIA = ("youden", "bal_acc", "eer", "acc_max")

#: metrics returned by :func:`metrics_at`.
METRIC_NAMES = ("acc", "bal_acc", "tpr", "fpr", "precision", "f1", "mcc", "youden")


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def row_weights(cw: ClusterWeights, clusters, chunk: slice | None = None) -> np.ndarray:
    """Expand cluster weights to one weight per *row*.

    ``cw.index`` maps cluster -> column of ``cw.W``; a row inherits its
    cluster's multiplicity, which is exactly what resampling whole clusters
    means.  ``chunk`` selects a block of replicates so callers can stream.
    """
    idx = np.asarray([cw.index[c] for c in clusters], dtype=np.int64)
    w = cw.W if chunk is None else cw.W[chunk]
    return w[:, idx]


def unit_weights(n: int) -> np.ndarray:
    """``(1, n)`` weight matrix -- the point estimate as a one-replicate run."""
    return np.ones((1, n), dtype=np.float64)


def weights_for(clusters, n_boot: int = 2000, seed: int = 0) -> ClusterWeights:
    return make_cluster_weights(np.asarray(clusters), n_boot=n_boot, seed=seed)


def ci(reps: np.ndarray, conf: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval along the replicate axis (axis 0)."""
    a = (1.0 - conf) / 2.0 * 100.0
    with np.errstate(invalid="ignore"):
        return np.nanpercentile(reps, a, axis=0), np.nanpercentile(reps, 100.0 - a, axis=0)


def est_ci(point: float, reps: np.ndarray, conf: float = 0.95) -> dict:
    """``{estimate, lo, hi, se}`` for one scalar statistic."""
    reps = np.asarray(reps, dtype=np.float64).ravel()
    if reps.size == 0 or not np.isfinite(reps).any():
        return {"estimate": _f(point), "lo": None, "hi": None, "se": None}
    lo, hi = ci(reps[:, None], conf)
    return {
        "estimate": _f(point),
        "lo": _f(lo[0]),
        "hi": _f(hi[0]),
        "se": _f(np.nanstd(reps, ddof=1)),
    }


def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else x


# --------------------------------------------------------------------------
# Confusion-matrix metrics at fixed thresholds
# --------------------------------------------------------------------------


def counts_at(y, score, thresholds, W: np.ndarray) -> tuple[np.ndarray, ...]:
    """Weighted ``(TP, FP, TN, FN)``, each ``(B, T)``.

    ``W`` is a ``(B, n)`` row-weight matrix (use :func:`unit_weights` for the
    point estimate).  Declaring Generated is ``score >= t``.
    """
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    pred = (score[:, None] >= t[None, :]).astype(np.float64)  # (n, T)
    pos = y[:, None] * pred
    tp = W @ pos                                   # (B, T)
    fp = W @ (pred - pos)
    p_tot = (W @ y)[:, None]                       # (B, 1) total weight of positives
    n_tot = W.sum(axis=1)[:, None] - p_tot
    return tp, fp, n_tot - fp, p_tot - tp


def metrics_from_counts(tp, fp, tn, fn) -> dict[str, np.ndarray]:
    """The seven headline detector metrics from weighted confusion counts."""
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = tp + fn
        neg = tn + fp
        total = pos + neg
        tpr = np.where(pos > 0, tp / pos, np.nan)
        fpr = np.where(neg > 0, fp / neg, np.nan)
        prec = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        acc = np.where(total > 0, (tp + tn) / total, np.nan)
        bal = (tpr + (1.0 - fpr)) / 2.0
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)
    return {
        "acc": acc,
        "bal_acc": bal,
        "tpr": tpr,
        "fpr": fpr,
        "precision": prec,
        "f1": f1,
        "mcc": mcc,
        "youden": tpr - fpr,
    }


def metrics_at(y, score, thresholds, W: np.ndarray) -> dict[str, np.ndarray]:
    return metrics_from_counts(*counts_at(y, score, thresholds, W))


# --------------------------------------------------------------------------
# ROC / PR under cluster weights
# --------------------------------------------------------------------------


def _candidate_thresholds(u: np.ndarray) -> np.ndarray:
    """``G+1`` thresholds bracketing the ``G`` distinct scores.

    ``cand[j]`` declares Generated exactly the score groups ``j..G-1``; the
    interior values are midpoints between neighbouring scores so a calibrated
    threshold never sits *on* an observed value.
    """
    if u.size == 1:
        pad = max(abs(u[0]) * 1e-6, 1e-12)
        return np.array([u[0] - pad, u[0] + pad])
    mid = (u[:-1] + u[1:]) / 2.0
    pad = max((u[-1] - u[0]) * 1e-6, 1e-12)
    return np.concatenate([[u[0] - pad], mid, [u[-1] + pad]])


def roc_analysis(
    y,
    score,
    W: np.ndarray,
    fpr_grid: np.ndarray | None = None,
    chunk: int = 256,
    criteria: tuple[str, ...] = CRITERIA,
) -> dict:
    """One streaming pass giving AUC, PR-AUC, selected thresholds and a TPR band.

    Returns a dict of ``(B,)`` arrays (``auc``, ``ap``, ``thr_<criterion>``,
    ``value_<criterion>``) plus ``tpr_at_fpr`` ``(B, len(fpr_grid))`` when a grid
    is given, and -- for the *first* replicate only -- the full ``fpr``/``tpr``
    curve, which is what you plot when ``W`` is :func:`unit_weights`.

    Ties are handled at score-group granularity, so identical scores can never
    be split by the sweep.
    """
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    n = y.size
    if W.shape[1] != n:
        raise ValueError(f"weight matrix has {W.shape[1]} columns for {n} rows")

    order = np.argsort(score, kind="stable")
    s_sorted = score[order]
    y_sorted = y[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    u = s_sorted[starts]
    cand = _candidate_thresholds(u)  # (G+1,)

    b = W.shape[0]
    out = {
        "auc": np.full(b, np.nan),
        "ap": np.full(b, np.nan),
        "prevalence": np.full(b, np.nan),
    }
    for crit in criteria:
        out[f"thr_{crit}"] = np.full(b, np.nan)
        out[f"value_{crit}"] = np.full(b, np.nan)
    if fpr_grid is not None:
        out["tpr_at_fpr"] = np.full((b, len(fpr_grid)), np.nan)
    curve = None

    for lo in range(0, b, chunk):
        hi = min(lo + chunk, b)
        wo = W[lo:hi][:, order]
        pg = np.add.reduceat(wo * y_sorted, starts, axis=1)
        ng = np.add.reduceat(wo * (1.0 - y_sorted), starts, axis=1)
        zeros = np.zeros((hi - lo, 1))
        ptail = np.concatenate([np.cumsum(pg[:, ::-1], axis=1)[:, ::-1], zeros], axis=1)
        ntail = np.concatenate([np.cumsum(ng[:, ::-1], axis=1)[:, ::-1], zeros], axis=1)
        wpos = ptail[:, :1]
        wneg = ntail[:, :1]
        with np.errstate(invalid="ignore", divide="ignore"):
            tpr = np.where(wpos > 0, ptail / wpos, np.nan)
            fpr = np.where(wneg > 0, ntail / wneg, np.nan)
            prev = np.where((wpos + wneg) > 0, wpos / (wpos + wneg), np.nan)

        out["prevalence"][lo:hi] = prev[:, 0]
        out["auc"][lo:hi] = np.sum(
            (fpr[:, :-1] - fpr[:, 1:]) * (tpr[:, :-1] + tpr[:, 1:]) / 2.0, axis=1
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = ptail[:, :-1] + ntail[:, :-1]
            prec = np.where(denom > 0, ptail[:, :-1] / denom, np.nan)
        out["ap"][lo:hi] = np.nansum((tpr[:, :-1] - tpr[:, 1:]) * prec, axis=1)

        for crit in criteria:
            if crit in ("youden", "bal_acc"):
                obj = tpr - fpr
            elif crit == "eer":
                obj = -np.abs(fpr - (1.0 - tpr))
            elif crit == "acc_max":
                obj = tpr * prev + (1.0 - fpr) * (1.0 - prev)
            else:
                raise ValueError(f"unknown criterion {crit!r}")
            j = np.argmax(np.where(np.isnan(obj), -np.inf, obj), axis=1)
            rows_i = np.arange(hi - lo)
            out[f"thr_{crit}"][lo:hi] = cand[j]
            # Report each criterion in its own natural units: J for youden,
            # balanced accuracy for bal_acc, the equal error *rate* for eer,
            # accuracy for acc_max.
            if crit == "bal_acc":
                val = (obj[rows_i, j] + 1.0) / 2.0
            elif crit == "eer":
                val = (fpr[rows_i, j] + (1.0 - tpr[rows_i, j])) / 2.0
            else:
                val = obj[rows_i, j]
            out[f"value_{crit}"][lo:hi] = val

        if fpr_grid is not None:
            for i in range(hi - lo):
                out["tpr_at_fpr"][lo + i] = np.interp(fpr_grid, fpr[i, ::-1], tpr[i, ::-1])
        if curve is None:
            curve = (fpr[0].copy(), tpr[0].copy())

    out["thresholds"] = cand
    out["curve_fpr"], out["curve_tpr"] = curve
    return out


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------


def point_and_boot(
    y,
    score,
    clusters,
    thresholds,
    cw: ClusterWeights | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    conf: float = 0.95,
) -> dict:
    """Metrics at the given thresholds, with cluster-bootstrap CIs.

    ``{threshold: {metric: {estimate, lo, hi, se}}}`` plus ``n``, ``n_pos``,
    ``n_neg``, ``n_clusters``.
    """
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    clusters = np.asarray(clusters)
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))

    if cw is None:
        cw = weights_for(clusters, n_boot=n_boot, seed=seed)
    w_boot = row_weights(cw, clusters)

    point = metrics_at(y, score, thresholds, unit_weights(y.size))
    reps = metrics_at(y, score, thresholds, w_boot)

    out = {
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_clusters": int(len(np.unique(clusters))),
        "thresholds": {},
    }
    for ti, t in enumerate(thresholds):
        out["thresholds"][f"{t:.6g}"] = {
            m: est_ci(point[m][0, ti], reps[m][:, ti], conf) for m in METRIC_NAMES
        }
    return out
