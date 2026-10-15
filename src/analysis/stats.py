"""Cluster-aware statistics for the DriveJudgeBench results.

Every accuracy quoted in the paper is a bare point estimate over items that are
*not* independent: each base driving scenario is re-rendered under up to eight
weather conditions ("_Snowy", "_Foggy", ...) and each rendered video is asked
several questions.  Treating the 7371 (video, question) rows as i.i.d. therefore
understates the uncertainty.  This module provides

  * ``add_cluster_ids``      video / scenario cluster ids derived from the path,
  * ``cluster_bootstrap``    a vectorised cluster (block) bootstrap,
  * ``macro_accuracy``       category-macro accuracy with coherent CIs,
  * ``wilson_ci``            a closed-form i.i.d. reference interval,
  * ``iid_bootstrap``        an i.i.d. bootstrap for width comparison,
  * ``paired_diff_ci``       paired between-model differences (+ exact McNemar),
  * ``class_metrics``        per-question-family recall / precision / F1 / MCC /
                             balanced accuracy / linear-weighted kappa.

Run ``python src/analysis/stats.py --selftest`` for the sanity checks.

Design note -- *coherent* replicates
------------------------------------
All bootstrap entry points accept (and ``run_stats.py`` reuses) a single
``ClusterWeights`` object: one ``(n_boot, n_clusters)`` matrix of multinomial
cluster weights drawn once per call to ``make_cluster_weights``.  Sharing it
across groups means the per-category, macro, per-source and overall replicates
come from the *same* resampled worlds, so e.g. the macro-average of the
per-category replicates is itself a valid replicate of the macro accuracy.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: results file -> canonical model id used everywhere in the paper.
#: NOTE: the ``model_id`` column inside ``Qwen_omni_analyzed.parquet`` has at
#: times been written as "Qwen2.5-Omni"; the filename map is authoritative.
MODEL_FILES: dict[str, str] = {
    "Cosmos_Reason_analyzed.parquet": "Cosmos-Reason",
    "InternVL3_5-30B_analyzed.parquet": "InternVL3.5-30B",
    "InternVL3_5-8B_analyzed.parquet": "InternVL3.5-8B",
    "Qwen3VL_analyzed.parquet": "Qwen3-VL",
    "Qwen_omni_analyzed.parquet": "Qwen3-Omni",
    "llava_onevision_analyzed.parquet": "LLaVA-OneVision",
    "v4o_analyzed.parquet": "Qwen3-VL+agent",
    "v4p_analyzed.parquet": "Qwen3-VL+agent+hints",
    "GPT-5.4-mini_analyzed.parquet": "GPT-5.4-mini",
    "Gemini_analyzed.parquet": "Gemini-3-Flash",
}

#: display / iteration order for the ten systems.
MODEL_ORDER: list[str] = [
    "Cosmos-Reason",
    "InternVL3.5-30B",
    "InternVL3.5-8B",
    "Qwen3-VL",
    "Qwen3-Omni",
    "LLaVA-OneVision",
    "Qwen3-VL+agent",
    "Qwen3-VL+agent+hints",
    "GPT-5.4-mini",
    "Gemini-3-Flash",
]

CLOSED_SOURCE = {"GPT-5.4-mini", "Gemini-3-Flash"}
AGENT_MODELS = {"Qwen3-VL+agent", "Qwen3-VL+agent+hints"}

#: canonical category order used in every figure and table.
CATEGORY_ORDER: list[str] = [
    "Visual understanding",
    "Spatial-temporal understanding",
    "Artifacts understanding",
    "Safety understanding",
    "Reality understanding",
    "Traffic laws understanding",
]

CATEGORY_SHORT = {
    "Visual understanding": "Visual",
    "Spatial-temporal understanding": "Spat-temp",
    "Artifacts understanding": "Artifacts",
    "Safety understanding": "Safety",
    "Reality understanding": "Reality",
    "Traffic laws understanding": "Traffic",
}

#: the ``artifacts`` folder holds the cosmos-predict1 clips, everything else is
#: cosmos-drive-dreams.
PREDICT1_FOLDERS = {"artifacts"}

#: weather re-renderings of one and the same base scenario.  Note the *space* in
#: "Golden hour" -- the suffix must be stripped off the stem with a regex, not
#: by splitting on "_".
WEATHER_RE = r"_(Original|Morning|Night|Snowy|Rainy|Foggy|Sunny|Golden hour)$"
SIZE_RE = r"_(7b|14b)$"

#: closed-source prompts appended the option list to the question text; strip it
#: so the question strings join against the in-house ones.
OPTION_SUFFIX_RE = r"\s*The answer is one of the following\s*:.*"

#: leading dataset prefix that some exports carry on ``video``.
SOURCE_PREFIX_RE = r"^(cosmos-drive-dreams|cosmos-predict1)/"

CANONICAL_COLS = [
    "model_id",
    "source",
    "folder",
    "video",
    "question",
    "category",
    "ground_truth",
    "vlm_answer",
    "correct",
    "video_id",
    "scenario_id",
]

# --------------------------------------------------------------------------
# Question families / label sets
# --------------------------------------------------------------------------

YES_NO = ["Yes", "No"]
REAL_GEN = ["Real", "Generated"]
LIKERT = ["1", "2", "3"]
OVERTAKE = ["Left", "Right", "It's not overtaking"]

FAMILY_LABELS: dict[str, list[str]] = {
    "yes_no": YES_NO,
    "real_generated": REAL_GEN,
    "likert": LIKERT,
    "overtake": OVERTAKE,
}

#: sentinel used for a prediction that is not one of the admissible labels
#: (an unparsed / free-form / "NO_PARSE" / "NO MATCH !" answer).  It never
#: equals a ground truth, i.e. unparsed answers count as wrong.
UNPARSED = "UNPARSED"


def question_family(question: str, gts: list[str] | None = None) -> str:
    """Map a question to its answer family."""
    q = question.strip().lower()
    if q.startswith("overall, how"):
        return "likert"
    if "real video or generated" in q:
        return "real_generated"
    if "which side is the ego car overtaking" in q:
        return "overtake"
    if gts:
        s = set(gts)
        for fam, labels in FAMILY_LABELS.items():
            if s <= set(labels):
                return fam
    return "yes_no"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def add_cluster_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``folder``, ``video_id`` and ``scenario_id`` columns.

    ``video`` is normalised to the bare ``<folder>/<file>.mp4`` form (a leading
    ``cosmos-drive-dreams/`` or ``cosmos-predict1/`` is stripped).

    * ``video_id``    == the normalised ``video`` (one clip = one cluster),
    * ``scenario_id`` == ``<folder>/<stem without weather / model-size suffix>``
      -- all weather re-renderings of a base scenario share one id.
    """
    df = df.with_columns(pl.col("video").str.replace(SOURCE_PREFIX_RE, "").alias("video"))
    df = df.with_columns(
        pl.col("video").str.split("/").list.first().alias("folder"),
        pl.col("video").str.split("/").list.last().str.replace(r"\.mp4$", "").alias("_stem"),
    )
    df = df.with_columns(pl.col("_stem").str.replace(WEATHER_RE, "").alias("_stem"))
    df = df.with_columns(pl.col("_stem").str.replace(SIZE_RE, "").alias("_stem"))
    df = df.with_columns(
        pl.col("video").alias("video_id"),
        (pl.col("folder") + "/" + pl.col("_stem")).alias("scenario_id"),
    )
    return df.drop("_stem")


def _derive_source(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("folder").is_in(list(PREDICT1_FOLDERS)))
        .then(pl.lit("cosmos-predict1"))
        .otherwise(pl.lit("cosmos-drive-dreams"))
        .alias("source")
    )


def load_model(path: str | Path, model_id: str | None = None) -> pl.DataFrame:
    """Load one ``*_analyzed.parquet`` into the canonical long format."""
    path = Path(path)
    if model_id is None:
        model_id = MODEL_FILES.get(path.name, path.stem)
    df = pl.read_parquet(path)
    df = df.with_columns(
        pl.col("question").str.replace(OPTION_SUFFIX_RE, "").str.strip_chars().alias("question"),
        pl.col("correct").cast(pl.Boolean),
        pl.lit(model_id).alias("model_id"),
    )
    df = add_cluster_ids(df)
    df = _derive_source(df)
    return df.select(CANONICAL_COLS)


def load_all_models(results_dir: str | Path = "results") -> pl.DataFrame:
    """Load every known results parquet into one long dataframe.

    Columns: ``model_id, source, folder, video, question, category,
    ground_truth, vlm_answer, correct, video_id, scenario_id``.
    """
    results_dir = Path(results_dir)
    frames = []
    for fname in MODEL_FILES:
        p = results_dir / fname
        if not p.exists():
            continue
        frames.append(load_model(p))
    if not frames:
        raise FileNotFoundError(f"no results parquet found under {results_dir}")
    out = pl.concat(frames, how="vertical")
    return out.with_columns(
        pl.col("model_id").cast(pl.Enum(MODEL_ORDER))
        if set(out["model_id"].unique()) <= set(MODEL_ORDER)
        else pl.col("model_id")
    ).with_columns(pl.col("model_id").cast(pl.Utf8))


# --------------------------------------------------------------------------
# Bootstrap machinery
# --------------------------------------------------------------------------


@dataclass
class ClusterWeights:
    """One shared draw of multinomial cluster weights.

    ``W[b, c]`` is how many times cluster ``c`` was drawn in replicate ``b``;
    each row sums to ``C`` (the number of clusters), i.e. every replicate
    resamples ``C`` clusters with replacement.
    """

    clusters: np.ndarray
    W: np.ndarray  # (n_boot, n_clusters) float64
    n_boot: int
    seed: int
    index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {c: i for i, c in enumerate(self.clusters.tolist())}

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)


def make_cluster_weights(clusters, n_boot: int = 2000, seed: int = 0) -> ClusterWeights:
    """Draw the ``(n_boot, n_clusters)`` multinomial weight matrix once."""
    clusters = np.unique(np.asarray(clusters))
    c = len(clusters)
    if c == 0:
        raise ValueError("no clusters")
    rng = np.random.default_rng(seed)
    w = rng.multinomial(c, np.full(c, 1.0 / c), size=n_boot).astype(np.float64)
    return ClusterWeights(clusters=clusters, W=w, n_boot=n_boot, seed=seed)


@dataclass
class BootResult:
    """Result of a (grouped) cluster bootstrap.

    ``table`` holds one row per group with ``n, n_clusters, estimate, lo, hi``;
    ``replicates`` is the ``(n_boot, n_groups)`` replicate matrix, column
    ``g`` matching row ``g`` of ``table``.
    """

    table: pl.DataFrame
    replicates: np.ndarray | None
    group_cols: list[str]
    conf: float = 0.95

    def _single(self) -> dict:
        if len(self.table) != 1:
            raise ValueError("BootResult holds more than one group")
        return self.table.row(0, named=True)

    @property
    def estimate(self) -> float:
        return float(self._single()["estimate"])

    @property
    def lo(self) -> float:
        return float(self._single()["lo"])

    @property
    def hi(self) -> float:
        return float(self._single()["hi"])

    @property
    def width(self) -> float:
        r = self._single()
        return float(r["hi"] - r["lo"])


def _percentiles(reps: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray]:
    a = (1.0 - conf) / 2.0 * 100.0
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(reps, a, axis=0)
        hi = np.nanpercentile(reps, 100.0 - a, axis=0)
    return lo, hi


def cluster_bootstrap(
    df_items: pl.DataFrame,
    value_col: str = "correct",
    cluster_col: str = "scenario_id",
    n_boot: int = 2000,
    seed: int = 0,
    group_cols: list[str] | None = None,
    weights: ClusterWeights | None = None,
    conf: float = 0.95,
    return_replicates: bool = True,
    group_order: list | None = None,
) -> BootResult:
    """Vectorised cluster bootstrap of ``mean(value_col)``.

    Items are first aggregated to per-``(cluster[, group])`` ``(n, sum)`` pairs;
    a replicate mean is then ``(W @ sums) / (W @ ns)`` -- i.e. resampling whole
    clusters with replacement, which is the correct unit of independence here.

    Pass the same ``weights`` object to several calls to obtain *coherent*
    replicates across groups (needed for macro accuracy).
    """
    group_cols = list(group_cols or [])
    d = df_items.select([*group_cols, cluster_col, value_col]).with_columns(
        pl.col(value_col).cast(pl.Float64)
    )
    if d.height == 0:
        raise ValueError("empty item frame")

    if weights is None:
        weights = make_cluster_weights(d[cluster_col].unique().to_numpy(), n_boot, seed)
    w = weights.W
    n_clusters_global = weights.n_clusters

    agg = d.group_by([*group_cols, cluster_col]).agg(
        pl.len().alias("_n"), pl.col(value_col).sum().alias("_s")
    )
    ci = agg[cluster_col].replace_strict(weights.index, default=None, return_dtype=pl.Int64)
    if ci.null_count():
        missing = agg.filter(ci.is_null())[cluster_col].unique().to_list()[:5]
        raise KeyError(f"clusters absent from the shared weight matrix: {missing}")
    agg = agg.with_columns(ci.alias("_ci"))

    if group_cols:
        keys = agg.select(group_cols).unique()
        if group_order is not None:
            order_map = {k: i for i, k in enumerate(group_order)}
            if len(group_cols) != 1:
                raise ValueError("group_order only supported for a single group column")
            keys = keys.filter(pl.col(group_cols[0]).is_in(list(order_map))).with_columns(
                pl.col(group_cols[0])
                .replace_strict(order_map, default=len(order_map), return_dtype=pl.Int64)
                .alias("_o")
            ).sort("_o").drop("_o")
        else:
            keys = keys.sort(group_cols)
        key_rows = keys.rows()
    else:
        keys = pl.DataFrame()
        key_rows = [()]

    g = len(key_rows)
    s_mat = np.zeros((n_clusters_global, g))
    n_mat = np.zeros((n_clusters_global, g))
    if group_cols:
        parts = agg.partition_by(group_cols, as_dict=True)
        for gi, key in enumerate(key_rows):
            part = parts.get(key)
            if part is None:
                continue
            idx = part["_ci"].to_numpy()
            s_mat[idx, gi] = part["_s"].to_numpy()
            n_mat[idx, gi] = part["_n"].to_numpy()
    else:
        idx = agg["_ci"].to_numpy()
        s_mat[idx, 0] = agg["_s"].to_numpy()
        n_mat[idx, 0] = agg["_n"].to_numpy()

    num = w @ s_mat  # (B, G)
    den = w @ n_mat
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = np.where(den > 0, num / den, np.nan)

    lo, hi = _percentiles(reps, conf)
    n_items = n_mat.sum(axis=0)
    n_cl = (n_mat > 0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = np.where(n_items > 0, s_mat.sum(axis=0) / n_items, np.nan)

    table = keys if group_cols else pl.DataFrame({"_dummy": [0]}).drop("_dummy")
    table = pl.DataFrame(
        {
            **{c: [row[i] for row in key_rows] for i, c in enumerate(group_cols)},
            "n": n_items.astype(np.int64),
            "n_clusters": n_cl.astype(np.int64),
            "estimate": point,
            "lo": lo,
            "hi": hi,
        }
    )
    return BootResult(
        table=table,
        replicates=reps if return_replicates else None,
        group_cols=group_cols,
        conf=conf,
    )


def macro_accuracy(
    df_items: pl.DataFrame,
    value_col: str = "correct",
    category_col: str = "category",
    categories: list[str] | None = None,
    cluster_col: str = "scenario_id",
    n_boot: int = 2000,
    seed: int = 0,
    weights: ClusterWeights | None = None,
    conf: float = 0.95,
) -> tuple[BootResult, BootResult]:
    """Category-macro accuracy: mean over the categories *inside* each replicate.

    Returns ``(per_category, macro)`` where both share the same cluster weights,
    so the macro replicate ``b`` is exactly the mean of the per-category
    replicates ``b``.
    """
    categories = list(categories or CATEGORY_ORDER)
    per_cat = cluster_bootstrap(
        df_items,
        value_col=value_col,
        cluster_col=cluster_col,
        n_boot=n_boot,
        seed=seed,
        group_cols=[category_col],
        weights=weights,
        conf=conf,
        group_order=categories,
    )
    reps = per_cat.replicates
    with np.errstate(invalid="ignore"):
        macro_reps = np.nanmean(reps, axis=1)
    lo, hi = _percentiles(macro_reps[:, None], conf)
    point = float(np.nanmean(per_cat.table["estimate"].to_numpy()))
    macro = BootResult(
        table=pl.DataFrame(
            {
                "n": [int(per_cat.table["n"].sum())],
                "n_clusters": [int(per_cat.table["n_clusters"].max())],
                "estimate": [point],
                "lo": [float(lo[0])],
                "hi": [float(hi[0])],
            }
        ),
        replicates=macro_reps[:, None],
        group_cols=[],
        conf=conf,
    )
    return per_cat, macro


# --------------------------------------------------------------------------
# i.i.d. references
# --------------------------------------------------------------------------


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (i.i.d. assumption)."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - (1.0 - conf) / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def iid_bootstrap_proportion(
    k: int, n: int, n_boot: int = 2000, seed: int = 0, conf: float = 0.95
) -> tuple[float, float, float, np.ndarray]:
    """i.i.d. bootstrap of a proportion.

    Resampling ``n`` items with replacement and counting successes is *exactly*
    ``Binomial(n, k/n)``, so no index sampling is needed.
    """
    if n == 0:
        return (float("nan"),) * 3 + (np.array([]),)
    rng = np.random.default_rng(seed)
    reps = rng.binomial(n, k / n, size=n_boot) / n
    lo, hi = _percentiles(reps[:, None], conf)
    return (k / n, float(lo[0]), float(hi[0]), reps)


def iid_bootstrap(
    values, n_boot: int = 2000, seed: int = 0, conf: float = 0.95, chunk: int = 256
) -> tuple[float, float, float, np.ndarray]:
    """Generic i.i.d. (item-level) bootstrap of a mean."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return (float("nan"),) * 3 + (np.array([]),)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        out[done : done + b] = v[idx].mean(axis=1)
        done += b
    lo, hi = _percentiles(out[:, None], conf)
    return (float(v.mean()), float(lo[0]), float(hi[0]), out)


# --------------------------------------------------------------------------
# Paired model comparison
# --------------------------------------------------------------------------

JOIN_KEYS = ["source", "video", "question"]


def paired_diff_ci(
    df_long: pl.DataFrame,
    model_a: str,
    model_b: str,
    cluster_col: str = "scenario_id",
    n_boot: int = 2000,
    seed: int = 0,
    weights: ClusterWeights | None = None,
    conf: float = 0.95,
) -> dict:
    """Paired difference in accuracy, ``acc(A) - acc(B)``, on the shared items.

    Cluster bootstrap of the per-item difference ``d = correct_a - correct_b``
    (same cluster weights as the marginal estimates when ``weights`` is
    supplied), plus an exact McNemar test on the discordant pairs as an i.i.d.
    reference.
    """
    a = df_long.filter(pl.col("model_id") == model_a).select(
        [*JOIN_KEYS, cluster_col, "correct"]
    )
    b = df_long.filter(pl.col("model_id") == model_b).select([*JOIN_KEYS, "correct"])
    j = a.join(b, on=JOIN_KEYS, how="inner", suffix="_b")
    n_shared = j.height
    if n_shared == 0:
        raise ValueError(f"no shared items between {model_a} and {model_b}")
    j = j.with_columns(
        (pl.col("correct").cast(pl.Int8) - pl.col("correct_b").cast(pl.Int8))
        .cast(pl.Float64)
        .alias("_d")
    )

    res = cluster_bootstrap(
        j,
        value_col="_d",
        cluster_col=cluster_col,
        n_boot=n_boot,
        seed=seed,
        weights=weights,
        conf=conf,
    )
    reps = res.replicates[:, 0]
    n_le = int(np.sum(reps <= 0))
    n_ge = int(np.sum(reps >= 0))
    b = reps.size
    p_boot = min(1.0, 2.0 * min(n_le, n_ge) / b)
    # (r + 1) / (B + 1) correction so the reported p is never exactly 0; with
    # B = 2000 the smallest attainable value is ~1e-3.
    p_boot_adj = min(1.0, 2.0 * (min(n_le, n_ge) + 1) / (b + 1))

    n_b = int(j.filter(pl.col("correct") & ~pl.col("correct_b")).height)
    n_c = int(j.filter(~pl.col("correct") & pl.col("correct_b")).height)
    if n_b + n_c == 0:
        mcnemar_p = 1.0
    else:
        from scipy.stats import binomtest

        mcnemar_p = float(binomtest(n_b, n_b + n_c, 0.5, alternative="two-sided").pvalue)

    return {
        "A": model_a,
        "B": model_b,
        "n_shared": n_shared,
        "acc_A": float(j["correct"].mean()),
        "acc_B": float(j["correct_b"].mean()),
        "diff": res.estimate,
        "lo": res.lo,
        "hi": res.hi,
        "p_boot": p_boot,
        "p_boot_adj": p_boot_adj,
        "n_boot": b,
        "n_discordant": n_b + n_c,
        "n_A_only": n_b,
        "n_B_only": n_c,
        "mcnemar_p": mcnemar_p,
        "n_clusters": int(res.table["n_clusters"][0]),
    }


# --------------------------------------------------------------------------
# Per-class metrics
# --------------------------------------------------------------------------


def _positive_class(gts: list[str], labels: list[str]) -> tuple[str, bool]:
    """Positive class = the *minority* ground-truth label of the question.

    Rationale: for every binary question in the benchmark the interesting event
    is the rare one (the violation / the generated clip / the odd scene).  We do
    not hard-code a per-question convention -- the minority GT label *is* the
    convention, and it is written to the output so the choice is auditable.
    A question whose GT is constant is flagged ``degenerate_gt`` and gets no
    MCC / balanced accuracy.
    """
    counts: dict[str, int] = {}
    for g in gts:
        counts[g] = counts.get(g, 0) + 1
    present = [lb for lb in labels if lb in counts]
    if len(present) <= 1:
        return (present[0] if present else labels[0]), True
    return min(present, key=lambda lb: (counts[lb], labels.index(lb))), False


def class_metrics(
    df_items: pl.DataFrame,
    group_cols: list[str] | None = None,
    min_n: int = 20,
) -> pl.DataFrame:
    """Per-(model, question) classification metrics.

    One row per ``(*group_cols, question, class)``; the summary metrics
    (accuracy, macro-F1, MCC, balanced accuracy, unparsed rate, and for the
    Likert questions linear-weighted kappa and MAE) are repeated on every row
    of the group so the parquet is trivial to filter.

    Predictions outside the question's admissible label set (``NO_PARSE``,
    ``NO MATCH !``, free-form text, ...) are mapped to ``UNPARSED``, count as
    wrong, and are reported separately as ``unparsed_rate``.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    group_cols = list(group_cols or ["model_id"])
    rows: list[dict] = []
    for key, part in df_items.partition_by([*group_cols, "question"], as_dict=True).items():
        question = key[-1]
        gts = part["ground_truth"].to_list()
        fam = question_family(question, gts)
        labels = FAMILY_LABELS[fam]
        n = len(gts)
        if n < min_n:
            continue
        pos, degenerate = _positive_class(gts, labels)
        y_true = np.array(gts)
        raw_pred = part["vlm_answer"].to_list()
        y_pred = np.array([p if p in labels else UNPARSED for p in raw_pred])
        unparsed = int((y_pred == UNPARSED).sum())

        present = [lb for lb in labels if lb in set(y_true.tolist())]
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        macro_f1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
        acc = float((y_true == y_pred).mean())
        if degenerate:
            mcc = float("nan")
            bal = float("nan")
        else:
            mcc = float(matthews_corrcoef(y_true, y_pred))
            bal = float(balanced_accuracy_score(y_true, y_pred))
        kappa = float("nan")
        mae = float("nan")
        if fam == "likert":
            ok = y_pred != UNPARSED
            if ok.sum() and len(present) > 1:
                kappa = float(
                    cohen_kappa_score(
                        y_true[ok], y_pred[ok], labels=labels, weights="linear"
                    )
                )
            # unparsed answers get the worst-case distance to the truth
            num = y_true.astype(int)
            pr = np.where(ok, np.where(ok, y_pred, "1"), "1").astype(int)
            dist = np.abs(num - pr).astype(float)
            worst = np.maximum(np.abs(num - 1), np.abs(num - 3)).astype(float)
            dist = np.where(ok, dist, worst)
            mae = float(dist.mean())

        base = {c: key[i] for i, c in enumerate(group_cols)}
        base.update(
            {
                "question": question,
                "family": fam,
                "category": part["category"][0],
                "n": n,
                "accuracy": acc,
                "unparsed": unparsed,
                "unparsed_rate": unparsed / n,
                "positive_class": pos,
                "degenerate_gt": degenerate,
                "macro_f1": macro_f1,
                "mcc": mcc,
                "balanced_accuracy": bal,
                "kappa_linear": kappa,
                "mae": mae,
            }
        )
        for i, lb in enumerate(labels):
            rows.append(
                {
                    **base,
                    "class": lb,
                    "is_positive": lb == pos,
                    "support": int(sup[i]),
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                    "n_predicted": int((y_pred == lb).sum()),
                }
            )
        rows.append(
            {
                **base,
                "class": UNPARSED,
                "is_positive": False,
                "support": 0,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "n_predicted": unparsed,
            }
        )
    out = pl.DataFrame(rows)
    return out.sort([*group_cols, "question", "class"])


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _selftest(results_dir: str | Path = "results", n_boot: int = 2000) -> None:
    print("=" * 72)
    print("stats.py --selftest")
    print("=" * 72)

    # ---- 1. cluster CI must be wider than the i.i.d. CI under clustering ----
    rng = np.random.default_rng(7)
    n_cl, per_cl = 120, 12
    cl_p = rng.beta(2, 2, size=n_cl)  # strong within-cluster correlation
    rows = []
    for c in range(n_cl):
        for _ in range(per_cl):
            rows.append(
                {
                    "scenario_id": f"c{c:03d}",
                    "correct": bool(rng.random() < cl_p[c]),
                    "category": CATEGORY_ORDER[c % 6],
                }
            )
    syn = pl.DataFrame(rows)
    cb = cluster_bootstrap(syn, n_boot=n_boot, seed=1)
    k = int(syn["correct"].sum())
    n = syn.height
    _, ilo, ihi, _ = iid_bootstrap_proportion(k, n, n_boot=n_boot, seed=1)
    wlo, whi = wilson_ci(k, n)
    print(
        f"[1] synthetic clustered data (n={n}, C={n_cl}, ICC>0):\n"
        f"    cluster CI  [{cb.lo:.4f}, {cb.hi:.4f}]  width {cb.width:.4f}\n"
        f"    iid CI      [{ilo:.4f}, {ihi:.4f}]  width {ihi - ilo:.4f}\n"
        f"    Wilson CI   [{wlo:.4f}, {whi:.4f}]  width {whi - wlo:.4f}"
    )
    assert cb.width > (ihi - ilo), "cluster CI should be wider than the iid CI"
    assert cb.width > (whi - wlo), "cluster CI should be wider than Wilson"
    print("    OK: cluster CI is wider than both i.i.d. references")

    # ---- 2. Wilson against a textbook value ----
    lo, hi = wilson_ci(50, 100)
    print(f"[2] wilson_ci(50, 100) = [{lo:.4f}, {hi:.4f}] (expect [0.4038, 0.5962])")
    assert abs(lo - 0.40383) < 1e-4 and abs(hi - 0.59617) < 1e-4
    lo2, hi2 = wilson_ci(0, 20)
    assert lo2 == 0.0 or abs(lo2) < 1e-12
    print("    OK")

    # ---- 3. macro accuracy is the mean of the per-category replicates ----
    w = make_cluster_weights(syn["scenario_id"].unique().to_numpy(), n_boot, seed=3)
    per_cat, macro = macro_accuracy(syn, weights=w, n_boot=n_boot, seed=3)
    assert np.allclose(
        macro.replicates[:, 0], np.nanmean(per_cat.replicates, axis=1), equal_nan=True
    )
    assert abs(macro.estimate - float(np.mean(per_cat.table["estimate"].to_numpy()))) < 1e-12
    print(
        f"[3] macro accuracy {macro.estimate:.4f} [{macro.lo:.4f}, {macro.hi:.4f}] "
        f"coherent with its 6 per-category replicates -- OK"
    )

    # ---- 4. a model paired against itself ----
    syn2 = syn.with_columns(
        pl.lit("A").alias("model_id"),
        pl.lit("cosmos-drive-dreams").alias("source"),
        pl.col("scenario_id").alias("video"),
        pl.int_range(pl.len()).cast(pl.Utf8).alias("question"),
    )
    both = pl.concat([syn2, syn2.with_columns(pl.lit("B").alias("model_id"))])
    d = paired_diff_ci(both, "A", "B", n_boot=n_boot, seed=5)
    print(
        f"[4] paired diff of a model against itself: diff={d['diff']:.6f} "
        f"CI=[{d['lo']:.6f}, {d['hi']:.6f}] p_boot={d['p_boot']:.3f} "
        f"mcnemar_p={d['mcnemar_p']:.3f} n_shared={d['n_shared']}"
    )
    assert d["diff"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0
    assert d["p_boot"] == 1.0 and d["mcnemar_p"] == 1.0
    print("    OK")

    # ---- 5. cluster ids on the real data ----
    results_dir = Path(results_dir)
    probe = results_dir / "Qwen3VL_analyzed.parquet"
    if probe.exists():
        df = load_model(probe)
        vids = df.select("source", "video", "video_id", "scenario_id").unique()
        print("[5] cluster ids on the real benchmark:")
        for src in ["cosmos-drive-dreams", "cosmos-predict1"]:
            s = vids.filter(pl.col("source") == src)
            print(
                f"    {src:22s} videos={s['video_id'].n_unique():5d}  "
                f"scenario clusters={s['scenario_id'].n_unique():5d}"
            )
        cdd = vids.filter(pl.col("source") == "cosmos-drive-dreams")
        cp1 = vids.filter(pl.col("source") == "cosmos-predict1")
        assert cdd["video_id"].n_unique() == 1103
        assert cp1["video_id"].n_unique() == 494
        assert 225 <= cdd["scenario_id"].n_unique() <= 240
        assert 195 <= cp1["scenario_id"].n_unique() <= 210
        print("    OK (expected ~233 drive-dreams / ~200 predict1 scenario clusters)")
    else:
        print(f"[5] skipped: {probe} not found")

    print("\nAll self-tests passed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.results_dir, args.n_boot)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
