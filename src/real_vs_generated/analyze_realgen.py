"""Accuracy analysis for the real/generated studies (v1 100-clip, v2 balanced 2000).

Reads every ``*_realgen*.json`` file in ``--results-dir`` and joins each entry
against a ground-truth JSON on the video path relative to ``final_dataset``.

Handles both result shapes:
  - Baseline shape: ``{video, question, answer}`` where ``answer`` is the raw
    "Feedback:::\\nEvaluation: …\\nAnswer: X" text emitted by the prompt
    template.  The final answer is the LAST ``Answer:`` occurrence, normalised
    to Real / Generated.
  - Agent / closed-source shape: ``{video, question, evaluation, answer,
    history, …}`` where ``answer`` is already the clean final token.

An unparseable response (no Real/Generated token after normalisation) counts as
**wrong** and is reported separately as ``unparsed``.

Two modes
---------
``--mode v1`` (default, unchanged behaviour)
    prints the legacy table and writes ``realgen_accuracy_summary.json``.

``--mode v2``
    adds, per model, the metrics a balanced two-class benchmark actually needs
    -- accuracy, **balanced accuracy**, per-class recall (Real / Generated),
    precision / F1 / MCC with *Generated* as the positive class -- each with a
    95 % **cluster bootstrap** CI (``cluster_col=scenario_id``; real clips are
    singleton clusters, generated clips cluster over the weather / model-size
    re-renderings of one scene).  It also evaluates the trivial baselines a
    reviewer will ask about:

    ==================  =====================================================
    ``always-Real``     the degenerate policy the v1 study exposed
    ``always-Generated``the policy that wins the all-generated reality set
    ``random-0.5``      seeded coin flip
    ``fft-threshold``   ``aniso_mean >= --fft-threshold`` -> Generated, from
                        ``results/fft/features_*.parquet`` (one row per FFT
                        condition, e.g. ``raw`` and ``n10``)
    ==================  =====================================================

    Outputs: ``results/realgen/realgen_metrics_v2.parquet``,
    ``realgen_summary_v2.json``, ``realgen_v2.tex`` and the figure
    ``report/figures/fig_realgen_balanced.pdf`` (+ ``.png``).

Examples
--------
    # v1 reproduction (the historical numbers)
    .venv/bin/python src/real_vs_generated/analyze_realgen.py

    # v2, full set
    .venv/bin/python src/real_vs_generated/analyze_realgen.py --mode v2 \
        --gt src/real_vs_generated/v2/groundtruth_realgen_v2.json \
        --results-dir src/real_vs_generated/v2/results \
        --out results/realgen

    # v2, pilot subset only
    .venv/bin/python src/real_vs_generated/analyze_realgen.py --mode v2 \
        --gt src/real_vs_generated/v2/groundtruth_realgen_v2.json \
        --results-dir src/real_vs_generated/v2/results --pilot-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_GT = ROOT / "groundtruth_realgen.json"
DEFAULT_V2_OUT = REPO / "results" / "realgen"
FFT_DIR = REPO / "results" / "fft"
FIG_DIR = REPO / "report" / "figures"

#: the operating point calibrated on the original 18+18 FFT clips.
DEFAULT_FFT_THRESHOLD = 0.048
FFT_FEATURE = "aniso_mean"

# Optional pretty model labels for known files; anything else falls back to the
# stem of the filename without the `_realgen*` suffix.
MODEL_LABELS = {
    "Cosmos_Reason_realgen.json":            "Cosmos-Reason",
    "InternVL3_5-30B_realgen.json":          "InternVL3.5-30B",
    "InternVL3_5-8B_realgen.json":           "InternVL3.5-8B",
    "Qwen3VL_realgen.json":                  "Qwen3-VL",
    "Qwen3VL-hints_realgen.json":            "Qwen3-VL+hints",
    "Qwen_omni_realgen.json":                "Qwen3-Omni",
    "llava_onevision_realgen.json":          "LLaVA-OneVision",
    "Qwen3VL_agent_realgen.json":            "Qwen3-VL+agent",
    "Qwen3VL_agent_hints_realgen.json":      "Qwen3-VL+agent+hints",
    # ---- v2 ----
    "Cosmos_Reason_realgen_v2.json":         "Cosmos-Reason",
    "InternVL3_5-30B_realgen_v2.json":       "InternVL3.5-30B",
    "InternVL3_5-8B_realgen_v2.json":        "InternVL3.5-8B",
    "Qwen3VL_realgen_v2.json":               "Qwen3-VL",
    "Qwen3VL_hints_realgen_v2.json":         "Qwen3-VL+hints",
    "Qwen_omni_realgen_v2.json":             "Qwen3-Omni",
    "llava_onevision_realgen_v2.json":       "LLaVA-OneVision",
    "v4o_realgen_v2.json":                   "Qwen3-VL+agent",
    "v4p_realgen_v2.json":                   "Qwen3-VL+agent+hints",
    "GPT-5.4-mini_realgen_v2.json":          "GPT-5.4-mini",
    "Gemini_realgen_v2.json":                "Gemini-3-Flash",
}

BASELINE_LABELS = ("always-Real", "always-Generated", "random-0.5")

ANSWER_RE = re.compile(r"Answer\s*:\s*([^\n<]+)", re.IGNORECASE)
PATH_RE = re.compile(r"final_dataset/(.+)$")

#: same convention as src/analysis/stats.py / src/fft/common.py
WEATHER_RE = r"_(Original|Morning|Night|Snowy|Rainy|Foggy|Sunny|Golden hour)$"
SIZE_RE = r"_(7b|14b)$"
GENERATOR_OF_SOURCE = {
    "cosmos-drive-dreams": "dd",
    "cosmos-predict1": "p1",
    "cosmos-predict2": "p2",
    "real": "real",
}


# --------------------------------------------------------------------------
# Parsing helpers (v1 behaviour, unchanged)
# --------------------------------------------------------------------------


def relative_video(abs_path: str) -> str:
    """Convert /mnt/.../final_dataset/<source>/<rest>.mp4 -> <source>/<rest>.mp4."""
    if abs_path is None:
        return ""
    m = PATH_RE.search(abs_path)
    return m.group(1) if m else abs_path


def normalise_token(text: str) -> str | None:
    """Return 'Real' or 'Generated' if the text contains one of those tokens
    (case-insensitive, last occurrence wins so chain-of-thought references
    don't shadow the final verdict). Otherwise None.
    """
    if not text:
        return None
    t = text.strip().strip(" .\t\r\n*\"'")
    low = t.lower()
    if low == "real":
        return "Real"
    if low == "generated":
        return "Generated"
    last_real = low.rfind("real")
    last_gen = low.rfind("generated")
    if last_real == -1 and last_gen == -1:
        return None
    return "Generated" if last_gen > last_real else "Real"


def extract_answer(entry: dict) -> str | None:
    """Return the model's final Real/Generated verdict, or None if unparseable."""
    raw = entry.get("answer")
    if isinstance(raw, str) and len(raw) <= 60 and ("evaluation" in entry or "history" in entry):
        tok = normalise_token(raw)
        if tok is not None:
            return tok
        ev = entry.get("evaluation")
        if isinstance(ev, str):
            matches = list(ANSWER_RE.finditer(ev))
            if matches:
                tok = normalise_token(matches[-1].group(1))
                if tok is not None:
                    return tok
            return normalise_token(ev)
        return None
    if isinstance(raw, str):
        matches = list(ANSWER_RE.finditer(raw))
        if matches:
            return normalise_token(matches[-1].group(1))
        return normalise_token(raw)
    return None


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def derive_scenario_id(rel_video: str) -> str:
    """``<source>/<folder>/<stem>.mp4`` -> ``<gen>/<folder>/<scenario>``.

    Same rule as ``src/fft/common.parse_clip_name`` / ``stats.add_cluster_ids``:
    the weather suffix and then the ``_7b`` / ``_14b`` model-size suffix are
    stripped, so all re-renderings of one scene share a cluster id.  Real clips
    are singleton clusters.
    """
    parts = rel_video.split("/")
    source = parts[0]
    gen = GENERATOR_OF_SOURCE.get(source, source)
    folder = parts[1] if len(parts) >= 3 else None
    stem = parts[-1]
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    if gen in ("real", "real_crash"):
        return f"{gen}/{stem}"
    stem = re.sub(WEATHER_RE, "", stem)
    if gen in ("p1", "p2"):
        stem = re.sub(SIZE_RE, "", stem)
    return f"{gen}/{folder}/{stem}" if folder else f"{gen}/{stem}"


def derive_clip_id(rel_video: str) -> str:
    parts = rel_video.split("/")
    gen = GENERATOR_OF_SOURCE.get(parts[0], parts[0])
    stem = parts[-1]
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    folder = parts[1] if len(parts) >= 3 else None
    return f"{gen}/{folder}/{stem}" if folder else f"{gen}/{stem}"


def load_gt(gt_path: Path, pilot_only: bool = False) -> tuple[dict[str, dict], list[dict]]:
    """Load a ground-truth JSON (v1 or v2) into ``rel_video -> record``.

    v1 records only carry ``video / rel_video / source / ground_truth``; the
    cluster and clip ids are derived from the path so that both studies go
    through exactly the same statistics.
    """
    with open(gt_path) as f:
        records = json.load(f)
    out = []
    for r in records:
        rec = dict(r)
        rec.setdefault("rel_video", relative_video(rec.get("video", "")))
        rec.setdefault("scenario_id", derive_scenario_id(rec["rel_video"]))
        rec.setdefault("clip_id", derive_clip_id(rec["rel_video"]))
        rec.setdefault("in_pilot", True)
        rec.setdefault("category", rec.get("folder") or rec["source"])
        if pilot_only and not rec["in_pilot"]:
            continue
        out.append(rec)
    return {r["rel_video"]: r for r in out}, out


# --------------------------------------------------------------------------
# v1 evaluation (kept byte-for-byte compatible)
# --------------------------------------------------------------------------


def evaluate_one(path: Path, gt_map: dict[str, dict]) -> dict:
    with open(path) as f:
        entries = json.load(f)
    per_gt = defaultdict(lambda: {"n": 0, "correct": 0, "unparseable": 0})
    total_n = total_correct = total_unparseable = skipped_no_gt = 0

    for e in entries:
        rel = relative_video(e.get("video", ""))
        if rel not in gt_map:
            skipped_no_gt += 1
            continue
        gt = gt_map[rel]["ground_truth"]
        pred = extract_answer(e)
        unparseable = pred is None
        correct = (pred == gt) and (pred is not None)

        total_n += 1
        total_correct += int(correct)
        total_unparseable += int(unparseable)
        per_gt[gt]["n"] += 1
        per_gt[gt]["correct"] += int(correct)
        per_gt[gt]["unparseable"] += int(unparseable)

    return {
        "n": total_n,
        "correct": total_correct,
        "accuracy": (total_correct / total_n) if total_n else 0.0,
        "unparseable": total_unparseable,
        "skipped_no_gt": skipped_no_gt,
        "per_gt": {
            gt: {
                "n": s["n"],
                "correct": s["correct"],
                "accuracy": (s["correct"] / s["n"]) if s["n"] else 0.0,
                "unparseable": s["unparseable"],
            }
            for gt, s in per_gt.items()
        },
    }


def run_v1(args) -> int:
    gt_map, gt_records = load_gt(args.gt, args.pilot_only)

    gt_counts: dict[str, int] = defaultdict(int)
    for r in gt_records:
        gt_counts[r["ground_truth"]] += 1
    print(f"Ground truth ({len(gt_records)} videos): "
          + ", ".join(f"{k}={v}" for k, v in sorted(gt_counts.items())))
    print()

    result_files = find_results(args.results_dir)
    if not result_files:
        print(f"No *_realgen*.json files in {args.results_dir}")
        return 1

    summary = {}
    print(f"{'Model':24s} {'n':>4s} {'acc':>7s} {'Real acc':>10s} {'Gen acc':>10s} {'unparse':>8s}")
    print("-" * 70)
    for path in result_files:
        label = model_label(path)
        try:
            r = evaluate_one(path, gt_map)
        except Exception as exc:
            print(f"{label:24s}  ERROR: {exc}")
            continue
        real = r["per_gt"].get("Real", {"n": 0, "correct": 0, "accuracy": 0.0, "unparseable": 0})
        gen = r["per_gt"].get("Generated", {"n": 0, "correct": 0, "accuracy": 0.0, "unparseable": 0})
        print(f"{label:24s} {r['n']:>4d} "
              f"{r['accuracy']*100:>6.1f}% "
              f"{real['accuracy']*100:>8.1f}% ({real['correct']}/{real['n']}) "
              f"{gen['accuracy']*100:>5.1f}% ({gen['correct']}/{gen['n']}) "
              f"{r['unparseable']:>7d}")
        summary[label] = {"file": path.name, **r}

    out_path = Path(args.summary_out) if args.summary_out else ROOT / "realgen_accuracy_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


# --------------------------------------------------------------------------
# v2: per-item frame + metrics
# --------------------------------------------------------------------------


def find_results(results_dir: Path) -> list[Path]:
    return sorted(p for p in Path(results_dir).glob("*_realgen*.json") if p.is_file())


def model_label(path: Path) -> str:
    if path.name in MODEL_LABELS:
        return MODEL_LABELS[path.name]
    return re.sub(r"_realgen(_v2)?(_pilot)?$", "", path.stem)


def items_frame(path: Path, gt_map: dict[str, dict], label: str):
    """One row per scored clip: ``pred`` in {Real, Generated, UNPARSED}."""
    import polars as pl

    with open(path) as f:
        entries = json.load(f)
    rows = []
    skipped = 0
    for e in entries:
        rel = relative_video(e.get("video", ""))
        rec = gt_map.get(rel)
        if rec is None:
            skipped += 1
            continue
        pred = extract_answer(e)
        rows.append({
            "model": label,
            "rel_video": rel,
            "clip_id": rec["clip_id"],
            "scenario_id": rec["scenario_id"],
            "source": rec["source"],
            "category": rec.get("category"),
            "in_pilot": bool(rec.get("in_pilot", True)),
            "ground_truth": rec["ground_truth"],
            "pred": pred if pred is not None else "UNPARSED",
            "unparsed": pred is None,
            "correct": bool(pred == rec["ground_truth"]),
        })
    if not rows:
        raise ValueError(f"no rows of {path.name} joined against the ground truth")
    return pl.DataFrame(rows), skipped


def baseline_frames(gt_records: list[dict], seed: int = 42):
    """always-Real / always-Generated / seeded coin flip, over the same clips."""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(seed)
    coin = rng.integers(0, 2, size=len(gt_records))
    frames = []
    for label in BASELINE_LABELS:
        rows = []
        for i, rec in enumerate(gt_records):
            if label == "always-Real":
                pred = "Real"
            elif label == "always-Generated":
                pred = "Generated"
            else:
                pred = "Generated" if coin[i] else "Real"
            rows.append({
                "model": label,
                "rel_video": rec["rel_video"],
                "clip_id": rec["clip_id"],
                "scenario_id": rec["scenario_id"],
                "source": rec["source"],
                "category": rec.get("category"),
                "in_pilot": bool(rec.get("in_pilot", True)),
                "ground_truth": rec["ground_truth"],
                "pred": pred,
                "unparsed": False,
                "correct": pred == rec["ground_truth"],
            })
        frames.append(pl.DataFrame(rows))
    return frames


def fft_frames(gt_records: list[dict], features: list[Path], threshold: float):
    """``aniso_mean >= threshold -> Generated``, one pseudo-model per condition.

    The FFT feature tables are keyed by ``clip_id``; only the clips present in
    both the study and the feature table are scored, and the row count is
    reported so a partial coverage is obvious.
    """
    import polars as pl

    gt = pl.DataFrame([
        {k: rec[k] for k in ("rel_video", "clip_id", "scenario_id", "source", "ground_truth")}
        | {"category": rec.get("category"), "in_pilot": bool(rec.get("in_pilot", True))}
        for rec in gt_records
    ])
    frames = []
    for path in features:
        feat = pl.read_parquet(path)
        if FFT_FEATURE not in feat.columns:
            continue
        if "error" in feat.columns:
            feat = feat.filter(pl.col("error").is_null())
        conditions = feat["condition"].unique().to_list() if "condition" in feat.columns else [None]
        for cond in conditions:
            f = feat if cond is None else feat.filter(pl.col("condition") == cond)
            f = f.select("clip_id", FFT_FEATURE).unique(subset="clip_id")
            j = gt.join(f, on="clip_id", how="inner")
            if j.height == 0:
                continue
            name = f"fft-threshold ({cond})" if cond else "fft-threshold"
            j = j.with_columns(
                pl.lit(name).alias("model"),
                pl.when(pl.col(FFT_FEATURE) >= threshold)
                .then(pl.lit("Generated")).otherwise(pl.lit("Real")).alias("pred"),
                pl.lit(False).alias("unparsed"),
            ).with_columns((pl.col("pred") == pl.col("ground_truth")).alias("correct"))
            frames.append(j.drop(FFT_FEATURE))
    return frames


def _load_stats():
    """``src/analysis/stats.py`` -- the shared cluster-bootstrap machinery."""
    analysis_dir = str(REPO / "src" / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    import stats

    return stats


def metrics_for(df, n_boot: int, seed: int, conf: float = 0.95) -> dict:
    """Accuracy / balanced accuracy / per-class recall / precision / F1 / MCC,
    each with a percentile **cluster** bootstrap CI over ``scenario_id``.

    Every metric is recomputed inside each bootstrap replicate from the same
    resampled world (one shared ``ClusterWeights`` draw from
    ``src/analysis/stats.make_cluster_weights``), so the reported balanced
    accuracy CI is coherent with the two recall CIs.

    Unparsed answers: like ``stats.class_metrics``, an unparseable verdict is a
    third predicted label.  It always counts as *wrong* (it lowers the recall
    of whichever class the clip belongs to) but it never counts as a
    ``Generated`` **prediction**, so it stays out of the precision denominator.
    MCC is the only exception -- it needs a genuine 2x2 table, so there an
    unparsed answer is folded into the opposite class.
    """
    import numpy as np

    make_cluster_weights = _load_stats().make_cluster_weights

    clusters = df["scenario_id"].to_numpy()
    w = make_cluster_weights(clusters, n_boot=n_boot, seed=seed)
    idx = np.array([w.index[c] for c in clusters])
    C = w.n_clusters

    gt = df["ground_truth"].to_numpy()
    pred = df["pred"].to_numpy()
    is_gen = gt == "Generated"
    is_real = gt == "Real"
    pred_gen = pred == "Generated"
    pred_real = pred == "Real"

    def collapse(mask: np.ndarray) -> np.ndarray:
        return np.bincount(idx, weights=mask.astype(float), minlength=C)

    # per-cluster counts of the confusion-matrix cells
    cells = {
        "tp": collapse(is_gen & pred_gen),
        "fn": collapse(is_gen & ~pred_gen),          # incl. UNPARSED on a generated clip
        "tn": collapse(is_real & pred_real),
        "fp": collapse(is_real & pred_gen),          # strictly *predicted* Generated
        "fp_mcc": collapse(is_real & ~pred_real),    # + UNPARSED on a real clip
        "n_gen": collapse(is_gen),
        "n_real": collapse(is_real),
        "n_all": collapse(np.ones_like(is_gen, dtype=bool)),
        "n_corr": collapse(df["correct"].to_numpy().astype(bool)),
        "n_unp": collapse(df["unparsed"].to_numpy().astype(bool)),
    }

    def stats_from(c: dict) -> dict:
        TP, FN, TN, FP = c["tp"], c["fn"], c["tn"], c["fp"]
        FPM, NG, NR, NA, NC, NU = (c["fp_mcc"], c["n_gen"], c["n_real"],
                                   c["n_all"], c["n_corr"], c["n_unp"])
        with np.errstate(invalid="ignore", divide="ignore"):
            rec_gen = np.where(NG > 0, TP / NG, np.nan)
            rec_real = np.where(NR > 0, TN / NR, np.nan)
            acc = np.where(NA > 0, NC / NA, np.nan)
            bal = (rec_gen + rec_real) / 2.0
            pp = TP + FP  # number of clips actually predicted "Generated"
            prec = np.where(pp > 0, TP / pp, np.nan)
            f1 = np.where(np.nan_to_num(prec) + rec_gen > 0,
                          2 * prec * rec_gen / (prec + rec_gen), 0.0)
            den = np.sqrt((TP + FPM) * (TP + FN) * (TN + FPM) * (TN + FN))
            mcc = np.where(den > 0, (TP * TN - FPM * FN) / den, 0.0)
            unp = np.where(NA > 0, NU / NA, np.nan)
        return {"accuracy": acc, "balanced_accuracy": bal, "recall_generated": rec_gen,
                "recall_real": rec_real, "precision_generated": prec, "f1_generated": f1,
                "mcc": mcc, "unparsed_rate": unp}

    point = stats_from({k: v.sum() for k, v in cells.items()})
    reps = stats_from({k: w.W @ v for k, v in cells.items()})

    a = (1.0 - conf) / 2.0 * 100.0
    out: dict = {
        "n": int(df.height),
        "n_clusters": int(C),
        "n_real": int(cells["n_real"].sum()),
        "n_generated": int(cells["n_gen"].sum()),
        "n_unparsed": int(cells["n_unp"].sum()),
        "n_pred_generated": int(pred_gen.sum()),
    }
    import warnings

    for k, v in point.items():
        r = np.asarray(reps[k], dtype=float)
        # a model that never predicts "Generated" has an undefined precision in
        # every replicate -> an all-NaN slice, which is the right answer, not a
        # problem worth a RuntimeWarning.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = float(np.nanpercentile(r, a))
            hi = float(np.nanpercentile(r, 100.0 - a))
        out[k] = float(v)
        out[f"{k}_lo"] = lo
        out[f"{k}_hi"] = hi
    return out


# --------------------------------------------------------------------------
# v2: outputs
# --------------------------------------------------------------------------


def latex_table(rows: list[dict]) -> str:
    lines = [
        "% generated by src/real_vs_generated/analyze_realgen.py --mode v2",
        "% do not edit by hand",
        "\\begin{tabular}{lrccccc}",
        "\\toprule",
        "System & $n$ & Acc. & Bal. acc. [95\\% CI] & Rec. Real & Rec. Gen. & MCC \\\\",
        "\\midrule",
    ]
    models = [r for r in rows if not r["is_baseline"]]
    bases = [r for r in rows if r["is_baseline"]]
    for group in (models, bases):
        if group is bases and models:
            lines.append("\\midrule")
        for r in sorted(group, key=lambda x: -x["balanced_accuracy"]):
            name = r["model"].replace("&", "\\&").replace("_", "\\_")
            lines.append(
                f"{name} & {r['n']} & {r['accuracy']:.3f} & "
                f"{r['balanced_accuracy']:.3f} [{r['balanced_accuracy_lo']:.3f}, "
                f"{r['balanced_accuracy_hi']:.3f}] & "
                f"{r['recall_real']:.3f} & {r['recall_generated']:.3f} & {r['mcc']:.3f} \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% Balanced real/generated set; CIs are percentile cluster bootstraps",
        "% resampling scenario clusters (real clips are singleton clusters).",
        "% Positive class = Generated.",
    ]
    return "\n".join(lines) + "\n"


def make_figure(rows: list[dict], out_dir: Path, suffix: str,
                n_real: int, n_gen: int) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": "#e2e1de", "axes.labelcolor": "#0b0b0b",
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#e2e1de", "grid.linewidth": 0.6, "grid.linestyle": "-",
        "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    C_BASELINE, C_AGENT, C_CLOSED = "#2a78d6", "#eb6834", "#1baf7a"
    C_REAL, C_GEN, C_FFT = "#0b7a55", "#8b5cf6", "#b4468f"
    C_INK, C_INK2 = "#0b0b0b", "#52514e"

    models = sorted([r for r in rows if not r["is_baseline"]],
                    key=lambda r: r["balanced_accuracy"])
    bases = {r["model"]: r for r in rows if r["is_baseline"]}
    if not models:
        return []

    def colour_of(name: str) -> str:
        if name in ("GPT-5.4-mini", "Gemini-3-Flash"):
            return C_CLOSED
        return C_AGENT if "agent" in name else C_BASELINE

    fig, ax = plt.subplots(figsize=(7.8, 0.42 * len(models) + 1.6))
    ys = np.arange(len(models))
    bal = np.array([r["balanced_accuracy"] for r in models])
    lo = np.array([r["balanced_accuracy_lo"] for r in models])
    hi = np.array([r["balanced_accuracy_hi"] for r in models])

    ax.barh(ys, bal, color=[colour_of(r["model"]) for r in models],
            height=0.6, zorder=2, alpha=0.9)
    ax.errorbar(bal, ys, xerr=[bal - lo, hi - bal], fmt="none",
                ecolor=C_INK, elinewidth=1.1, capsize=2.5, zorder=5)
    ax.scatter([r["recall_real"] for r in models], ys, marker="o", s=26,
               facecolor="white", edgecolor=C_REAL, linewidth=1.4, zorder=6,
               label="recall — Real clips")
    ax.scatter([r["recall_generated"] for r in models], ys, marker="D", s=24,
               facecolor="white", edgecolor=C_GEN, linewidth=1.4, zorder=6,
               label="recall — Generated clips")

    # --- reference lines -------------------------------------------------
    # the three trivial policies all sit at 0.5 balanced accuracy by
    # construction on a balanced set, so they get a single labelled line.
    trivial = [bases[k]["balanced_accuracy"] for k in BASELINE_LABELS if k in bases]
    if trivial:
        ax.axvline(0.5, color=C_INK2, linewidth=1.0, linestyle=(0, (5, 3)), zorder=3,
                   label="always-Real / always-Generated / random = 0.500")
    for name in sorted(k for k in bases if k.startswith("fft-threshold")):
        b = bases[name]
        ax.axvline(b["balanced_accuracy"], color=C_FFT, linewidth=1.1,
                   linestyle=(0, (2.5, 2)), zorder=3,
                   label=f"{name} = {b['balanced_accuracy']:.3f}")

    # value labels inside the bar so they never collide with the recall dots
    for y, r in zip(ys, models):
        ax.text(0.012, y, f"{r['balanced_accuracy']:.3f}", va="center", ha="left",
                fontsize=7.5, color="white", zorder=7, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in models])
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Balanced accuracy (Real / Generated), 95 % cluster-bootstrap CI")
    ax.set_title(f"Balanced real-vs-generated benchmark{' — pilot' if suffix else ''} "
                 f"(n = {n_real + n_gen}: {n_real} real / {n_gen} generated)", pad=10)
    ax.grid(True, axis="x", zorder=0)
    fig.tight_layout()
    # legend below the figure (figure fraction, so it does not depend on the
    # number of models); savefig(bbox_inches="tight") grows the canvas for it.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, handlelength=2.6, columnspacing=1.6)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"fig_realgen_balanced{suffix}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def run_v2(args) -> int:
    import polars as pl

    gt_map, gt_records = load_gt(args.gt, args.pilot_only)
    counts: dict[str, int] = defaultdict(int)
    for r in gt_records:
        counts[r["ground_truth"]] += 1
    scope = " (pilot only)" if args.pilot_only else ""
    print(f"Ground truth{scope}: {len(gt_records)} clips, "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f", {len({r['scenario_id'] for r in gt_records})} scenario clusters")
    print()

    frames = []
    result_files = find_results(args.results_dir)
    for path in result_files:
        label = model_label(path)
        try:
            df, skipped = items_frame(path, gt_map, label)
        except Exception as exc:
            print(f"  {label:24s} ERROR: {exc}")
            continue
        if args.pilot_only:
            df = df.filter(pl.col("in_pilot"))
        if skipped:
            why = "outside the pilot subset" if args.pilot_only else "with no ground-truth match"
            print(f"  [{label}] skipped {skipped} result entries {why}")
        frames.append(df.with_columns(pl.lit(False).alias("is_baseline")))
    if not result_files:
        print(f"No *_realgen*.json files in {args.results_dir} "
              "-- reporting the trivial baselines only.")

    for df in baseline_frames(gt_records, seed=args.seed):
        frames.append(df.with_columns(pl.lit(True).alias("is_baseline")))

    feature_files = (
        [Path(p) for p in args.fft_features] if args.fft_features
        else sorted(FFT_DIR.glob("features_*.parquet"))
    )
    feature_files = [p for p in feature_files if p.exists()]
    if feature_files:
        fft = fft_frames(gt_records, feature_files, args.fft_threshold)
        for df in fft:
            print(f"  [{df['model'][0]}] scored {df.height} / {len(gt_records)} clips "
                  f"at {FFT_FEATURE} >= {args.fft_threshold}")
            frames.append(df.with_columns(pl.lit(True).alias("is_baseline")))
        if not fft:
            print(f"  [fft] no clip of this study is present in {len(feature_files)} "
                  "feature file(s) yet")
    else:
        print("  [fft] no results/fft/features_*.parquet yet -- skipping the FFT baseline")

    rows = []
    for df in frames:
        label = df["model"][0]
        m = metrics_for(df, n_boot=args.n_boot, seed=args.seed)
        m["model"] = label
        m["is_baseline"] = bool(df["is_baseline"][0])
        rows.append(m)

    per_source = []
    for df in frames:
        if bool(df["is_baseline"][0]):
            continue
        for src, part in df.partition_by("source", as_dict=True).items():
            per_source.append({
                "model": df["model"][0],
                "source": src[0] if isinstance(src, tuple) else src,
                "n": part.height,
                "accuracy": float(part["correct"].mean()),
                "unparsed_rate": float(part["unparsed"].mean()),
            })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_pilot" if args.pilot_only else ""

    metrics = pl.DataFrame(rows).sort(["is_baseline", "balanced_accuracy"], descending=[False, True])
    metrics.write_parquet(out_dir / f"realgen_metrics_v2{suffix}.parquet")

    hdr = (f"{'System':24s} {'n':>5s} {'acc':>7s} {'bal.acc [95% CI]':>24s} "
           f"{'rec Real':>9s} {'rec Gen':>9s} {'MCC':>7s} {'unpars':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in metrics.iter_rows(named=True):
        print(f"{r['model']:24s} {r['n']:>5d} {r['accuracy']*100:>6.1f}% "
              f"{r['balanced_accuracy']:.3f} [{r['balanced_accuracy_lo']:.3f},"
              f"{r['balanced_accuracy_hi']:.3f}] "
              f"{r['recall_real']*100:>8.1f}% {r['recall_generated']*100:>8.1f}% "
              f"{r['mcc']:>7.3f} {r['unparsed_rate']*100:>6.1f}%")

    summary = {
        "gt": str(args.gt),
        "results_dir": str(args.results_dir),
        "pilot_only": bool(args.pilot_only),
        "n_clips": len(gt_records),
        "n_real": counts.get("Real", 0),
        "n_generated": counts.get("Generated", 0),
        "n_clusters": len({r["scenario_id"] for r in gt_records}),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "fft_threshold": args.fft_threshold,
        "fft_features": [str(p) for p in feature_files],
        "models": {r["model"]: r for r in rows},
        "per_source": per_source,
    }
    (out_dir / f"realgen_summary_v2{suffix}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"realgen_v2{suffix}.tex").write_text(latex_table(rows))

    figs = make_figure(rows, Path(args.fig_dir), suffix,
                       counts.get("Real", 0), counts.get("Generated", 0))

    print(f"\nWrote {out_dir / f'realgen_metrics_v2{suffix}.parquet'}")
    print(f"Wrote {out_dir / f'realgen_summary_v2{suffix}.json'}")
    print(f"Wrote {out_dir / f'realgen_v2{suffix}.tex'}")
    for p in figs:
        print(f"Wrote {p}")
    return 0


# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("v1", "v2"), default="v1")
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_V2_OUT, help="v2 output directory")
    p.add_argument("--summary-out", type=Path, default=None,
                   help="v1: where to write realgen_accuracy_summary.json")
    p.add_argument("--pilot-only", action="store_true",
                   help="restrict to the clips flagged in_pilot in the ground truth")
    p.add_argument("--fft-features", nargs="*", default=None,
                   help="FFT feature parquets (default: results/fft/features_*.parquet)")
    p.add_argument("--fft-threshold", type=float, default=DEFAULT_FFT_THRESHOLD,
                   help=f"{FFT_FEATURE} >= threshold -> Generated (default: the "
                        f"{DEFAULT_FFT_THRESHOLD} operating point from the 18+18 calibration)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.gt).exists():
        print(f"ERROR: ground-truth file missing at {args.gt}", file=sys.stderr)
        return 1
    return run_v1(args) if args.mode == "v1" else run_v2(args)


if __name__ == "__main__":
    sys.exit(main())
