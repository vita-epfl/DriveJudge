"""Recompute open-source accuracy on the (Gemini ∩ GPT) question set.

Closed-source runs (Gemini 3 Flash, GPT-5.4-mini) covered slightly fewer
(video, question) pairs than the 7 371 used for the open-source eval. To make
the numbers comparable, we filter the eight in-house enriched parquets down to
the intersection of Gemini and GPT custom_ids before recomputing accuracy.

The original enriched parquets are NOT modified — filtered DataFrames live
only in memory.

Also prints the count of "real or generated" questions in the trimmed set.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report"

GEMINI_JSON = ROOT / "src/evaluation/closed_source/gemini/results/gemini_batch_results.json"
GPT_JSON = ROOT / "src/evaluation/closed_source/gpt/results/gpt_all_results.json"

ENRICHED = {
    "Cosmos-Reason":          "cosmos_reason_enriched.parquet",
    "InternVL3.5-30B":        "internvl3_5_30b_enriched.parquet",
    "InternVL3.5-8B":         "internvl3_5_8b_enriched.parquet",
    "LLaVA-OneVision":        "llava_onevision_enriched.parquet",
    "Qwen3-Omni":             "qwen3_omni_enriched.parquet",
    "Qwen3-VL":               "qwen3_vl_enriched.parquet",
    "Qwen3-VL+agent":         "qwen3_vl_agent_enriched.parquet",
    "Qwen3-VL+agent+hints":   "qwen3_vl_agent_hints_enriched.parquet",
}

# Closed-source question text has an extra suffix " The answer is one of the following : ... "
SUFFIX_RE = re.compile(r"\s*The answer is one of the following\s*:.*$", re.DOTALL)
PATH_RE = re.compile(r"final_dataset/(cosmos-[^/]+)/(.+)$")


def norm_q(q: str) -> str:
    return SUFFIX_RE.sub("", q or "").strip()


def closed_key(rec) -> tuple[str, str, str] | None:
    m = PATH_RE.search(rec["video"])
    if not m:
        return None
    return (m.group(1), m.group(2), norm_q(rec["question"]))


def build_intersection() -> set[tuple[str, str, str]]:
    gem = json.load(open(GEMINI_JSON))
    gpt = json.load(open(GPT_JSON))
    g_keys = {k for r in gem if (k := closed_key(r)) is not None}
    p_keys = {k for r in gpt if (k := closed_key(r)) is not None}
    print(f"Gemini coverage : {len(g_keys):>5d} unique (video, question) pairs")
    print(f"GPT coverage    : {len(p_keys):>5d} unique (video, question) pairs")
    inter = g_keys & p_keys
    print(f"Intersection    : {len(inter):>5d} pairs (= eval set)")
    print(f"Gemini-only     : {len(g_keys - p_keys):>5d}")
    print(f"GPT-only        : {len(p_keys - g_keys):>5d}")
    return inter


def filter_one(parquet_name: str, eval_keys: set) -> pl.DataFrame:
    df = pl.read_parquet(f"{OUT_DIR}/{parquet_name}")
    df = df.with_columns(
        pl.col("question").map_elements(norm_q, return_dtype=pl.Utf8).alias("q_norm")
    )
    df = df.with_columns(
        pl.struct(["source", "video", "q_norm"])
          .map_elements(lambda s: (s["source"], s["video"], s["q_norm"]) in eval_keys,
                        return_dtype=pl.Boolean)
          .alias("in_eval_set")
    )
    return df.filter(pl.col("in_eval_set")).drop(["q_norm", "in_eval_set"])


def per_category(df: pl.DataFrame) -> list[dict]:
    return (
        df.group_by("category")
          .agg(pl.len().alias("n"), pl.col("correct").mean().alias("accuracy"))
          .sort("category")
    ).to_dicts()


def main():
    eval_keys = build_intersection()

    # Count real-or-generated questions in the trimmed set (use a reference model)
    ref = filter_one(ENRICHED["Qwen3-VL"], eval_keys)
    realgen_mask = ref["question"].str.contains("real video or generated|real or generated")
    n_realgen = int(realgen_mask.sum())
    print(f"\n'Is it a real video or generated ?' items in trimmed set: {n_realgen}")
    # Sanity: also show count in untrimmed open-source
    raw = pl.read_parquet(f"{OUT_DIR}/{ENRICHED['Qwen3-VL']}")
    n_realgen_raw = int(raw["question"].str.contains("real video or generated|real or generated").sum())
    print(f"'Is it a real video or generated ?' items in untrimmed open-source: {n_realgen_raw}")

    print("\nTrimmed accuracy per model")
    print("-" * 88)
    print(f"{'model':24s} {'n_kept':>7s} {'n_dropped':>10s} {'acc_trim':>9s} {'acc_full':>9s}")
    print("-" * 88)
    summary = {"eval_set_size": len(eval_keys), "models": {}}
    for mid, fname in ENRICHED.items():
        full = pl.read_parquet(f"{OUT_DIR}/{fname}")
        sub = filter_one(fname, eval_keys)
        acc_full = float(full["correct"].mean())
        acc_trim = float(sub["correct"].mean())
        n_kept = len(sub)
        n_drop = len(full) - n_kept
        print(f"{mid:24s} {n_kept:>7d} {n_drop:>10d} {acc_trim*100:>8.2f}% {acc_full*100:>8.2f}%")
        summary["models"][mid] = {
            "n_kept": n_kept,
            "n_dropped": n_drop,
            "accuracy_trimmed": acc_trim,
            "accuracy_full": acc_full,
            "per_category_trimmed": per_category(sub),
        }

    out = f"{OUT_DIR}/trimmed_accuracy_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
