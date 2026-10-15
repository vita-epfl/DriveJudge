"""Cross-model accuracy + per-category failure dumps for the 10 systems in
`results/`.

Re-parses raw_output for the six open-source baselines (their `vlm_answer` is
the raw text, not the extracted final answer) and writes the re-extracted
`vlm_answer` / `correct` back into `results/<model>_analyzed.parquet`, so the
tracked parquets carry a valid `correct` column. `raw_output` is preserved
untouched, which makes the rewrite idempotent. The agentic v4o/v4p variants and
the two closed-source models (GPT-5.4-mini, Gemini-3-Flash) already store a
clean `vlm_answer` and are left alone on disk.

Writes one enriched parquet per model under `report/`, a single all-models
accuracy summary JSON, and one failure JSON per (model, category) pair under
`report/failures_by_model_category/`.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
OUT_DIR = ROOT / "report"
FAIL_DIR = OUT_DIR / "failures_by_model_category"

# Files to process: parquet name -> short model id used in summaries
MODELS = {
    "Cosmos_Reason_analyzed.parquet":    "Cosmos-Reason",
    "InternVL3_5-30B_analyzed.parquet":  "InternVL3.5-30B",
    "InternVL3_5-8B_analyzed.parquet":   "InternVL3.5-8B",
    "Qwen3VL_analyzed.parquet":          "Qwen3-VL",
    "Qwen_omni_analyzed.parquet":        "Qwen3-Omni",
    "llava_onevision_analyzed.parquet":  "LLaVA-OneVision",
    "v4o_analyzed.parquet":              "Qwen3-VL+agent",
    "v4p_analyzed.parquet":              "Qwen3-VL+agent+hints",
    "GPT-5.4-mini_analyzed.parquet":     "GPT-5.4-mini",
    "Gemini_analyzed.parquet":           "Gemini-3-Flash",
}
# Models whose vlm_answer column already holds the extracted answer
AGENTIC = {
    "v4o_analyzed.parquet",
    "v4p_analyzed.parquet",
    "GPT-5.4-mini_analyzed.parquet",
    "Gemini_analyzed.parquet",
}

# Closed-source parquets prefix `video` with the dataset name; the in-house ones
# don't. Strip it so all 10 models share one (source, video) key convention.
SOURCE_PREFIX_RE = r"^(?:cosmos-drive-dreams|cosmos-predict1)/"


# ---------------- answer extraction ----------------
ANS_RE = re.compile(r"Answer\s*:\s*([^\n<]+)", re.IGNORECASE)


def extract_answer(raw: str, question: str) -> str:
    """Pull the final 'Answer: X' from a raw 'Feedback:::\n...\nAnswer: X' string.

    Falls back to lightweight question-aware heuristics (Yes/No, 1-3, Real/Generated,
    Left/Right) on the last 300 chars when no explicit Answer: marker exists.
    """
    if raw is None:
        return "NO_PARSE"
    # Take the LAST occurrence of "Answer:" since some models repeat it inside reasoning
    matches = list(ANS_RE.finditer(raw))
    if matches:
        ans = matches[-1].group(1).strip()
        ans = re.sub(r"<\|im_end\|>.*$", "", ans).strip()
        # Strip trailing punctuation/whitespace
        ans = ans.strip(" .\t\r\n*\"'")
        # Capitalise to match GT casing (Yes/No/Real/Generated/Left/Right/1/2/3)
        if ans:
            return ans[:1].upper() + ans[1:]
    # ------- heuristics -------
    tail = (raw or "")[-400:].lower()
    q = (question or "").lower()
    if "score that ranges from 1 to 3" in q or re.search(r"\b1\s*2\s*3\b", q):
        digits = re.findall(r"\b([123])\b", tail)
        if digits:
            return digits[-1]
    if "real video or generated" in q or "real or generated" in q:
        if tail.count("generated") > tail.count("real"):
            return "Generated"
        if tail.count("real") > tail.count("generated"):
            return "Real"
    if "which side" in q and "overtak" in q:
        if "not overtaking" in tail:
            return "It's not overtaking"
        if tail.rfind("left") > tail.rfind("right"):
            return "Left"
        if tail.rfind("right") > tail.rfind("left"):
            return "Right"
    if any(k in q for k in ["is it a safe", "is the ego car", "has the ego car", "are the traffic", "do some objects", "does any object", "is the lighting"]):
        y = len(re.findall(r"\byes\b", tail))
        n = len(re.findall(r"\bno\b", tail))
        if y > n:
            return "Yes"
        if n > y:
            return "No"
    return "NO_PARSE"


# ---------------- pipeline ----------------

def load_one(file_name: str, model_id: str) -> pl.DataFrame:
    path = RES_DIR / file_name
    df = pl.read_parquet(path)
    # Normalise `video`: the closed-source parquets carry the dataset name as a
    # leading path segment ("cosmos-predict1/artifacts/x.mp4"), the in-house
    # ones do not ("artifacts/x.mp4"). Strip it so every model uses one key.
    df = df.with_columns(
        pl.col("video").str.replace(SOURCE_PREFIX_RE, "", literal=False).alias("video")
    )
    # Add source column if missing
    # Source: videos under the `artifacts/` folder come from cosmos-predict1;
    # all other folders (negative/, aggressive_takeover/, crossing_red_lights/,
    # crossing_stop/) come from cosmos-drive-dreams.
    df = df.with_columns(
        pl.when(pl.col("video").str.starts_with("artifacts/"))
          .then(pl.lit("cosmos-predict1"))
          .otherwise(pl.lit("cosmos-drive-dreams"))
          .alias("source")
    )
    if file_name in AGENTIC:
        # Already-extracted answers; keep them, just normalise capitalisation
        df = df.with_columns(
            pl.col("vlm_answer").cast(pl.Utf8).alias("vlm_answer")
        )
    else:
        # Re-extract from raw_output
        rows = df.to_dicts()
        for r in rows:
            r["vlm_answer"] = extract_answer(r["raw_output"], r["question"])
        df = pl.DataFrame(rows)
    df = df.with_columns([
        pl.col("vlm_answer").cast(pl.Utf8),
        pl.col("ground_truth").cast(pl.Utf8),
        pl.col("category").cast(pl.Utf8),
    ])
    df = df.with_columns(
        (pl.col("vlm_answer") == pl.col("ground_truth")).alias("correct")
    )
    df = df.with_columns(pl.lit(model_id).alias("model_id"))
    return df


def table(df: pl.DataFrame, group: list[str]) -> pl.DataFrame:
    return (
        df.group_by(group)
        .agg(
            pl.len().alias("n"),
            pl.col("correct").sum().alias("n_correct"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort(group)
    )


def slug(s: str) -> str:
    return re.sub(r"\W+", "_", s).strip("_").lower()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)

    summary = {"models": {}, "models_overall": []}
    for fname, mid in MODELS.items():
        df = load_one(fname, mid)
        if fname not in AGENTIC:
            # Persist the re-extracted vlm_answer / correct back into the
            # tracked results parquet. raw_output is untouched, so re-running
            # simply re-derives the same columns (idempotent).
            df.write_parquet(RES_DIR / fname)
            print(f"  rewrote {RES_DIR / fname} (vlm_answer + correct re-extracted)")
        df.write_parquet(f"{OUT_DIR}/{slug(mid)}_enriched.parquet")
        overall = float(df["correct"].mean())
        n = len(df)
        n_correct = int(df["correct"].sum())
        per_source = table(df, ["source"]).to_dicts()
        per_cat = table(df, ["category"]).to_dicts()
        per_src_cat = table(df, ["source", "category"]).to_dicts()
        per_q = table(df, ["question"]).to_dicts()

        # confusion table
        confusion = (
            df.group_by(["category", "ground_truth", "vlm_answer"]).agg(pl.len().alias("n"))
            .sort(["category", "n"], descending=[False, True])
        ).to_dicts()
        confusion_overall = (
            df.group_by(["ground_truth", "vlm_answer"]).agg(pl.len().alias("n"))
            .sort("n", descending=True)
        ).to_dicts()

        summary["models"][mid] = {
            "file": fname,
            "n": n,
            "n_correct": n_correct,
            "overall_accuracy": overall,
            "per_source": per_source,
            "per_category": per_cat,
            "per_source_category": per_src_cat,
            "per_question": per_q,
            "confusion_per_category": confusion,
            "confusion_overall": confusion_overall,
        }
        summary["models_overall"].append({"model": mid, "n": n, "accuracy": overall})

        # ---- per-category failure dumps for subagent reviewers ----
        wrong = df.filter(~pl.col("correct"))
        for cat in sorted(df["category"].unique().to_list()):
            cw = wrong.filter(pl.col("category") == cat)
            recs = cw.select([
                "source", "video", "question", "category",
                "ground_truth", "vlm_answer", "raw_output",
            ]).to_dicts()
            # Truncate very long raw_outputs for the JSON dump
            for r in recs:
                if r.get("raw_output") and len(r["raw_output"]) > 4000:
                    r["raw_output"] = r["raw_output"][:4000] + "...[TRUNCATED]"
            cat_slug = slug(cat)
            with open(f"{FAIL_DIR}/{slug(mid)}__{cat_slug}.json", "w") as f:
                json.dump(recs, f, indent=2, ensure_ascii=False)

        print(f"{mid:25s}  n={n}  acc={overall:.4f}  wrong={len(wrong)}")

    summary["models_overall"] = sorted(summary["models_overall"], key=lambda x: -x["accuracy"])

    with open(f"{OUT_DIR}/all_models_accuracy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nWrote:", f"{OUT_DIR}/all_models_accuracy_summary.json")
    print("Per-(model,category) failure dumps in:", FAIL_DIR)


if __name__ == "__main__":
    sys.exit(main())
