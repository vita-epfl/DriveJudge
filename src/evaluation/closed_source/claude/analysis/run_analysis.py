"""Analysis pipeline for Claude batch results.

Twin of `src/evaluation/closed_source/gemini/analysis/run_analysis.py`. The
Claude result JSON already matches the canonical agent shape (video, question,
evaluation, answer, history, usage, model, custom_id), so the cleaning + GT-join
logic is identical — only the paths, the output prefix and the CLI differ.

    python analysis/run_analysis.py                                   # defaults
    python analysis/run_analysis.py --results results/claude_all_results.json \\
                                    --out-name claude-opus-5
    python analysis/run_analysis.py --model claude-opus-5             # infers both

Writes, next to this file:
    <out-name>_results_enriched.parquet / .csv
    <out-name>_accuracy_summary.json
    failures_by_category/<slug>_failures.parquet / .json
and, at the repo root:
    results/Claude-<model>_analyzed.parquet     (10 columns, identical schema to
                                                 results/Gemini_analyzed.parquet)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[4]
QUESTIONS_DD = ROOT / "dataset/dataset_final/questions_cosmos-drive-dreams.json"
QUESTIONS_P1 = ROOT / "dataset/dataset_final/questions_cosmos-predict1.json"
CLUSTER_PREFIX = "/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/"

DEFAULT_MODEL = "claude-sonnet-5"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitise(name: str) -> str:
    return _UNSAFE.sub("_", (name or "").strip()).strip("._-") or "model"


def display_name(model: str) -> str:
    """`claude-sonnet-5` -> `Claude-sonnet-5` (the repo results/ naming style)."""
    slug = sanitise(model)
    for prefix in ("claude-", "claude_", "claude"):
        if slug.lower().startswith(prefix):
            slug = slug[len(prefix):].lstrip("-_")
            break
    return f"Claude-{slug}" if slug else "Claude"


def clean_records(raw):
    suffix_pattern = r"\s*The answer is one of the following\s*:.*"
    cleaned = []
    for el in raw:
        question = re.sub(suffix_pattern, "", el["question"], flags=re.IGNORECASE).strip()
        video_path = re.sub(re.escape(CLUSTER_PREFIX), "", el["video"])
        cleaned.append({
            "video": video_path,
            "history": el.get("history", []),
            "question": question,
            "raw_output": el["evaluation"],
            "vlm_answer": el["answer"],
            "usage": el.get("usage"),
            "custom_id": el.get("custom_id"),
        })
    return cleaned


def build_lookup(entries):
    lookup = {}
    for entry in entries:
        v = entry["video"]
        for qa in entry.get("qa_pairs", []):
            lookup[(v, qa["question"])] = {
                "answer": qa["answer"],
                "category": qa.get("category"),
                "type": qa.get("type"),
                "possible_answers": qa.get("possible_answers"),
            }
    return lookup


def enrich(cleaned, lookup_dd, lookup_p1):
    enriched = []
    missing = 0
    for item in cleaned:
        path = item["video"]
        if path.startswith("cosmos-drive-dreams/"):
            source = "cosmos-drive-dreams"
            base = path[len("cosmos-drive-dreams/") :]
            lk = lookup_dd
        elif path.startswith("cosmos-predict1/"):
            source = "cosmos-predict1"
            base = path[len("cosmos-predict1/") :]
            lk = lookup_p1
        else:
            raise ValueError(f"Unknown video prefix: {path}")

        info = lk.get((base, item["question"]))
        if info is None:
            missing += 1
            continue

        out = item.copy()
        out["source"] = source
        out["video"] = base
        out["ground_truth"] = info["answer"]
        out["category"] = info["category"]
        out["question_type"] = info["type"]
        enriched.append(out)
    return enriched, missing


def to_df(enriched):
    """The canonical 10-column frame — same schema as results/Gemini_analyzed.parquet."""
    rows = [
        {
            "source": r["source"],
            "video": r["video"],
            "question": r["question"],
            "category": r["category"],
            "question_type": r["question_type"],
            "ground_truth": str(r["ground_truth"]),
            "vlm_answer": str(r["vlm_answer"]),
            "raw_output": r["raw_output"],
            "custom_id": r.get("custom_id"),
        }
        for r in enriched
    ]
    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("vlm_answer") == pl.col("ground_truth")).alias("correct")
    )
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Model id, used to derive the default output names.")
    ap.add_argument("--results", type=Path, default=None,
                    help="Results JSON (default ../results/<stem>_batch_results.json).")
    ap.add_argument("--out-name", default=None,
                    help="Prefix for the local artifacts (default `claude` for "
                         "the default model, the sanitised model id otherwise).")
    ap.add_argument("--analyzed-out", type=Path, default=None,
                    help="Repo-level analyzed parquet "
                         "(default results/Claude-<model>_analyzed.parquet).")
    ap.add_argument("--out-dir", type=Path, default=_HERE)
    args = ap.parse_args(argv)

    stem = "claude" if args.model == DEFAULT_MODEL else sanitise(args.model)
    if args.out_name is None:
        args.out_name = stem
    if args.results is None:
        args.results = _HERE.parent / "results" / f"{stem}_batch_results.json"
    if args.analyzed_out is None:
        args.analyzed_out = ROOT / "results" / f"{display_name(args.model)}_analyzed.parquet"
    return args


def main(argv=None):
    args = parse_args(argv)

    with open(args.results) as f:
        raw = json.load(f)
    with open(QUESTIONS_DD) as f:
        qdd = json.load(f)
    with open(QUESTIONS_P1) as f:
        qp1 = json.load(f)

    cleaned = clean_records(raw)
    enriched, missing = enrich(cleaned, build_lookup(qdd), build_lookup(qp1))
    df = to_df(enriched)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    df.write_parquet(f"{out_dir}/{args.out_name}_results_enriched.parquet")
    df.write_csv(f"{out_dir}/{args.out_name}_results_enriched.csv")

    args.analyzed_out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.analyzed_out)
    print(f"Wrote {args.analyzed_out} ({df.shape[0]} rows x {df.shape[1]} cols).")

    overall_acc = df["correct"].mean()
    by_source = table(df, ["source"])
    by_cat = table(df, ["category"])
    by_src_cat = table(df, ["source", "category"])
    by_q = table(df, ["question"])

    cat_dir = f"{out_dir}/failures_by_category"
    os.makedirs(cat_dir, exist_ok=True)
    per_cat_counts = {}
    for cat in sorted(df["category"].unique().to_list()):
        cat_df = df.filter(pl.col("category") == cat)
        wrong = cat_df.filter(~pl.col("correct"))
        per_cat_counts[cat] = {
            "n": len(cat_df),
            "n_wrong": len(wrong),
            "accuracy": float(cat_df["correct"].mean()),
        }
        slug = re.sub(r"\W+", "_", cat).strip("_").lower()
        wrong.write_parquet(f"{cat_dir}/{slug}_failures.parquet")
        with open(f"{cat_dir}/{slug}_failures.json", "w") as fp:
            json.dump(wrong.to_dicts(), fp, indent=2, ensure_ascii=False)

    summary = {
        "model": args.model,
        "results_path": str(args.results),
        "n_records_raw": len(raw),
        "n_records_cleaned": len(cleaned),
        "n_missing_gt": missing,
        "n_records_enriched": len(enriched),
        "overall_accuracy": float(overall_acc),
        "per_source": by_source.to_dicts(),
        "per_category": by_cat.to_dicts(),
        "per_source_category": by_src_cat.to_dicts(),
        "per_question": by_q.to_dicts(),
        "per_category_failure_counts": per_cat_counts,
    }
    with open(f"{out_dir}/{args.out_name}_accuracy_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Raw records:      {len(raw)}")
    print(f"Cleaned records:  {len(cleaned)}")
    print(f"Missing in GT:    {missing}")
    print(f"Enriched records: {len(enriched)}")
    print(f"\nOVERALL ACCURACY: {overall_acc:.4%}\n")

    print("== Per source ==")
    print(by_source)
    print("\n== Per category ==")
    print(by_cat)
    print("\n== Per source x category ==")
    print(by_src_cat)
    print("\n== Per question ==")
    with pl.Config(fmt_str_lengths=200, tbl_rows=30):
        print(by_q)

    print("\n== vlm_answer x ground_truth (top combos) ==")
    combo = (
        df.group_by(["ground_truth", "vlm_answer"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    with pl.Config(tbl_rows=30):
        print(combo)


if __name__ == "__main__":
    sys.exit(main())
