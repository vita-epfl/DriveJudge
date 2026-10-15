#!/usr/bin/env python3
"""Convert the filled adjudication CSV into reliability_adjudicated.jsonl consumed by agreement.py
(--adjudicated), and print the reason breakdown + label-error rate with Wilson CIs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "annotation"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RESULTS / "adjudication_reliability.csv")
    ap.add_argument("--agreement", type=Path, default=RESULTS / "agreement_reliability.json",
                    help="agreement output; gives n items scored (denominator of the label-error rate)")
    ap.add_argument("--out", type=Path, default=RESULTS / "reliability_adjudicated.jsonl")
    ap.add_argument("--summary", type=Path, default=RESULTS / "adjudication_summary.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    reason_col = next(c for c in rows[0] if c.startswith("reason"))
    missing = [r for r in rows if not r["final"].strip() or not r[reason_col].strip()]
    if missing:
        raise SystemExit(f"{len(missing)} rows still unfilled")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            source, video = r["clip"].split("/", 1)
            f.write(json.dumps({
                "source": source, "video": video, "question_prefix": r["question"],
                "answer": r["final"], "original_label": r["original_label"], "annotator2": r["annotator2"],
                "reason": r[reason_col], "note": r["note"],
            }, ensure_ascii=False) + "\n")

    n_scored = json.load(open(args.agreement))["overall"]["n"]
    realism = [r for r in rows if r["question"].lower().startswith("overall, how realistic")]
    rest = [r for r in rows if r not in realism]
    reasons_all = Counter(r[reason_col] for r in rows)
    reasons_rest = Counter(r[reason_col] for r in rest)
    gt_wrong = [r for r in rows if r["final"] != r["original_label"]]
    a2_wrong = [r for r in rows if r["final"] != r["annotator2"]]
    inconsistent = [r for r in rows if r["final"] != r["original_label"] and r[reason_col] != "label_error"]

    summary = {
        "n_items_scored": n_scored,
        "n_disagreements": len(rows),
        "n_realism_bulk_convention": len(realism),
        "reasons_all": dict(reasons_all),
        "reasons_non_realism": dict(reasons_rest),
        "gt_label_errors": len(gt_wrong),
        "gt_label_error_rate": {"point": len(gt_wrong) / n_scored, "wilson95": wilson(len(gt_wrong), n_scored)},
        "gt_label_error_rate_non_realism": {
            "point": len(gt_wrong) / (n_scored - sum(1 for _ in [])),  # same denominator, kept for clarity
        },
        "annotator2_errors_vs_final": len(a2_wrong),
        "annotator2_error_rate": {"point": len(a2_wrong) / n_scored, "wilson95": wilson(len(a2_wrong), n_scored)},
        "ambiguous_clips": [f"{r['clip']} | {r['question'][:50]}" for r in rows if r[reason_col] == "ambiguous_clip"],
        "inconsistent_rows_final_ne_gt_but_reason_ne_label_error": [
            f"{r['clip']} | {r['question'][:50]} | GT {r['original_label']} A2 {r['annotator2']} final {r['final']} reason {r[reason_col]}"
            for r in inconsistent],
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}\nwrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
