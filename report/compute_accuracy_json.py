"""Compute per-model accuracy + mean inference time and emit two JSON files.

- report/models-accuracy.json:  all 10 evaluated systems (8 in-house + GPT-5.4-mini + Gemini-3-Flash)
- report/accuracy-vs-time.json: the 8 in-house systems, which are the only ones
  with a measured `inference_time` (the closed-source runs go through batch APIs)

Everything is read straight from `results/*_analyzed.parquet`; the `correct`
column there is authoritative (regenerate it with `report/run_analysis.py`,
which re-extracts `vlm_answer` from `raw_output` and writes it back). No answer
re-parsing happens here.
"""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "report"

# In-house, timed models. file stem -> (display name, category)
# Display names follow the HF checkpoints actually evaluated, see
# src/evaluation/open_source/eval_*.py.
TIMED_MODELS = {
    "Qwen3VL_analyzed":          ("Qwen3-VL-30B",     "base"),   # Qwen3-VL-30B-A3B-Instruct
    "Qwen_omni_analyzed":        ("Qwen3-Omni-30B",   "base"),   # Qwen3-Omni-30B-A3B-Instruct
    "llava_onevision_analyzed":  ("LLaVA-OneVision",  "base"),
    "Cosmos_Reason_analyzed":    ("Cosmos-Reason",    "base"),
    "InternVL3_5-8B_analyzed":   ("InternVL3.5-8B",   "base"),
    "InternVL3_5-30B_analyzed":  ("InternVL3.5-30B",  "base"),
    "v4o_analyzed":              ("Agent v4o",        "agent"),
    "v4p_analyzed":              ("Agent v4p",        "agent"),
}

# Closed-source models: accuracy only, no comparable wall-clock time.
CLOSED_MODELS = {
    "GPT-5.4-mini_analyzed": "GPT-5.4-mini",
    "Gemini_analyzed":       "Gemini-3-Flash",
}


accuracy_entries = []
point_entries = []

for stem, (display_name, category) in TIMED_MODELS.items():
    df = pl.read_parquet(RESULTS_DIR / f"{stem}.parquet")

    accuracy = round(df["correct"].cast(pl.Int64).mean() * 100, 1)
    time_s = round(df["inference_time"].mean(), 2)

    accuracy_entries.append({"name": display_name, "accuracy": accuracy, "highlight": False})
    point_entries.append({
        "name": display_name,
        "time": time_s,
        "accuracy": accuracy,
        "category": category,
        "highlight": False,
    })

for stem, display_name in CLOSED_MODELS.items():
    df = pl.read_parquet(RESULTS_DIR / f"{stem}.parquet")
    accuracy = round(df["correct"].cast(pl.Int64).mean() * 100, 1)
    accuracy_entries.append({"name": display_name, "accuracy": accuracy, "highlight": False})

accuracy_entries.sort(key=lambda e: e["accuracy"])
point_entries.sort(key=lambda e: e["accuracy"])

(OUT_DIR / "models-accuracy.json").write_text(json.dumps({
    "title": "Accuracy of the models",
    "models": accuracy_entries,
}, indent=2) + "\n")

(OUT_DIR / "accuracy-vs-time.json").write_text(json.dumps({
    "xLabel": "Inference time (s)",
    "yLabel": "Accuracy (%)",
    "points": point_entries,
}, indent=2) + "\n")

print("models-accuracy.json:")
for e in accuracy_entries:
    print(f"  {e['name']:20s}  {e['accuracy']:>5.1f}%")
print("\naccuracy-vs-time.json:")
for p in point_entries:
    print(f"  {p['name']:20s}  t={p['time']:>7.2f}s  acc={p['accuracy']:>5.1f}%  [{p['category']}]")
