#!/usr/bin/env python3
"""Merge the per-batch Sonnet grades of the full v4o set and report hallucination / consistency rates.

Rates carry 95 % cluster-bootstrap CIs (clusters = base scenario, all weather variants together).
Also: agreement with the earlier 100-item graders on overlapping item ids (grader stability).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "annotation"))
from stats import add_cluster_ids  # noqa: E402
from agreement import cohen_kappa  # noqa: E402

BASE = ROOT / "results" / "annotation" / "faithfulness_full"
B, SEED = 2000, 0


def boot(df: pl.DataFrame, col: str) -> tuple[float, float, float]:
    g = df.group_by("scenario_id").agg(pl.len().alias("n"), pl.col(col).sum().alias("k"))
    n, k = g["n"].to_numpy(), g["k"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(n), size=(B, len(n)))
    reps = k[idx].sum(1) / n[idx].sum(1)
    return float(k.sum() / n.sum()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def table(df: pl.DataFrame, by: str | None) -> list[dict]:
    groups = [("all", df)] if by is None else [(str(key[0]), sub) for key, sub in df.group_by(by, maintain_order=True)]
    out = []
    for name, sub in sorted(groups, key=lambda t: -t[1].height):
        row = {"group": name, "n": sub.height}
        for col in ("hallucination", "false_claim", "missed_event", "inconsistent"):
            p, lo, hi = boot(sub, col)
            row[col] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)]
        out.append(row)
    return out


import re

FFT_RE = re.compile(r"anisotrop|\bfft\b|quot|confidence|0\.0\d{2}", re.I)
MOTION_RE = re.compile(r"\bstop|stationary|static|proceed|moving|motion|overtak|\bpass|forward|crossing|\bran\b|runs|halt|"
                       r"brak|clos(e|ing)|distance|lane change|speed", re.I)


def claim_type(note: str) -> str:
    """Approximate type of a flagged claim from the grader note (keyword rules, FFT > motion > static)."""
    if FFT_RE.search(note or ""):
        return "fft_misquote"
    if MOTION_RE.search(note or ""):
        return "motion"
    return "static_content"


def md(rows: list[dict], title: str) -> str:
    s = [f"### {title}", "", "| group | n | hallucination % | false claim % | missed event % | inconsistent % |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda v: f"{v[0]} [{v[1]}, {v[2]}]"  # noqa: E731
        s.append(f"| {r['group'][:60]} | {r['n']} | {f(r['hallucination'])} | {f(r['false_claim'])} | "
                 f"{f(r['missed_event'])} | {f(r['inconsistent'])} |")
    return "\n".join(s) + "\n"


def main() -> int:
    items = pl.read_ndjson(BASE / "items.jsonl")
    grades = []
    for p in sorted((BASE / "ratings").glob("batch_*.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("item_id"):
                    r["batch"] = p.stem
                    grades.append(r)
    g = pl.DataFrame(grades).unique("item_id", keep="last")
    bad = g.filter(~pl.col("consistent").is_in(["Y", "N"]) | ~pl.col("false_claim").is_in(["Y", "N"])
                   | ~pl.col("missed_event").is_in(["Y", "N"]))
    extra = g.filter(~pl.col("item_id").is_in(items["item_id"]))
    d = items.join(g.drop("batch"), on="item_id", how="inner")
    d = d.with_columns(
        (pl.col("false_claim") == "Y").alias("false_claim"),
        (pl.col("missed_event") == "Y").alias("missed_event"),
        (pl.col("consistent") == "N").alias("inconsistent"),
    ).with_columns((pl.col("false_claim") | pl.col("missed_event")).alias("hallucination"))
    d = d.with_columns(pl.when(pl.col("hallucination"))
                       .then(pl.col("note").map_elements(claim_type, return_dtype=pl.Utf8))
                       .otherwise(pl.lit("none")).alias("claim_type"))
    d = d.with_columns(*[(pl.col("claim_type") == t).alias(f"h_{t}") for t in ("fft_misquote", "motion", "static_content")])
    v4o = pl.read_parquet(ROOT / "results" / "v4o_analyzed.parquet").select("video", "question", "correct")
    d = d.with_columns(pl.col("clip").str.split("/").list.slice(1).list.join("/").alias("video"))
    d = d.join(v4o, on=["video", "question"], how="left")
    d = add_cluster_ids(d).with_columns(
        pl.when(pl.col("correct")).then(pl.lit("answer correct")).otherwise(pl.lit("answer wrong")).alias("correctness"),
        pl.col("clip").str.split("/").list.first().alias("source"),
        pl.when(pl.col("provenance_collision")).then(pl.lit("collision clip")).otherwise(pl.lit("clean clip"))
        .alias("provenance"))

    res = {
        "n_items_total": items.height, "n_graded": d.height, "n_invalid_rows": bad.height,
        "n_unknown_ids": extra.height,
        "batches_done": sorted({r["batch"] for r in grades}),
        "overall": table(d, None), "by_category": table(d, "category"), "by_question": table(d, "question"),
        "by_source": table(d, "source"), "by_correctness": table(d, "correctness"),
        "by_provenance": table(d, "provenance"),
        "claim_type_counts": dict(d.group_by("claim_type").len().iter_rows()),
        "claim_type_rates": {t: list(boot(d, f"h_{t}")) for t in ("fft_misquote", "motion", "static_content")},
        "claim_type_by_provenance": {name: {t: round(100 * sub[f"h_{t}"].mean(), 2)
                                            for t in ("fft_misquote", "motion", "static_content")}
                                     for (name,), sub in d.group_by("provenance")},
    }
    # replicate graders (same batch, fresh grader) -> between-grader agreement and rate spread
    rep = {}
    for p in sorted((BASE / "ratings_rep").glob("batch_*.csv")):
        a = {r["item_id"]: r for r in csv.DictReader(open(BASE / "ratings" / p.name))}
        b = {r["item_id"]: r for r in csv.DictReader(open(p))}
        ids = [i for i in b if i in a]
        hz = lambda r: "Y" if (r["false_claim"] == "Y" or r["missed_event"] == "Y") else "N"  # noqa: E731
        pairs = [(hz(a[i]), hz(b[i])) for i in ids]
        if pairs:
            rep[p.stem] = {"n": len(pairs), "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 3),
                           "kappa": cohen_kappa(pairs), "rate_main": round(sum(x == "Y" for x, _ in pairs) / len(pairs), 3),
                           "rate_rep": round(sum(y == "Y" for _, y in pairs) / len(pairs), 3)}
    res["replicate_graders"] = rep
    # grader stability vs the 100-item study (same item ids)
    stab = {}
    for name in ("rater_sonnet_v2", "rater_opus_v2"):
        p = ROOT / "results" / "annotation" / "faithfulness_ratings" / f"{name}.csv"
        old = {r["item_id"]: r for r in csv.DictReader(open(p))}
        both = [r for r in d.iter_rows(named=True) if r["item_id"] in old]
        if not both:
            continue
        pairs = [("Y" if r["hallucination"] else "N",
                  "Y" if (old[r["item_id"]]["false_claim"] == "Y" or old[r["item_id"]]["missed_event"] == "Y") else "N")
                 for r in both]
        stab[name] = {"n": len(pairs), "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3),
                      "kappa_hallucination": cohen_kappa(pairs),
                      "rate_new": round(sum(a == "Y" for a, _ in pairs) / len(pairs), 3),
                      "rate_old": round(sum(b == "Y" for _, b in pairs) / len(pairs), 3)}
    res["stability_vs_100_item_study"] = stab

    (BASE / "faithfulness_full_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    d.drop("explanation").write_parquet(BASE / "graded_items.parquet")
    out = [f"# Explanation faithfulness — full v4o set, Sonnet 5 grader (rubric v2)\n",
           f"Graded {d.height} / {items.height} items ({len(res['batches_done'])} batches). "
           f"Invalid rows {bad.height}. CIs: 95 % cluster bootstrap over base scenarios.\n",
           md(res["overall"], "Overall"), md(res["by_category"], "By category"),
           md(res["by_source"], "By generator"), md(res["by_correctness"], "By answer correctness"),
           md(res["by_provenance"], "By provenance (basename-collision clips)"), md(res["by_question"], "By question"),
           "### Claim types (keyword classification of grader notes; approximate)\n",
           "```\n" + json.dumps({k: res[k] for k in ("claim_type_counts", "claim_type_rates", "claim_type_by_provenance")},
                                 indent=1) + "\n```\n",
           "### Replicate graders (same batch, fresh grader)\n", "```\n" + json.dumps(rep, indent=1) + "\n```\n",
           "### Grader stability vs the 100-item study\n", "```\n" + json.dumps(stab, indent=1) + "\n```\n"]
    (BASE / "faithfulness_full_summary.md").write_text("\n".join(out))
    print("\n".join(out[:3]))
    print(json.dumps(stab, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
