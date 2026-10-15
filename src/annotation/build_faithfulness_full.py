#!/usr/bin/env python3
"""Build the full-scale faithfulness grading set: every v4o answer (7371 items, 1597 clips).

Outputs under results/annotation/faithfulness_full/:
  items.jsonl            one row per (clip, question) with explanation + true FFT value + collision flag
  sheets/<clip_hash>.jpg one 2x4 contact sheet (8 frames, time order) per clip
  batches/batch_NNN.json items grouped by clip, --videos-per-batch clips per batch (grader unit)
Item ids use the same scheme as src/annotation/build_manifest.py, so they overlap with the 100-item study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "annotation" / "faithfulness_full"
VIDEO_ROOT = ROOT.parent / "final_dataset"
PREDICT1_FOLDERS = {"artifacts"}


def item_id(clip: str, q: str) -> str:
    return hashlib.sha1(f"{clip}||{q}".encode()).hexdigest()[:12]


def verdict(a: float) -> str:
    if a >= 0.048:
        return f"GENERATED ({'high' if a >= 0.055 else 'moderate'} confidence)"
    return f"REAL ({'high' if a <= 0.040 else 'moderate'} confidence)"


def make_sheet(src: Path, dst: Path) -> str:
    if dst.exists():
        return "skip"
    n = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(src)],
                           capture_output=True, text=True).stdout.strip() or 121)
    idx = [int((i + 0.5) * n / 8) for i in range(8)]
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf",
                        f"select='{sel}',scale=400:220,tile=4x2", "-vsync", "vfr", "-frames:v", "1",
                        "-q:v", "4", str(dst)], capture_output=True, text=True)
    return "ok" if r.returncode == 0 and dst.exists() else f"ERR {r.stderr[-200:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-per-batch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    d = pl.read_parquet(ROOT / "results" / "v4o_analyzed.parquet")
    d = d.with_columns(pl.col("video").str.split("/").list.first().alias("folder"))
    d = d.with_columns(
        pl.when(pl.col("folder").is_in(list(PREDICT1_FOLDERS))).then(pl.lit("cosmos-predict1"))
        .otherwise(pl.lit("cosmos-drive-dreams")).alias("source"))
    d = d.with_columns((pl.col("source") + "/" + pl.col("video")).alias("clip"))

    fft = pl.read_parquet(ROOT / "results" / "fft" / "features_raw_f60.parquet").unique("clip_id")
    aniso = dict(zip(fft["clip_id"], fft["aniso_mean"]))

    collisions = set()
    def _walk(o):
        if isinstance(o, dict):
            if o.get("type") == "video" and isinstance(o.get("video"), str):
                collisions.add(o["video"].split("/final_dataset/")[-1])
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
    _walk(json.load(open(ROOT / "src" / "agentic" / "counterfactual" / "rerun_collisions.json")))

    # 26 negative/ clips are not stored locally; the HF copy is byte-identical to the same basename in the
    # violation folder (md5 checked), so resolve to that sibling for the sheet and the FFT value.
    sibling = {}
    for c in d["clip"].unique():
        if not (VIDEO_ROOT / c).exists():
            src, rest = c.split("/", 1)
            hits = sorted((VIDEO_ROOT / src).glob(f"*/{Path(rest).name}"))
            if hits:
                sibling[c] = str(hits[0].relative_to(VIDEO_ROOT))

    rows, missing_fft = [], 0
    for r in d.iter_rows(named=True):
        local = sibling.get(r["clip"], r["clip"])
        src, rest = local.split("/", 1)
        key = ("p1/" if src == "cosmos-predict1" else "dd/") + rest[: -len(".mp4")]
        a = aniso.get(key)
        missing_fft += a is None
        h = hashlib.sha1(r["clip"].encode()).hexdigest()[:12]
        rows.append({
            "item_id": item_id(r["clip"], r["question"]),
            "clip": r["clip"],
            "category": r["category"],
            "question": r["question"],
            "model_answer": r["vlm_answer"],
            "explanation": r["raw_output"],
            "fft_anisotropy_actual": f"{a:.4f}" if a is not None else "NA",
            "fft_verdict_actual": verdict(a) if a is not None else "NA",
            "provenance_collision": r["clip"] in collisions,
            "local_clip": local,
            "sheet": f"results/annotation/faithfulness_full/sheets/{h}.jpg",
        })
    assert len({x["item_id"] for x in rows}) == len(rows), "duplicate item ids"
    with open(OUT / "items.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    clips = sorted({x["clip"] for x in rows})
    sheet_of = {x["clip"]: x["sheet"] for x in rows}
    local_of = {x["clip"]: x["local_clip"] for x in rows}
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda c: make_sheet(VIDEO_ROOT / local_of[c], ROOT / sheet_of[c]), clips))
    errs = [(c, r) for c, r in zip(clips, res) if r.startswith("ERR")]

    by_clip: dict[str, list] = {}
    for x in rows:
        by_clip.setdefault(x["clip"], []).append(x)
    for old in (OUT / "batches").glob("batch_*.json"):
        old.unlink()
    nb = 0
    for i in range(0, len(clips), args.videos_per_batch):
        chunk = clips[i:i + args.videos_per_batch]
        items = [x for c in chunk for x in by_clip[c]]
        (OUT / "batches" / f"batch_{nb:03d}.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
        nb += 1
    print(f"items {len(rows)} | clips {len(clips)} | missing fft {missing_fft} | collisions "
          f"{sum(x['provenance_collision'] for x in rows)} items | sheets err {len(errs)} | batches {nb}")
    for e in errs[:5]:
        print(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
