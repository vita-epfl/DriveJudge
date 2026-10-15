"""
Did the agent's FFT tool actually look at the clip it was asked about?

190 basenames exist in more than one pool under ``final_dataset`` (a
cosmos-predict1 clip and a cosmos-drive-dreams clip can share a filename, and a
third pool on disk duplicates many of them again).  The agent used to resolve a
video by basename against a single ``basename -> Path`` cache, so for those
clips the FFT / RAFT / SAM tools could silently have run on *another pool's*
copy -- which, given that the whole real-vs-generated signal is a container
fingerprint, would be a provenance bug worth knowing about.

The check is empirical: the agent quotes the anisotropy value in its answer, and
``src/agentic/fft/compute_fft.py`` prints it as ``anisotropy = <mean>`` rounded
to four decimals.  Extract the quoted number from ``results/v4o_analyzed.parquet``
for the colliding clips and see which file on disk reproduces it.

Note on the regex: the *tool* prints ``anisotropy = 0.0456``, but the model
paraphrases -- in v4o only 4 of the 4307 quoted values keep the literal ``=``
form, the rest read "anisotropy score of 0.0456".  The pattern below therefore
accepts any short filler between the word and the number, and the summary
reports how many rows each spelling accounts for.

Usage
-----
    .venv/bin/python src/fft/provenance_check.py
    .venv/bin/python src/fft/provenance_check.py --results results/v4o_analyzed.parquet

Re-runnable: it uses whatever rows ``features_raw_f60.parquet`` currently has
and reports how many clips are still waiting for features.
Output: ``results/fft/provenance_check.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft.common import (  # noqa: E402
    DEFAULT_MANIFEST,
    FINAL_DATASET_ROOT,
    GENERATED_SOURCES,
    REPO,
    RESULTS_FFT_DIR,
    make_clip_id,
    rel_to_repo,
)

#: strict form printed by the tool, and the paraphrases the model actually uses.
ANISO_RE_STRICT = re.compile(r"anisotropy = (\d\.\d+)")
ANISO_RE_LOOSE = re.compile(r"[Aa]nisotrop\w*[^0-9\n]{0,25}?(\d\.\d+)")

#: the tool prints ``f"{mean:.4f}"``, so an exact quote agrees to 5e-5.
TOL_EXACT = 5e-5
TOL_LOOSE = 5e-4

#: folder -> source pool, the same rule ``src/analysis/stats.py`` uses.
PREDICT1_FOLDERS = {"artifacts"}


def basename_index() -> dict[str, list[dict]]:
    """``basename -> [{clip_id, generator, path}]`` for every mp4 under ``final_dataset``.

    Deliberately read off the *filesystem* rather than off the manifest: the
    manifest excludes cosmos-predict2 (it is not part of the paper), but those
    files are still on disk and the agent's basename cache could still have
    resolved to them, so they must keep counting as collisions here.
    """
    idx: dict[str, list[dict]] = defaultdict(list)
    for source, gen in GENERATED_SOURCES.items():
        root = FINAL_DATASET_ROOT / source
        if not root.is_dir():
            continue
        for mp4 in root.rglob("*.mp4"):
            folder = str(mp4.parent.relative_to(root))
            idx[mp4.name].append(
                {
                    "clip_id": make_clip_id(gen, None if folder == "." else folder, mp4.stem),
                    "generator": gen,
                    "path": str(mp4),
                }
            )
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "results" / "v4o_analyzed.parquet")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--features", type=Path, default=RESULTS_FFT_DIR / "features_raw_f60.parquet")
    ap.add_argument("--out", type=Path, default=RESULTS_FFT_DIR / "provenance_check.json")
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args()

    for p in (args.results, args.manifest):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    man = pl.read_parquet(args.manifest)
    res = pl.read_parquet(args.results)
    print(f"agent results: {rel_to_repo(args.results)}  ({res.height} rows, "
          f"{res['video'].n_unique()} distinct videos)")

    idx = basename_index()
    colliding = {b for b, rows in idx.items() if len(rows) > 1}
    print(f"final_dataset: {sum(len(v) for v in idx.values())} files, {len(idx)} basenames, "
          f"{len(colliding)} of them colliding")

    # ---- the colliding benchmark clips ----------------------------------
    videos = sorted(set(res["video"].to_list()))
    coll_videos = [v for v in videos if Path(v).name in colliding]
    print(f"benchmark clips with a colliding basename: {len(coll_videos)}")

    # ---- features --------------------------------------------------------
    aniso: dict[str, float] = {}
    if args.features.exists():
        f = pl.read_parquet(args.features).filter(
            (pl.col("condition") == "raw")
            & (pl.col("max_frames") == args.max_frames)
            & (pl.col("frame_offset") == 0)
            & pl.col("error").is_null()
            & pl.col("aniso_mean").is_not_null()
        )
        aniso = dict(zip(f["clip_id"].to_list(), f["aniso_mean"].to_list()))
        print(f"features    : {rel_to_repo(args.features)}  ({len(aniso)} clips at "
              f"max_frames={args.max_frames})")
    else:
        print(f"features    : {rel_to_repo(args.features)} does not exist yet -- "
              f"nothing can be matched, run src/fft/extract_features.py")

    # ---- per-row verdicts ------------------------------------------------
    verdicts: Counter = Counter()
    spelling: Counter = Counter()
    details = []
    per_video: dict[str, Counter] = defaultdict(Counter)

    rows = res.filter(pl.col("video").is_in(coll_videos)).select("video", "question", "raw_output")
    for r in rows.iter_rows(named=True):
        text = r["raw_output"] or ""
        strict = ANISO_RE_STRICT.findall(text)
        loose = ANISO_RE_LOOSE.findall(text)
        spelling["strict_equals_form"] += 1 if strict else 0
        if not loose:
            verdicts["no_value_quoted"] += 1
            per_video[r["video"]]["no_value_quoted"] += 1
            continue
        spelling["value_quoted"] += 1
        quoted = float(loose[0])

        name = Path(r["video"]).name
        stem = name[:-4]
        folder = r["video"].split("/")[0]
        want_gen = "p1" if folder in PREDICT1_FOLDERS else "dd"
        candidates = idx[name]
        correct = [c for c in candidates if c["generator"] == want_gen]
        others = [c for c in candidates if c["generator"] != want_gen]
        if not correct:
            verdicts["no_correct_candidate"] += 1
            per_video[r["video"]]["no_correct_candidate"] += 1
            continue

        have_all = all(c["clip_id"] in aniso for c in candidates)
        hit_correct = [c["clip_id"] for c in correct
                       if c["clip_id"] in aniso and abs(aniso[c["clip_id"]] - quoted) <= TOL_EXACT]
        hit_other = [c["clip_id"] for c in others
                     if c["clip_id"] in aniso and abs(aniso[c["clip_id"]] - quoted) <= TOL_EXACT]
        if not have_all and not hit_correct and not hit_other:
            verdicts["features_missing"] += 1
            per_video[r["video"]]["features_missing"] += 1
            continue

        if hit_correct and hit_other:
            v = "ambiguous_both_match"
        elif hit_correct:
            v = "matches_correct_clip"
        elif hit_other:
            v = "matches_other_pool"
        else:
            loose_correct = [c["clip_id"] for c in correct
                             if c["clip_id"] in aniso and abs(aniso[c["clip_id"]] - quoted) <= TOL_LOOSE]
            loose_other = [c["clip_id"] for c in others
                           if c["clip_id"] in aniso and abs(aniso[c["clip_id"]] - quoted) <= TOL_LOOSE]
            v = ("near_miss_correct" if loose_correct else
                 "near_miss_other_pool" if loose_other else "matches_nothing")
        verdicts[v] += 1
        per_video[r["video"]][v] += 1
        details.append({
            "video": r["video"], "stem": stem, "quoted": quoted, "verdict": v,
            "correct_clip": correct[0]["clip_id"],
            "correct_aniso": aniso.get(correct[0]["clip_id"]),
            "other_clips": {c["clip_id"]: aniso.get(c["clip_id"]) for c in others},
        })

    print("\n=== per-row verdicts (one row = one (video, question) answer) ===")
    total = sum(verdicts.values())
    for k, n in verdicts.most_common():
        print(f"  {k:<24} {n:5d}  ({100 * n / max(1, total):.1f}%)")
    print(f"  {'TOTAL':<24} {total:5d}")

    resolved = verdicts["matches_correct_clip"] + verdicts["matches_other_pool"] + verdicts["ambiguous_both_match"]
    if resolved:
        print(f"\n  of the {resolved} rows whose quoted value matches a file on disk: "
              f"{verdicts['matches_correct_clip']} the correct clip, "
              f"{verdicts['matches_other_pool']} another pool's clip, "
              f"{verdicts['ambiguous_both_match']} both (the two files agree to 4 dp)")
    else:
        print("\n  no row could be resolved yet -- the feature file does not cover both sides "
              "of any collision.")

    video_verdict = {}
    for v, c in per_video.items():
        video_verdict[v] = c.most_common(1)[0][0]
    by_video = Counter(video_verdict.values())
    print("\n=== per-video majority verdict ===")
    for k, n in by_video.most_common():
        print(f"  {k:<24} {n:5d}")

    payload = {
        "results_file": rel_to_repo(args.results),
        "features_file": rel_to_repo(args.features),
        "features_available": len(aniso),
        "final_dataset_root": str(FINAL_DATASET_ROOT),
        "n_files_indexed": sum(len(v) for v in idx.values()),
        "n_basenames": len(idx),
        "n_colliding_basenames": len(colliding),
        "n_benchmark_videos_with_collision": len(coll_videos),
        "tolerance_exact": TOL_EXACT,
        "tolerance_loose": TOL_LOOSE,
        "regex_strict": ANISO_RE_STRICT.pattern,
        "regex_loose": ANISO_RE_LOOSE.pattern,
        "rows_matching_strict_regex": spelling["strict_equals_form"],
        "rows_with_a_quoted_value": spelling["value_quoted"],
        "row_verdicts": dict(verdicts),
        "video_verdicts": dict(by_video),
        "details": details[:500],
        "partial": len(aniso) < man.height,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {rel_to_repo(args.out)}")
    if payload["partial"]:
        print("!! features_raw_f60.parquet is still incomplete -- re-run when it is finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
