"""Build ``rerun_collisions.json`` -- the benchmark items hit by the provenance bug.

The v4 agent resolved the bare filename the VLM handed to RAFT / SAM / FFT
through a ``basename -> Path`` cache in which the last ``rglob`` hit won.  Every
benchmark clip whose basename also exists somewhere else under
``final_dataset`` (in ``cosmos-predict2``, or in the other benchmark pool) may
therefore have been analysed with tool output computed on the *wrong* file, and
has to be re-run with the fixed agent (``src/agentic/video_resolve.py``).

Affected set
    { benchmark (source, video) : basename occurs > 1 time under final_dataset }

Benchmark items come from ``questions_cosmos-drive-dreams.json`` /
``questions_cosmos-predict1.json`` (entries flagged ``"remove": 1`` are not part
of the benchmark, and the stem ``nOP1blfMCTg_48624_14b`` is excluded -- it is the
one clip the analysed parquets drop, 494 not 495 predict1 videos).

The output is produced by *filtering* ``dataset/dataset_final/Questions_raw.json``
itself (what ``main_multi_tools_v4.py`` consumes) rather than by re-rendering the
prompts, so the question strings -- including the ``"The answer is one of the
following : 1 2 3"`` suffix that ``prep_data_direct_prompting.py`` appends to the
Likert questions -- and the ``/mnt/vita/scratch/...`` prefix are byte-identical to
the original run.  It can be dropped straight into ``--questions_file``.

``v4o`` used ``Questions_raw.json`` (the ``main_multi_tools_v4.py`` default) and
``v4p`` used the same file plus ``--ins`` (hints are injected at runtime from
``resources/hint_questions.json``), so a single file serves both re-runs.

Usage (from the repo root)::

    .venv/bin/python src/agentic/counterfactual/build_rerun_collisions.py \
        --final-dataset "/home/matthieu/Documents/Projet de master/final_dataset"
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO / "dataset" / "dataset_final"
DEFAULT_FINAL_DATASET = REPO.parent / "final_dataset"

#: prefix used inside Questions_raw.json (kept verbatim so the pod resolves it)
POD_PREFIX = "/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset"

SOURCES = {
    "cosmos-drive-dreams": "questions_cosmos-drive-dreams.json",
    "cosmos-predict1": "questions_cosmos-predict1.json",
}

#: the one predict1 clip absent from every analysed parquet
SKIP_STEMS = {"nOP1blfMCTg_48624_14b"}


def duplicate_basenames(final_dataset: Path) -> dict[str, list[Path]]:
    """basename -> [paths] for every mp4 basename that occurs more than once."""
    cache: dict[str, list[Path]] = collections.defaultdict(list)
    for ext in ("*.mp4", "*.MP4"):
        for p in final_dataset.rglob(ext):
            if p.is_file():
                cache[p.name].append(p)
    return {k: sorted(v) for k, v in cache.items() if len(v) > 1}


def load_benchmark(dataset_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """(source, relative video) -> qa_pairs, for the benchmark items only."""
    out: dict[tuple[str, str], list[dict]] = {}
    for source, fname in SOURCES.items():
        for entry in json.loads((dataset_dir / fname).read_text()):
            if entry.get("remove"):
                continue
            video = entry["video"]
            if Path(video).stem in SKIP_STEMS:
                continue
            out[(source, video)] = entry["qa_pairs"]
    return out


def item_key(item: dict) -> tuple[str, str]:
    """(source, relative video) of a Questions_raw.json chat item."""
    video = next(c for c in item["content"] if c["type"] == "video")["video"]
    rel = video.split("final_dataset/", 1)[1]
    source, rest = rel.split("/", 1)
    return source, rest


def build(dataset_dir: Path, final_dataset: Path) -> tuple[list[dict], dict]:
    dupes = duplicate_basenames(final_dataset)
    bench = load_benchmark(dataset_dir)
    raw = json.loads((dataset_dir / "Questions_raw.json").read_text())

    affected = {k for k in bench if Path(k[1]).name in dupes}
    # filter Questions_raw.json so the prompts stay byte-identical
    items = [it for it in raw if item_key(it) in affected]

    raw_keys = {item_key(it) for it in raw}
    missing = sorted(affected - raw_keys)

    per_source = collections.Counter(s for s, _ in affected)
    per_source_items = collections.Counter(item_key(it)[0] for it in items)
    per_pools = collections.Counter()
    for _source, video in affected:
        pools = tuple(sorted({p.relative_to(final_dataset).parts[0]
                              for p in dupes[Path(video).name]}))
        per_pools[pools] += 1

    stats = {
        "final_dataset": str(final_dataset),
        "mp4_files_in_duplicate_groups": sum(len(v) for v in dupes.values()),
        "duplicate_basenames": len(dupes),
        "benchmark_videos": len(bench),
        "benchmark_items": sum(len(v) for v in bench.values()),
        "questions_raw_items": len(raw),
        "affected_videos": len(affected),
        "affected_items": len(items),
        "affected_videos_per_source": dict(per_source),
        "affected_items_per_source": dict(per_source_items),
        "collision_pools": {" + ".join(k): v for k, v in per_pools.items()},
        "affected_videos_absent_from_questions_raw": len(missing),
    }
    return items, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                    help="dir holding questions_cosmos-*.json")
    ap.add_argument("--final-dataset", type=Path, default=DEFAULT_FINAL_DATASET,
                    help="local copy of the final_dataset video tree")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("rerun_collisions.json"))
    args = ap.parse_args()

    items, stats = build(args.dataset_dir, args.final_dataset)

    print("=" * 68)
    print("rerun_collisions.json")
    print("=" * 68)
    for k, v in stats.items():
        print(f"  {k:32s} {v}")
    print("=" * 68)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"wrote {len(items)} items -> {args.out} "
          f"({args.out.stat().st_size / 1024:.0f} KiB)")

    if stats["affected_videos"] != 188 or stats["affected_items"] != 736:
        print(f"WARNING: expected 188 videos / 736 items, got "
              f"{stats['affected_videos']} / {stats['affected_items']} -- "
              f"is --final-dataset the full tree (incl. cosmos-predict2)?")


if __name__ == "__main__":
    main()
