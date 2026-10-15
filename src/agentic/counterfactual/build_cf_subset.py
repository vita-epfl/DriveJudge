"""Build the two E10 counterfactual evaluation subsets.

    .venv/bin/python src/agentic/counterfactual/build_cf_subset.py

``subsets/cf_forensic.json``  600 items -- the questions that route to SOP-B
    (``get_frequency_analysis`` + ``get_masks``):

    * 300 x "Is it a real video or generated ?", 150 cosmos-drive-dreams /
      150 cosmos-predict1, each half v4o-correct / half v4o-wrong *where
      possible* (predict1 only has 60 wrong ones left after the exclusions, so
      that half is capped and the correct half absorbs the rest),
    * 300 x Artifacts-understanding items (cosmos-predict1), balanced over the
      two artifact questions and over v4o correctness.

``subsets/cf_motion.json``  300 items -- the questions that route to SOP-C
    (``get_motion_info``): folders ``crossing_red_lights`` / ``crossing_stop`` /
    ``aggressive_takeover``, balanced over

        "Is it a safe way of driving for the ego car ?"
        "Is the ego car following the traffic rules ?"
        "Has the ego car stopped ?"
        "On which side is the ego car overtaking ?"

``subsets/cf_subsets_gt.json``  item -> ground_truth, category, folder,
    scenario_id, v4o_correct (plus the raw prompt, so ``analyze_cf.py`` can join
    the agent's echoed question back onto the ground truth).

Two exclusions keep the causal read-out clean:

* **basename collisions.**  Any clip whose mp4 basename occurs more than once
  under ``final_dataset`` is dropped: for those the v4o baseline itself may have
  run the tools on the wrong file (see ``build_rerun_collisions.py``), so a flip
  could not be attributed to the intervention.
* **at most 2 items per (question, scenario).**  A "scenario" is the base clip
  before the weather re-render, i.e. the stem with ``_(Original|Morning|Night|
  Snowy|Rainy|Foggy|Sunny|Golden hour)`` and then ``_(7b|14b)`` stripped
  (``src/analysis/stats.py``).  Eight weather variants of one scene are eight
  near-identical items; capping keeps the effective sample size close to the
  nominal one and keeps the cluster bootstrap in ``analyze_cf.py`` honest.
  The cap is applied per *question* -- a global cap of 2 per scenario would make
  the requested subset sizes unreachable (predict1 has ~186 artifact scenarios
  but the forensic subset needs 450 predict1 items).

Selection is deterministic: ``random.Random(--seed)``, default 0.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from analysis.stats import OPTION_SUFFIX_RE, add_cluster_ids  # noqa: E402

DEFAULT_PARQUET = REPO / "results" / "v4o_analyzed.parquet"
DEFAULT_DATASET = REPO / "dataset" / "dataset_final"
DEFAULT_FINAL_DATASET = REPO.parent / "final_dataset"
DEFAULT_OUT = Path(__file__).resolve().parent / "subsets"

Q_REALGEN = "Is it a real video or generated ?"
Q_ARTIFACTS = [
    "Do some objects change shape or appearance ?",
    "Does any object appear or disappear without continuous motion ?",
]
MOTION_FOLDERS = ["crossing_red_lights", "crossing_stop", "aggressive_takeover"]
Q_MOTION = [
    "Is it a safe way of driving for the ego car ?",
    "Is the ego car following the traffic rules ?",
    "Has the ego car stopped ?",
    "On which side is the ego car overtaking ?",
]

MAX_PER_SCENARIO = 2


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def duplicate_basenames(final_dataset: Path) -> set[str]:
    counts: collections.Counter[str] = collections.Counter()
    for ext in ("*.mp4", "*.MP4"):
        for p in final_dataset.rglob(ext):
            if p.is_file():
                counts[p.name] += 1
    return {name for name, n in counts.items() if n > 1}


def clean_question(text: str) -> str:
    return re.sub(OPTION_SUFFIX_RE, "", text).strip()


def load_prompts(dataset_dir: Path) -> dict[tuple[str, str, str], dict]:
    """(source, video, cleaned question) -> the Questions_raw.json chat item."""
    raw = json.loads((dataset_dir / "Questions_raw.json").read_text())
    out: dict[tuple[str, str, str], dict] = {}
    for item in raw:
        video = next(c for c in item["content"] if c["type"] == "video")["video"]
        text = next(c for c in item["content"] if c["type"] == "text")["text"]
        rel = video.split("final_dataset/", 1)[1]
        source, video_rel = rel.split("/", 1)
        out[(source, video_rel, clean_question(text))] = item
    return out


def load_candidates(parquet: Path, final_dataset: Path) -> pl.DataFrame:
    df = pl.read_parquet(parquet)
    df = df.rename({"model": "source"}) if "model" in df.columns else df
    df = df.with_columns(
        pl.col("question").str.replace(OPTION_SUFFIX_RE, "").str.strip_chars().alias("question"),
        pl.col("correct").cast(pl.Boolean).alias("v4o_correct"),
    )
    df = add_cluster_ids(df)
    dupes = duplicate_basenames(final_dataset)
    df = df.with_columns(
        pl.col("video").str.split("/").list.last().is_in(list(dupes)).alias("collides")
    )
    return df


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _rows(df: pl.DataFrame) -> list[dict]:
    return df.to_dicts()


def take(rows: list[dict], target: int, strata: list[str], rng: random.Random,
         cap: int = MAX_PER_SCENARIO,
         used: collections.Counter | None = None) -> list[dict]:
    """Round-robin over ``strata`` groups, at most ``cap`` per (question, scenario).

    ``used`` lets several calls share one budget (so the same scenario is not
    used twice by two strata of the same question).
    """
    used = used if used is not None else collections.Counter()
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[tuple(r[c] for c in strata)].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: (r["source"], r["video"], r["question"]))
        rng.shuffle(groups[key])

    order = sorted(groups)
    picked: list[dict] = []
    exhausted: set[tuple] = set()
    while len(picked) < target and len(exhausted) < len(order):
        for key in order:
            if len(picked) >= target:
                break
            if key in exhausted:
                continue
            bucket = groups[key]
            while bucket:
                r = bucket.pop()
                budget_key = (r["question"], r["scenario_id"])
                if used[budget_key] >= cap:
                    continue
                used[budget_key] += 1
                picked.append(r)
                break
            else:
                exhausted.add(key)
    return picked


# ---------------------------------------------------------------------------
# subsets
# ---------------------------------------------------------------------------


def build_forensic(df: pl.DataFrame, rng: random.Random, n_realgen: int,
                   n_artifacts: int) -> list[dict]:
    picked: list[dict] = []
    budget: collections.Counter = collections.Counter()

    # -- real / generated: 150 per source, half correct / half wrong ---------
    for source in ("cosmos-drive-dreams", "cosmos-predict1"):
        per_source = n_realgen // 2
        pool = df.filter((pl.col("question") == Q_REALGEN) & (pl.col("source") == source))
        chosen: list[dict] = []
        # the smaller correctness half first, so "where possible" degrades
        # gracefully instead of starving the larger half
        halves = sorted(
            ((v, _rows(pool.filter(pl.col("v4o_correct") == v))) for v in (True, False)),
            key=lambda kv: len(kv[1]),
        )
        remaining = per_source
        for i, (_value, rows) in enumerate(halves):
            want = min(remaining, per_source // 2 if i == 0 else remaining)
            got = take(rows, want, ["source"], rng, used=budget)
            chosen += got
            remaining -= len(got)
        if remaining > 0:  # top up from whatever is left
            got = take(_rows(pool), remaining, ["v4o_correct"], rng, used=budget)
            seen = {(r["source"], r["video"], r["question"]) for r in chosen}
            chosen += [r for r in got if (r["source"], r["video"], r["question"]) not in seen]
        picked += chosen

    # -- artifacts (predict1), balanced over question x correctness ----------
    art = df.filter(
        (pl.col("category") == "Artifacts understanding")
        & (pl.col("question").is_in(Q_ARTIFACTS))
    )
    picked += take(_rows(art), n_artifacts, ["question", "v4o_correct"], rng, used=budget)
    return picked


def build_motion(df: pl.DataFrame, rng: random.Random, n_total: int) -> list[dict]:
    pool = df.filter(
        pl.col("folder").is_in(MOTION_FOLDERS) & pl.col("question").is_in(Q_MOTION)
    )
    per_q = n_total // len(Q_MOTION)
    budget: collections.Counter = collections.Counter()
    picked: list[dict] = []
    shortfall = 0
    for q in Q_MOTION:
        rows = _rows(pool.filter(pl.col("question") == q))
        got = take(rows, per_q, ["folder", "v4o_correct"], rng, used=budget)
        picked += got
        shortfall += per_q - len(got)
    if shortfall:  # redistribute over the questions that still have room
        seen = {(r["source"], r["video"], r["question"]) for r in picked}
        rest = [r for r in _rows(pool)
                if (r["source"], r["video"], r["question"]) not in seen]
        picked += take(rest, shortfall, ["question", "folder"], rng, used=budget)
    return picked


# ---------------------------------------------------------------------------
# reporting / writing
# ---------------------------------------------------------------------------


def table(name: str, rows: list[dict], by: list[str]) -> None:
    if not rows:
        print(f"\n{name}: EMPTY")
        return
    df = pl.DataFrame(rows)
    print(f"\n=== {name} ({len(rows)} items, {df['video'].n_unique()} clips, "
          f"{df['scenario_id'].n_unique()} scenarios) ===")
    agg = (df.group_by(by)
             .agg(pl.len().alias("n"),
                  pl.col("v4o_correct").mean().round(3).alias("v4o_acc"),
                  pl.col("scenario_id").n_unique().alias("scenarios"))
             .sort(by))
    with pl.Config(tbl_rows=40, fmt_str_lengths=52, tbl_hide_dataframe_shape=True):
        print(agg)
    per_scen = collections.Counter(r["scenario_id"] for r in rows)
    per_qs = collections.Counter((r["question"], r["scenario_id"]) for r in rows)
    print(f"    items per scenario:            max {max(per_scen.values())}, "
          f"mean {len(rows)/len(per_scen):.2f}")
    print(f"    items per (question, scenario): max {max(per_qs.values())} (cap "
          f"{MAX_PER_SCENARIO})")


def write_subset(rows: list[dict], prompts: dict, path: Path) -> list[dict]:
    items = []
    missing = []
    for r in rows:
        item = prompts.get((r["source"], r["video"], r["question"]))
        if item is None:
            missing.append((r["source"], r["video"], r["question"]))
            continue
        items.append(item)
    if missing:
        raise SystemExit(f"{len(missing)} selected items are absent from "
                         f"Questions_raw.json, e.g. {missing[:3]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"wrote {len(items):4d} items -> {path}  ({path.stat().st_size/1024:.0f} KiB)")
    return items


def gt_key(source: str, video: str, question: str) -> str:
    return f"{source}/{video}||{question}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--final-dataset", type=Path, default=DEFAULT_FINAL_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-realgen", type=int, default=300)
    ap.add_argument("--n-artifacts", type=int, default=300)
    ap.add_argument("--n-motion", type=int, default=300)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prompts = load_prompts(args.dataset_dir)
    df = load_candidates(args.parquet, args.final_dataset)

    n_all = df.height
    df = df.filter(~pl.col("collides"))
    print(f"v4o rows: {n_all} -> {df.height} after dropping "
          f"{n_all - df.height} basename-colliding items")

    forensic = build_forensic(df, rng, args.n_realgen, args.n_artifacts)
    motion = build_motion(df, rng, args.n_motion)

    table("cf_forensic", forensic, ["question", "source"])
    table("cf_motion", motion, ["question", "folder"])

    args.out.mkdir(parents=True, exist_ok=True)
    write_subset(forensic, prompts, args.out / "cf_forensic.json")
    write_subset(motion, prompts, args.out / "cf_motion.json")

    gt = {}
    for subset, rows in (("forensic", forensic), ("motion", motion)):
        for r in rows:
            item = prompts[(r["source"], r["video"], r["question"])]
            gt[gt_key(r["source"], r["video"], r["question"])] = {
                "subset": subset,
                "source": r["source"],
                "video": r["video"],
                "question": r["question"],
                "prompt": next(c for c in item["content"] if c["type"] == "text")["text"],
                "video_abs": next(c for c in item["content"] if c["type"] == "video")["video"],
                "ground_truth": r["ground_truth"],
                "category": r["category"],
                "folder": r["folder"],
                "scenario_id": r["scenario_id"],
                "v4o_correct": bool(r["v4o_correct"]),
                "v4o_answer": r["vlm_answer"],
            }
    meta = {
        "seed": args.seed,
        "max_per_question_scenario": MAX_PER_SCENARIO,
        "excluded_basename_collisions": n_all - df.height,
        "n_forensic": len(forensic),
        "n_motion": len(motion),
        "parquet": str(args.parquet),
    }
    out_gt = args.out / "cf_subsets_gt.json"
    out_gt.write_text(json.dumps({"meta": meta, "items": gt}, indent=2, ensure_ascii=False))
    print(f"wrote {len(gt):4d} keys  -> {out_gt}  ({out_gt.stat().st_size/1024:.0f} KiB)")

    if len(gt) != len(forensic) + len(motion):
        print(f"WARNING: {len(forensic) + len(motion) - len(gt)} duplicate keys collapsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
