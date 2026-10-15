"""Build the *balanced* real-vs-generated benchmark extension (v2).

Why v2
------
Reviewer objection to the paper: the "Reality understanding" category of the
main benchmark contains **only generated clips**, so a degenerate "always say
Generated" policy tops that category, and the 100-clip ablation in
``build_realgen_dataset.py`` (v1) is far too small to say anything.

v2 fixes both problems:

* **balanced**   ``--n-real 1000`` real clips (CarCrash ``videos/Normal``) and
  ``--n-gen 1000`` generated clips, so chance = 50 % and the trivial constant
  baselines score exactly 50 % balanced accuracy.
* **nested pilot**  the message lists are ordered *pilot first*: the first
  ``2 * --pilot`` messages are the 500 real + 250 dd + 250 p1 pilot subset.  A
  ``*_pilot*.json`` copy of every question file is written as well, because
  none of the eval scripts has a ``--limit`` flag.
* **clustered**  at most ``--max-per-cluster`` clips share a ``scenario_id``
  (the weather / model-size re-renderings of one underlying scene), so the
  cluster bootstrap in ``src/analysis/stats.py`` has many small clusters
  instead of a few huge ones.
* **stratified**  cosmos-drive-dreams is sampled proportionally over
  ``category x weather``, cosmos-predict1 over ``weather x gen_size``.
* **collision-safe**  the agent resolves videos *by basename*
  (``src/agentic/main_multi_tools_v4.py`` builds ``_VIDEO_CACHE`` from an
  ``rglob`` over ``final_dataset``), and 190 generated basenames occur in more
  than one pool (dd / p1 / p2), so a tool can silently analyse the wrong file.
  Clips whose basename collides are hard-excluded when the remaining pool is
  still big enough, otherwise merely de-prioritised; every record carries a
  ``basename_collides`` flag either way.

Inputs
------
``results/fft/manifest.parquet`` (built by ``src/fft/manifest.py``): one row per
mp4 under ``final_dataset`` plus the 3000 CarCrash ``Normal`` clips, with
``clip_id, path, label, generator, folder, scenario_id, weather, gen_size,
in_benchmark, in_realgen_v1, ...``.  ``category`` / ``n_questions`` / ``error``
are used when present.

Outputs (all under ``--out``)
-----------------------------
=======================================  ====================================
``questions_realgen_v2.json``            baseline prompt (Feedback::: template)
``questions_realgen_v2_hints.json``      baseline + 3 reality hints
``questions_realgen_v2_agent.json``      bare question (agent v4o)
``questions_realgen_v2_agent_hints.json``bare question + hints (agent v4p)
``questions_realgen_v2_raw.json``        ``Questions_raw.json`` shape, for the
                                         GPT / Gemini batch runners
``questions_realgen_v2_pilot*.json``     the same five, truncated to the pilot
``groundtruth_realgen_v2.json``          one record per clip
``realgen_v2_real_clips.txt``            CarCrash files to upload to the PVC
``realgen_v2_summary.md``                stratum tables
=======================================  ====================================

Example
-------
    .venv/bin/python src/real_vs_generated/build_realgen_v2.py \
        --n-real 1000 --n-gen 1000 --pilot 500 --seed 42 \
        --max-per-cluster 3 --out src/real_vs_generated/v2/
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "results" / "fft" / "manifest.parquet"
DEFAULT_OUT = REPO / "src" / "real_vs_generated" / "v2"

CLUSTER_ROOT = "/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset"

#: short generator tag -> on-disk / cluster source directory
SOURCE_OF_GENERATOR = {
    "dd": "cosmos-drive-dreams",
    "p1": "cosmos-predict1",
    "p2": "cosmos-predict2",
}

#: the two generated pools we draw from.  cosmos-predict2 (``p2``) is *not* part
#: of the labelled benchmark and is excluded.
GEN_SOURCES = ("dd", "p1")

# --------------------------------------------------------------------------
# Prompts -- copied verbatim from build_realgen_dataset.py (v1) so the two
# studies are directly comparable.  Do not reword: the downstream regexes in
# src/analysis/answer_analysis.py key off the "Feedback:::" / "Answer:" shape.
# --------------------------------------------------------------------------

QUESTION = "Is it a real video or generated ?"
REALITY_HINTS = [
    "Do the shadows cast by cars and objects match the direction of the light sources?",
    "Is the layout of the road normal ?",
    "Do road markings remain consistent in width and spacing?",
]

GENERATED_PROMPT = """You will be given a video and a question couple.
    The question is here to evaluate your performances.
    Your task is to analyze the video and then answer by Real or Generated to the question.

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (your rationale for the question, as a text)
    Answer: (your answer, as a Real or a Generated)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the question.

    Question: {question}

    Provide your feedback."""

GENERATED_PROMPT_HINTS = """You will be given a video and a main question. To ensure accuracy, you are also provided with a set of "Hint Questions" that you must answer first.

    Your task is to analyze the video and follow this two-step reasoning process:
    1. Answer the Hint Questions to gather objective evidence from the video.
    2. Use those answers to determine the final answer to the Main Question which should either be Real or Generated.

    Hint Questions:
    {secondary_questions}

    Provide your feedback as follows:

    Feedback:::
    Evaluation: (First, write the answer to each Hint Question. Then, explain your reasoning for the Main Question based on those facts.)
    Answer: (your answer, as a Real or a Generated)

    You MUST provide values for 'Evaluation:' and 'Answer:' in your answer.

    Now here is the Main Question.

    Question: {question}

    Provide your feedback."""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--n-real", type=int, default=1000)
    p.add_argument("--n-gen", type=int, default=1000, help="total generated, split over dd / p1")
    p.add_argument("--pilot", type=int, default=500,
                   help="pilot size *per label*: --pilot real + --pilot generated")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-cluster", type=int, default=3,
                   help="max clips sharing one scenario_id (weather re-renderings)")
    p.add_argument("--min-per-cell", type=int, default=1,
                   help="minimum quota for a non-empty stratum")
    p.add_argument("--shortfall", choices=("topup", "relax"), default="topup",
                   help="what to do when a generated source cannot reach its half of "
                        "--n-gen under --max-per-cluster: 'topup' keeps the cluster cap "
                        "and takes the missing clips from the other source (default), "
                        "'relax' raises the cap for that source instead")
    p.add_argument("--cluster-root", type=str, default=CLUSTER_ROOT,
                   help="final_dataset path inside the RunAI pod (goes into `video`)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Manifest -> candidate pools
# --------------------------------------------------------------------------


def load_manifest(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found at {path} -- build it with `python -m src.fft.manifest`"
        )
    df = pl.read_parquet(path)
    if "error" in df.columns:
        n_before = df.height
        df = df.filter(pl.col("error").is_null())
        if df.height != n_before:
            print(f"[manifest] dropped {n_before - df.height} rows with a probe error")
    # `category` (benchmark label folder) may not exist yet -> fall back to `folder`
    if "category" not in df.columns:
        df = df.with_columns(pl.col("folder").alias("category"))
    else:
        df = df.with_columns(pl.col("category").fill_null(pl.col("folder")).alias("category"))
    return df


def add_basename_collides(df: pl.DataFrame) -> pl.DataFrame:
    """Flag clips whose *basename* is not unique under ``final_dataset``.

    The agent resolves a video by basename (``_VIDEO_CACHE[video_path.name]``),
    so a duplicated basename means its tools may open a different pool's clip.
    Counted over **every** generated row (dd + p1 + p2), i.e. the whole
    ``final_dataset`` tree the agent rglobs -- not just the benchmark subset.
    """
    df = df.with_columns(pl.col("path").str.split("/").list.last().alias("basename"))
    gen = df.filter(pl.col("generator") != "real")
    counts = gen.group_by("basename").agg(pl.len().alias("_nb"))
    df = df.join(counts, on="basename", how="left").with_columns(
        pl.col("_nb").fill_null(0).alias("_nb")
    )
    # a real clip is uploaded into final_dataset/real/, so it collides only if
    # its numeric basename already exists somewhere in the generated tree.
    return df.with_columns((pl.col("_nb") > 1).alias("basename_collides")).drop("_nb")


def stratum_of(rec: dict, generator: str) -> str:
    """Stratification key.

    * ``dd`` -- ``<category>/<weather>``: the four labelled benchmark folders
      times the eight weather re-renderings.  A handful of clips carry two
      labels (``aggressive_takeover|negative``); only the *first* is used as a
      stratum key, so they do not create singleton strata.  The full label is
      kept in the ground-truth record.
    * ``p1`` -- ``<weather>`` when the clip has one, else ``no-weather/<size>``
      (the 7b / 14b model-size variants carry no weather suffix), so the two
      model sizes are represented proportionally.
    """
    weather = rec["weather"] or "none"
    if generator == "dd":
        return f"{(rec['category'] or rec['folder']).split('|')[0]}/{weather}"
    if rec["weather"]:
        return rec["weather"]
    return f"no-weather/{rec['gen_size'] or 'base'}"


# --------------------------------------------------------------------------
# Quota allocation + constrained draw
# --------------------------------------------------------------------------


def capacity(items: list[dict], cap: int) -> int:
    """How many of ``items`` can be picked under a ``cap``-per-scenario limit."""
    per_scen = Counter(it["scenario_id"] for it in items)
    return sum(min(k, cap) for k in per_scen.values())


def allocate(sizes: dict[str, int], total: int, caps: dict[str, int],
             min_cell: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, clamped by ``caps``.

    Strata that hit their capacity give their surplus back and the remainder is
    re-spread over the strata that still have room, until nothing moves.
    """
    keys = sorted(sizes)
    quota = {k: 0 for k in keys}
    free = {k: caps[k] for k in keys}
    remaining = min(total, sum(caps.values()))

    # guarantee a floor for every non-empty stratum first
    for k in keys:
        take = min(min_cell, free[k], remaining)
        quota[k] += take
        free[k] -= take
        remaining -= take

    while remaining > 0:
        open_keys = [k for k in keys if free[k] > 0]
        if not open_keys:
            break
        weight_total = sum(sizes[k] for k in open_keys)
        if weight_total == 0:
            break
        exact = {k: remaining * sizes[k] / weight_total for k in open_keys}
        floors = {k: min(int(exact[k]), free[k]) for k in open_keys}
        given = sum(floors.values())
        # distribute the leftover by largest fractional part (ties by key)
        left = remaining - given
        order = sorted(open_keys, key=lambda k: (-(exact[k] - int(exact[k])), k))
        i = 0
        while left > 0 and i < len(order) * 4:
            k = order[i % len(order)]
            if floors[k] < free[k]:
                floors[k] += 1
                left -= 1
            i += 1
        if given + (remaining - left) == 0:
            break
        for k in open_keys:
            quota[k] += floors[k]
            free[k] -= floors[k]
        moved = sum(floors.values())
        remaining -= moved
        if moved == 0:
            break
    return quota


def _take_one(pool: list[dict], used: Counter, cap: int, taken: set[str],
              seen: set[str]) -> dict | None:
    """First still-eligible clip of ``pool`` (tier-0 first, then tier-1)."""
    for it in pool:
        if it["clip_id"] in taken:
            continue
        scen = it["scenario_id"]
        if scen in seen or used[scen] >= cap:
            continue
        return it
    return None


def _commit(it: dict, used: Counter, taken: set[str], seen: set[str]) -> None:
    used[it["scenario_id"]] += 1
    taken.add(it["clip_id"])
    seen.add(it["scenario_id"])


def draw(pool: list[dict], n: int, used: Counter, cap: int, taken: set[str]) -> list[dict]:
    """Draw ``n`` clips from one pool, at most one per scenario per pass.

    ``pool`` must already be ordered (tier 0 = non-colliding basenames first,
    shuffled within a tier).  Spreading over scenarios pass by pass maximises
    the number of distinct clusters, which is what the cluster bootstrap needs.
    """
    out: list[dict] = []
    while len(out) < n:
        seen: set[str] = set()
        progress = False
        while len(out) < n:
            it = _take_one(pool, used, cap, taken, seen)
            if it is None:
                break
            _commit(it, used, taken, seen)
            out.append(it)
            progress = True
        if not progress:
            break
    return out


def draw_round_robin(by_stratum: dict[str, list[dict]], quota: dict[str, int],
                     used: Counter, cap: int, taken: set[str]) -> list[dict]:
    """Fill the per-stratum quotas one clip at a time, cycling over the strata.

    The ``cap``-per-scenario budget is *shared* across strata (one scenario has
    one clip per weather), so filling stratum after stratum would spend the
    whole budget on the alphabetically first weathers and starve the last ones.
    Cycling spends it evenly.
    """
    picked: list[dict] = []
    remaining = {k: quota.get(k, 0) for k in by_stratum}
    order = sorted(by_stratum)
    while any(v > 0 for v in remaining.values()):
        seen: set[str] = set()  # one clip per scenario per cycle
        progress = False
        for k in order:
            if remaining[k] <= 0:
                continue
            it = _take_one(by_stratum[k], used, cap, taken, seen)
            if it is None:
                continue
            _commit(it, used, taken, seen)
            picked.append(it)
            remaining[k] -= 1
            progress = True
        if not progress:
            break
    return picked


def sample_source(items: list[dict], generator: str, target: int, cap: int,
                  min_cell: int, rng: random.Random) -> tuple[list[dict], dict]:
    """Stratified, cluster-capped, collision-avoiding sample of one pool."""
    non_colliding = [it for it in items if not it["basename_collides"]]
    hard_exclude = capacity(non_colliding, cap) >= target
    n_excluded = len(items) - len(non_colliding) if hard_exclude else 0
    pool_items = non_colliding if hard_exclude else items

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for it in pool_items:
        it["_tier"] = 1 if it["basename_collides"] else 0
        by_stratum[stratum_of(it, generator)].append(it)
    for k in by_stratum:
        by_stratum[k].sort(key=lambda it: it["clip_id"])
        rng.shuffle(by_stratum[k])
        by_stratum[k].sort(key=lambda it: it["_tier"])

    sizes = {k: len(v) for k, v in by_stratum.items()}
    caps = {k: capacity(v, cap) for k, v in by_stratum.items()}
    quota = allocate(sizes, target, caps, min_cell)

    used: Counter = Counter()
    taken: set[str] = set()
    picked = draw_round_robin(by_stratum, quota, used, cap, taken)

    # global top-up: the per-scenario cap is shared across strata (one scenario
    # spans several weathers), so a stratum quota can end up unfillable.
    if len(picked) < target:
        flat = [it for k in sorted(by_stratum) for it in by_stratum[k]]
        flat.sort(key=lambda it: it["_tier"])
        picked += draw(flat, target - len(picked), used, cap, taken)

    info = {
        "generator": generator,
        "target": target,
        "picked": len(picked),
        "pool": len(items),
        "pool_after_collision_filter": len(pool_items),
        "collisions_hard_excluded": n_excluded,
        "collisions_in_sample": sum(1 for it in picked if it["basename_collides"]),
        "capacity_under_cap": capacity(pool_items, cap),
        "n_scenarios": len({it["scenario_id"] for it in picked}),
        "cap": cap,
        "quota": quota,
        "sizes": sizes,
    }
    return picked, info


# --------------------------------------------------------------------------
# Record construction
# --------------------------------------------------------------------------


def make_record(rec: dict, cluster_root: str) -> dict:
    gen = rec["generator"]
    if gen == "real":
        source = "real"
        rel_video = f"real/{Path(rec['path']).name}"
        category = "real"
        ground_truth = "Real"
    else:
        source = SOURCE_OF_GENERATOR[gen]
        rel_video = f"{source}/{rec['folder']}/{Path(rec['path']).name}"
        category = rec["category"] or rec["folder"]
        ground_truth = "Generated"
    return {
        "video": f"{cluster_root}/{rel_video}",
        "rel_video": rel_video,
        "local_path": rec["path"],
        "clip_id": rec["clip_id"],
        "source": source,
        "generator": gen,
        "ground_truth": ground_truth,
        "folder": rec["folder"],
        "category": category,
        "primary_category": category.split("|")[0],
        "scenario_id": rec["scenario_id"],
        "weather": rec["weather"],
        "gen_size": rec["gen_size"],
        "in_pilot": False,
        "prev_study_v1": bool(rec.get("in_realgen_v1", False)),
        "basename_collides": bool(rec["basename_collides"]),
        "width": rec["width"],
        "height": rec["height"],
        "fps": rec["fps"],
        "codec": rec["codec"],
    }


def build_message(video: str, prompt: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "video", "video": video},
            {"type": "text", "text": prompt},
        ],
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def stratum_table(records: list[dict], rows: str, cols: str) -> str:
    """Markdown cross-tab of ``rows`` x ``cols`` over ``records``."""
    counts: dict = defaultdict(Counter)
    col_keys: set = set()
    for r in records:
        rk = "-" if r[rows] is None else str(r[rows])
        ck = "-" if r[cols] is None else str(r[cols])
        counts[rk][ck] += 1
        col_keys.add(ck)
    cks = sorted(col_keys)
    out = ["| " + rows + " | " + " | ".join(cks) + " | total |",
           "|" + "---|" * (len(cks) + 2)]
    for rk in sorted(counts):
        cells = [str(counts[rk][c]) for c in cks]
        out.append(f"| {rk} | " + " | ".join(cells) + f" | {sum(counts[rk].values())} |")
    tot = Counter()
    for rk in counts:
        tot.update(counts[rk])
    out.append("| **total** | " + " | ".join(str(tot[c]) for c in cks)
               + f" | {sum(tot.values())} |")
    return "\n".join(out)


def print_table(title: str, text: str) -> None:
    print(f"\n### {title}")
    print(text)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    df = load_manifest(args.manifest)
    df = add_basename_collides(df)

    real_rows = df.filter(pl.col("generator") == "real").sort("clip_id").to_dicts()
    print(f"[pool] real (CarCrash Normal)                 : {len(real_rows)}")

    # ---------------------------------------------------------------- generated
    gen_pools: dict[str, list[dict]] = {}
    for g in GEN_SOURCES:
        rows = (
            df.filter((pl.col("generator") == g) & pl.col("in_benchmark"))
            .sort("clip_id")
            .to_dicts()
        )
        gen_pools[g] = rows
        n_col = sum(1 for r in rows if r["basename_collides"])
        print(f"[pool] {SOURCE_OF_GENERATOR[g]:<22s} in_benchmark : {len(rows):5d} "
              f"({len({r['scenario_id'] for r in rows})} scenarios, "
              f"{n_col} colliding basenames, "
              f"cap-{args.max_per_cluster} capacity {capacity(rows, args.max_per_cluster)})")

    per_source_target = {g: args.n_gen // len(GEN_SOURCES) for g in GEN_SOURCES}
    per_source_target[GEN_SOURCES[-1]] += args.n_gen - sum(per_source_target.values())

    picked_gen: dict[str, list[dict]] = {}
    infos: dict[str, dict] = {}
    caps = {g: args.max_per_cluster for g in GEN_SOURCES}

    for g in GEN_SOURCES:
        picked, info = sample_source(
            gen_pools[g], g, per_source_target[g], caps[g], args.min_per_cell, rng
        )
        if len(picked) < per_source_target[g] and args.shortfall == "relax":
            cap, best = caps[g], len(picked)
            # stop as soon as a bigger cap stops buying clips: the pool itself
            # is exhausted (cosmos-predict1 has only 494 benchmark clips).
            while len(picked) < per_source_target[g] and cap < 64:
                trial, trial_info = sample_source(
                    gen_pools[g], g, per_source_target[g], cap + 1, args.min_per_cell, rng
                )
                if len(trial) <= best:
                    break
                cap += 1
                best = len(trial)
                picked, info = trial, trial_info
            caps[g] = cap
            effective = max(Counter(it["scenario_id"] for it in picked).values())
            print(f"[shortfall] relaxed --max-per-cluster to {cap} for {g} "
                  f"({len(picked)} clips, {effective} in the largest cluster)")
        picked_gen[g] = picked
        infos[g] = info

    # top-up: keep the cluster cap, move the deficit to whichever source still
    # has spare capacity.  (p1 only has ~494 benchmark clips / 434 under cap 3.)
    deficit = args.n_gen - sum(len(v) for v in picked_gen.values())
    topups: dict[str, int] = {}
    if deficit > 0:
        for g in GEN_SOURCES:
            if deficit <= 0:
                break
            spare = infos[g]["capacity_under_cap"] - len(picked_gen[g])
            if spare <= 0:
                continue
            extra = min(spare, deficit)
            new_target = len(picked_gen[g]) + extra
            picked, info = sample_source(
                gen_pools[g], g, new_target, caps[g], args.min_per_cell, rng
            )
            topups[g] = len(picked) - len(picked_gen[g])
            deficit -= topups[g]
            picked_gen[g] = picked
            infos[g] = info
        if topups:
            print("[shortfall] kept the cluster cap and topped up: "
                  + ", ".join(f"{g} +{v}" for g, v in topups.items()))
    if deficit > 0:
        print(f"[shortfall] WARNING: {deficit} generated clips short of --n-gen")

    # -------------------------------------------------------------------- real
    forced = [r for r in real_rows if r.get("in_realgen_v1")]
    rest = [r for r in real_rows if not r.get("in_realgen_v1")]
    n_extra = max(0, args.n_real - len(forced))
    if n_extra > len(rest):
        raise ValueError(f"only {len(rest) + len(forced)} real clips available")
    picked_real = forced + rng.sample(rest, n_extra)
    picked_real = picked_real[: args.n_real]

    # ------------------------------------------------------------ order + pilot
    # deterministic per-source order; the first N of each source form the pilot
    recs_by_source: dict[str, list[dict]] = {}
    order_rng = random.Random(args.seed + 1)
    for key, rows in [("real", picked_real)] + [(g, picked_gen[g]) for g in GEN_SOURCES]:
        recs = [make_record(r, args.cluster_root) for r in rows]
        recs.sort(key=lambda r: r["clip_id"])
        order_rng.shuffle(recs)
        recs_by_source[key] = recs

    pilot_real = min(args.pilot, len(recs_by_source["real"]))
    gen_total = sum(len(recs_by_source[g]) for g in GEN_SOURCES)
    pilot_gen_target = min(args.pilot, gen_total)
    # split the generated half of the pilot proportionally to the sampled sizes
    pilot_gen: dict[str, int] = {}
    acc = 0
    for i, g in enumerate(GEN_SOURCES):
        if i == len(GEN_SOURCES) - 1:
            pilot_gen[g] = min(pilot_gen_target - acc, len(recs_by_source[g]))
        else:
            pilot_gen[g] = min(
                round(pilot_gen_target * len(recs_by_source[g]) / gen_total),
                len(recs_by_source[g]),
            )
        acc += pilot_gen[g]

    for key, k in [("real", pilot_real)] + [(g, pilot_gen[g]) for g in GEN_SOURCES]:
        for r in recs_by_source[key][:k]:
            r["in_pilot"] = True

    all_records = [r for key in ["real", *GEN_SOURCES] for r in recs_by_source[key]]
    pilot_records = [r for r in all_records if r["in_pilot"]]
    other_records = [r for r in all_records if not r["in_pilot"]]
    mix_rng = random.Random(args.seed + 2)
    mix_rng.shuffle(pilot_records)
    mix_rng.shuffle(other_records)
    ordered = pilot_records + other_records  # pilot-first: truncation == pilot

    # ------------------------------------------------------------------ prompts
    formatted_hints = "\n".join(f"- {h}" for h in REALITY_HINTS)
    prompt_no_hint = GENERATED_PROMPT.format(question=QUESTION)
    prompt_with_hint = GENERATED_PROMPT_HINTS.format(
        question=QUESTION, secondary_questions=formatted_hints
    )
    agent_prompt = QUESTION
    agent_prompt_hints = (
        f"{QUESTION}\n\nHint Questions to consider as supporting evidence:\n{formatted_hints}"
    )

    variants = {
        "": prompt_no_hint,
        "_hints": prompt_with_hint,
        "_agent": agent_prompt,
        "_agent_hints": agent_prompt_hints,
        "_raw": QUESTION,  # Questions_raw.json shape (GPT / Gemini batch runners)
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[Path, int]] = []
    for suffix, prompt in variants.items():
        msgs = [build_message(r["video"], prompt) for r in ordered]
        p = out_dir / f"questions_realgen_v2{suffix}.json"
        p.write_text(json.dumps(msgs, indent=4, ensure_ascii=False))
        written.append((p, len(msgs)))
        pilot_msgs = msgs[: len(pilot_records)]
        pp = out_dir / f"questions_realgen_v2_pilot{suffix}.json"
        pp.write_text(json.dumps(pilot_msgs, indent=4, ensure_ascii=False))
        written.append((pp, len(pilot_msgs)))

    gt_path = out_dir / "groundtruth_realgen_v2.json"
    gt_path.write_text(json.dumps(ordered, indent=4, ensure_ascii=False))
    written.append((gt_path, len(ordered)))

    real_list = sorted(Path(r["local_path"]).name for r in ordered if r["generator"] == "real")
    real_txt = out_dir / "realgen_v2_real_clips.txt"
    real_txt.write_text("\n".join(real_list) + "\n")
    written.append((real_txt, len(real_list)))

    # ------------------------------------------------------------------ summary
    gen_records = [r for r in ordered if r["ground_truth"] == "Generated"]
    real_records = [r for r in ordered if r["ground_truth"] == "Real"]
    dd_records = [r for r in gen_records if r["generator"] == "dd"]
    p1_records = [r for r in gen_records if r["generator"] == "p1"]

    t_label = stratum_table(ordered, "ground_truth", "source")
    t_dd = stratum_table(dd_records, "category", "weather")
    t_p1 = stratum_table(p1_records, "gen_size", "weather")
    t_pilot = stratum_table(ordered, "source", "in_pilot")

    print(f"\nreal      : {len(real_records)}  "
          f"({sum(r['prev_study_v1'] for r in real_records)} carried over from v1)")
    print(f"generated : {len(gen_records)}  (dd {len(dd_records)}, p1 {len(p1_records)})")
    print(f"pilot     : {len(pilot_records)}  "
          f"(real {sum(1 for r in pilot_records if r['generator'] == 'real')}, "
          f"dd {sum(1 for r in pilot_records if r['generator'] == 'dd')}, "
          f"p1 {sum(1 for r in pilot_records if r['generator'] == 'p1')})")
    print(f"clusters  : {len({r['scenario_id'] for r in ordered})} distinct scenario_id "
          f"(max {max(Counter(r['scenario_id'] for r in ordered).values())} clips per cluster)")
    n_coll = sum(r["basename_collides"] for r in ordered)
    print(f"basename collisions kept in the sample: {n_coll}")
    for g in GEN_SOURCES:
        i = infos[g]
        print(f"  {g}: hard-excluded {i['collisions_hard_excluded']} colliding clips, "
              f"{i['collisions_in_sample']} kept, {i['n_scenarios']} scenarios used")

    print_table("label x source", t_label)
    print_table("cosmos-drive-dreams: category x weather", t_dd)
    print_table("cosmos-predict1: gen_size x weather", t_p1)
    print_table("pilot membership", t_pilot)

    md = [
        "# real/generated v2 — balanced benchmark extension",
        "",
        f"Built by `src/real_vs_generated/build_realgen_v2.py` "
        f"(seed {args.seed}, max {args.max_per_cluster} clips per `scenario_id`).",
        "",
        f"* **{len(real_records)} real** clips (CarCrash `videos/Normal`), "
        f"{sum(r['prev_study_v1'] for r in real_records)} of them carried over from the v1 study",
        f"* **{len(gen_records)} generated** clips "
        f"(cosmos-drive-dreams {len(dd_records)}, cosmos-predict1 {len(p1_records)}) "
        "— benchmark folders only, cosmos-predict2 excluded",
        f"* **{len(pilot_records)}** clips flagged `in_pilot` "
        f"(first {len(pilot_records)} messages of every question file)",
        f"* {len({r['scenario_id'] for r in ordered})} distinct scenario clusters",
        f"* {n_coll} clips whose basename is ambiguous under `final_dataset` "
        f"(`basename_collides`); "
        + ", ".join(f"{g}: {infos[g]['collisions_hard_excluded']} hard-excluded"
                    for g in GEN_SOURCES),
        "",
        "## label x source", "", t_label, "",
        "## cosmos-drive-dreams: category x weather", "", t_dd, "",
        "## cosmos-predict1: gen_size x weather", "", t_p1, "",
        "## pilot membership", "", t_pilot, "",
        "## sampling diagnostics", "",
        "```json",
        json.dumps({g: {k: v for k, v in infos[g].items() if k not in ("quota", "sizes")}
                    for g in GEN_SOURCES}, indent=2),
        "```",
        "",
        "## files", "",
    ]
    for p, n in written:
        md.append(f"* `{p.name}` — {n} entries")
    (out_dir / "realgen_v2_summary.md").write_text("\n".join(md) + "\n")

    print("\nWrote:")
    for p, n in written:
        print(f"  {p}  ({n})")
    print(f"  {out_dir / 'realgen_v2_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
