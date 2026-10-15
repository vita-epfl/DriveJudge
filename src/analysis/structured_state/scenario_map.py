"""Recover which seed clip / frame window each benchmark scenario came from.

The problem
-----------
``cosmos-drive-dreams-toolkits/mapping.txt`` maps the *author's* scenario names
(``cross_red_026``, ``agg_tk_024``, ``cross_stop_007``, ...) to RDS-HQ seed
clip ids.  The benchmark uses a **different, renumbered** set of scenario names
in ``dataset/dataset_final/questions_cosmos-drive-dreams.json``
(``crossing_red_lights/cross_red_034_Snowy.mp4``, ...).  The renumbering
happened because every seed clip was rendered into *two* 121-frame control
videos (chunk 0 and chunk 1, see ``load_rds.chunk_frame_range``) and each chunk
became its own benchmark scenario.  Joining the two name spaces by string
equality would silently pair the wrong trajectories.

The evidence used here
----------------------
Cosmos-Transfer1 was driven by one caption per (control video, weather).  The
prompt-rewriting step left both halves on disk:

* ``rendering/generated/captions_rewritten/<key>.json`` -- a dict
  ``{weather: caption}`` per rendered control video, where ``<key>`` is the
  author's ``<scenario>_<chunk>`` name (or a raw ``<clip_id>_<chunk>`` for the
  "negative" renders);
* ``final_dataset/cosmos-drive-dreams/<folder>/captions/<scenario>_<weather>.txt``
  -- the caption actually used for each *benchmark* video.

Matching the second against the first by exact string equality identifies the
control video each benchmark clip was generated from.  Some captions are
duplicated across keys (identical base captions), so only matches that resolve
to a **single** key are counted, and the 8 weather variants of a scenario vote.

For the ``crossing_stop`` family the rewritten captions were not kept locally;
there the *base* captions in ``rendering/to_be_generated/captions/`` are used
instead, which match the benchmark's ``_Original`` caption exactly (60/60).

Output
------
``resolve()`` returns ``{(folder, benchmark_scenario): Resolution}`` with the
author scenario, the chunk index (may be ``None`` when the two chunks of a pair
cannot be told apart), the seed clip id from ``mapping.txt`` and a confidence
tag.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from load_rds import DEFAULT_TOOLKIT_ROOT, parse_mapping

DEFAULT_FINAL_DATASET = Path(
    os.environ.get(
        "FINAL_DATASET_ROOT",
        "/home/matthieu/Documents/Projet de master/final_dataset/cosmos-drive-dreams",
    )
)

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-")
_TWO_TAIL = re.compile(r"^(.*_[01])_0$")


@dataclass
class Resolution:
    folder: str
    benchmark_scenario: str
    author_scenario: str | None
    chunk: int | None
    clip_id: str | None
    confidence: str
    votes: int
    runner_up_votes: int
    n_weathers: int
    caption_key: str | None = None
    positional_key: str | None = None
    note: str = ""


#: which ``rendering/`` directory holds the control videos of each benchmark folder
RENDER_DIRS = {
    "aggressive_takeover": "generated/aggressive_takeover",
    "crossing_red_lights": "generated/crossing_red_lights",
    "wrong_way_driving": "generated/wrong_way_driving",
    "driving_on_wrong_lane": "generated/driving_on_wrong_lane",
    "aggressive_line_change": "generated/aggressive_line_change",
    "car_appearance": "generated/car_appearance",
    "drunk_driving": "generated/drunk_driving",
    "negative": "generated/negative",
    "crossing_stop": "to_be_generated/hdmap_to_be_processed/ftheta_camera_front_wide_120fov",
}


def _strip_copy_suffix(key: str) -> str:
    """``agg_tk_023_0_0`` -> ``agg_tk_023_0`` (the trailing ``_0`` is a copy).

    Verified by md5: ``<name>_0.mp4`` and ``<name>_0_0.mp4`` are byte-identical
    files in ``rendering/``.
    """
    m = _TWO_TAIL.match(key)
    return m.group(1) if m else key


def _split_chunk(key: str) -> tuple[str, int | None]:
    key = _strip_copy_suffix(key)
    m = re.match(r"^(.*)_([01])$", key)
    if m:
        return m.group(1), int(m.group(2))
    return key, None


def _build_caption_index(toolkit_root: Path) -> dict[tuple[str, str], set[str]]:
    """``(caption_text, weather) -> {control-video key}`` from the rewrites."""
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    root = toolkit_root / "rendering" / "generated" / "captions_rewritten"
    if root.is_dir():
        for f in sorted(root.iterdir()):
            if f.suffix != ".json":
                continue
            try:
                payload = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            for weather, text in payload.items():
                index[(text.strip(), weather)].add(f.stem)
    return index


def _build_base_caption_index(toolkit_root: Path) -> dict[str, set[str]]:
    """``caption_text -> {control-video key}`` from the un-rewritten captions.

    Used for ``crossing_stop``, whose rewritten captions are not on disk; the
    benchmark's ``Original`` weather reuses the base caption verbatim.
    """
    index: dict[str, set[str]] = defaultdict(set)
    for sub in ("to_be_generated", "generated"):
        root = toolkit_root / "rendering" / sub / "captions"
        if not root.is_dir():
            continue
        for f in sorted(root.iterdir()):
            if f.suffix != ".txt":
                continue
            index[f.read_text().strip()].add(f.stem)
    return index


def _positional_keys(toolkit_root: Path, folder: str) -> list[str]:
    """Ordered control videos of ``folder``, de-duplicated.

    ``<name>_0.mp4`` and ``<name>_0_0.mp4`` are byte-identical copies (md5
    checked), so the trailing ``_0`` is stripped before de-duplication.
    """
    sub = RENDER_DIRS.get(folder)
    if sub is None:
        return []
    d = toolkit_root / "rendering" / sub
    if not d.is_dir():
        return []
    seen: list[str] = []
    for f in sorted(d.iterdir()):
        if f.suffix != ".mp4":
            continue
        k = _strip_copy_suffix(f.stem)
        if k not in seen:
            seen.append(k)
    return sorted(seen)


def resolve(
    toolkit_root: Path | str = DEFAULT_TOOLKIT_ROOT,
    final_dataset: Path | str = DEFAULT_FINAL_DATASET,
) -> dict[tuple[str, str], Resolution]:
    toolkit_root = Path(toolkit_root)
    final_dataset = Path(final_dataset)
    rewritten = _build_caption_index(toolkit_root)
    base = _build_base_caption_index(toolkit_root)
    mapping = parse_mapping(toolkit_root)

    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    pair_votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    n_weathers: Counter = Counter()

    folders = sorted(p.name for p in final_dataset.iterdir() if p.is_dir())
    for folder in folders:
        cap_dir = final_dataset / folder / "captions"
        if not cap_dir.is_dir():
            continue
        for f in sorted(cap_dir.iterdir()):
            if f.suffix != ".txt":
                continue
            scenario, _, weather = f.stem.rpartition("_")
            if not scenario:
                continue
            text = f.read_text().strip()
            key = (folder, scenario)
            n_weathers[key] += 1
            hits = rewritten.get((text, weather))
            if not hits and weather == "Original":
                hits = base.get(text)
            if not hits:
                continue
            if len(hits) == 1:
                votes[key][next(iter(hits))] += 1
            else:
                # a caption shared by both halves of one seed clip still pins
                # down *which* seed clip the benchmark video came from
                authors = {_split_chunk(h)[0] for h in hits}
                if len(authors) == 1:
                    pair_votes[key][next(iter(authors))] += 1

    # ---- turn votes into resolutions -------------------------------------
    out: dict[tuple[str, str], Resolution] = {}
    for folder in folders:
        cap_dir = final_dataset / folder / "captions"
        if not cap_dir.is_dir():
            continue
        # the scenario list comes from the *videos*, not the captions: a few
        # stray caption files have no matching benchmark clip
        scenarios = sorted(
            {f.stem.rpartition("_")[0] for f in (final_dataset / folder).iterdir() if f.suffix == ".mp4"}
        )
        pos_keys = _positional_keys(toolkit_root, folder)
        positional = {}
        if len(pos_keys) == len(scenarios):
            positional = dict(zip(scenarios, pos_keys))

        for scenario in scenarios:
            key = (folder, scenario)
            ranked = votes.get(key, Counter()).most_common()
            cap_key, top_v = (ranked[0] if ranked else (None, 0))
            second_key, second_v = (ranked[1] if len(ranked) > 1 else (None, 0))
            pos_key = positional.get(scenario)

            cap_author, cap_chunk = _split_chunk(cap_key) if cap_key else (None, None)
            if cap_key and second_key and second_v == top_v:
                a2, _ = _split_chunk(second_key)
                cap_chunk = cap_chunk if a2 == cap_author else None
                if a2 != cap_author:
                    cap_author = None  # genuinely undecided
            pos_author, pos_chunk = _split_chunk(pos_key) if pos_key else (None, None)
            pair_author = pair_votes.get(key, Counter())
            pair_author = pair_author.most_common(1)[0][0] if pair_author else None

            author = chunk = None
            note = ""
            # 1. caption vote and positional rule agree on the seed clip
            if cap_author and pos_author and cap_author == pos_author:
                author = cap_author
                if cap_chunk is not None and cap_chunk == pos_chunk:
                    chunk, conf = cap_chunk, "clip+chunk (caption vote == render order)"
                else:
                    chunk, conf = pos_chunk, "clip certain, chunk from render order (caption vote disagrees)"
            # 2. shared caption pins the seed clip, render order gives the chunk
            elif pair_author and pos_author and pair_author == pos_author:
                author, chunk = pair_author, pos_chunk
                conf = "clip certain (shared caption), chunk from render order"
            # 3. only one source of evidence
            elif cap_author and not pos_author:
                author, chunk, conf = cap_author, cap_chunk, "caption vote only"
            elif pos_author and not cap_author and not pair_author:
                author, chunk, conf = pos_author, pos_chunk, "render order only"
            elif cap_author and pos_author:
                conf = "conflict (caption vote != render order)"
                note = f"caption={cap_key} render_order={pos_key}"
            else:
                conf = "unresolved"

            clip_id = None
            if author is not None:
                if _UUID.match(author):
                    clip_id = author
                    note = (note + " raw clip id => negative render (original poses)").strip()
                elif author in mapping:
                    clip_id = mapping[author]["clip_id"]
                    if mapping[author]["note"]:
                        note = (note + " " + mapping[author]["note"]).strip()
                else:
                    note = (note + " not in mapping.txt").strip()

            out[key] = Resolution(
                folder=folder,
                benchmark_scenario=scenario,
                author_scenario=author,
                chunk=chunk,
                clip_id=clip_id,
                confidence=conf,
                votes=top_v,
                runner_up_votes=second_v,
                n_weathers=n_weathers.get(key, 0),
                caption_key=cap_key,
                positional_key=pos_key,
                note=note,
            )
    return out


def to_json(resolutions: dict[tuple[str, str], Resolution]) -> list[dict]:
    return [asdict(r) for _, r in sorted(resolutions.items())]


if __name__ == "__main__":  # pragma: no cover - quick inspection helper
    res = resolve()
    by_conf = Counter(r.confidence for r in res.values())
    print(by_conf)
    mapped = [r for r in res.values() if r.clip_id]
    print(f"{len(mapped)}/{len(res)} benchmark scenarios linked to a seed clip")
    for r in sorted(res.values(), key=lambda r: (r.folder, r.benchmark_scenario)):
        if r.clip_id and not _UUID.match(r.author_scenario or ""):
            print(
                f"{r.folder:22s} {r.benchmark_scenario:24s} <- {r.author_scenario:24s} "
                f"chunk={r.chunk} {r.confidence:32s} {r.votes}/{r.runner_up_votes} {r.clip_id[:20]}"
            )
