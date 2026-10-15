#!/usr/bin/env python3
"""Inter-annotator agreement for the human-annotation studies (paper revision E9-E11).

Three tasks, three input shapes:

``--task reliability`` (E9)
    Annotator 1 is the **original ground truth** (``manifests/reliability_key.json`` +
    ``manifests/reliability_ext_key.json``, or the two ``questions_cosmos-*.json`` files) --
    the user does not re-annotate.  Annotator 2 is the annotation app's
    ``output/questions_<dataset>.json``.  Join on (source, video, question).  Entries the app
    marked ``{"video": ..., "remove": 1}`` mean "annotator flagged the clip unusable"; they
    are reported separately and excluded from kappa.

    The second rater must do the 100 **core** clips and may optionally continue on the 200
    **extension** clips.  ``--key`` therefore takes one *or more* key files and defaults to
    whichever of ``reliability_key.json`` / ``reliability_ext_key.json`` exist; only the
    clips the rater actually completed are joined, and the report prints how far they got.

``--task structured-state`` (E11)
    Two app output directories/files, one per annotator, same schema.

``--task faithfulness`` (E10)
    Two filled copies of ``manifests/faithfulness_rubric.csv``; kappa is computed per rubric
    column, joined on ``item_id``.

Statistics: percent agreement, Cohen's kappa (nominal), linear-weighted kappa for ordinal
scales, Krippendorff's alpha (nominal + ordinal), all with item-level bootstrap 95 % CIs
(B=2000, seed 0), broken down per question / category / folder.  ``--selftest`` checks the
estimators against published values.

Usage
-----
    python src/annotation/agreement.py --selftest

    python src/annotation/agreement.py --task reliability \
        --key src/annotation/manifests/reliability_key.json \
              src/annotation/manifests/reliability_ext_key.json \
        --files ~/annotations_teammate/output \
        --annotators ground_truth teammate \
        --out results/annotation/agreement_reliability.json --export-disagreements

    python src/annotation/agreement.py --task structured-state \
        --files out_matthieu/ out_teammate/ --annotators matthieu teammate \
        --out results/annotation/agreement_structured_state.json

    python src/annotation/agreement.py --task faithfulness \
        --files rubric_matthieu.csv rubric_teammate.csv \
        --out results/annotation/agreement_faithfulness.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manifest import (  # noqa: E402
    BASE_DOWNLOAD_URL,
    BLACKLISTED_STEMS,
    DATASET_DIR,
    GT_FILES,
    MANIFEST_DIR,
    REALITY_QUESTION,
    APP_VIDEOS_AVAILABLE,
    clip_stem,
    load_app_download_urls,
    scenario_id,
    weather_of,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "annotation"

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0

# Questions whose answers are an ordered 1..3 scale -> linear-weighted kappa + ordinal alpha.
ORDINAL_ANSWER_SETS = ({"1", "2", "3"},)

FAITHFULNESS_RUBRIC_COLUMNS = ["cites_tool", "citation_matches_output", "conclusion_follows", "hallucination"]
FAITHFULNESS_ORDINAL_COLUMNS = {"conclusion_follows"}

ADJUDICATION_REASONS = ["ambiguous_clip", "label_error", "annotator_slip", "convention"]


# ======================================================================================
# Estimators
# ======================================================================================


def percent_agreement(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def _categories(pairs: list[tuple[str, str]], order: list[str] | None = None) -> list[str]:
    seen = {v for pair in pairs for v in pair}
    if order:
        ordered = [c for c in order if c in seen]
        ordered += sorted(seen - set(ordered))
        return ordered
    return sorted(seen)


def cohen_kappa(pairs: list[tuple[str, str]], order: list[str] | None = None,
                weights: str = "nominal") -> float | None:
    """Cohen's kappa; ``weights='linear'`` gives the linearly weighted variant.

    For two categories the linear weighting is the identity, so weighted == unweighted.
    """
    if not pairs:
        return None
    cats = _categories(pairs, order)
    k = len(cats)
    if k < 2:
        return None  # a single observed category -> kappa undefined
    index = {c: i for i, c in enumerate(cats)}
    n = len(pairs)

    observed = [[0.0] * k for _ in range(k)]
    for a, b in pairs:
        observed[index[a]][index[b]] += 1.0
    rows = [sum(r) / n for r in observed]
    cols = [sum(observed[i][j] for i in range(k)) / n for j in range(k)]

    def weight(i: int, j: int) -> float:
        if weights == "linear":
            return 1.0 - abs(i - j) / (k - 1)
        return 1.0 if i == j else 0.0

    p_o = sum(observed[i][j] / n * weight(i, j) for i in range(k) for j in range(k))
    p_e = sum(rows[i] * cols[j] * weight(i, j) for i in range(k) for j in range(k))
    if abs(1.0 - p_e) < 1e-12:
        return None
    return (p_o - p_e) / (1.0 - p_e)


def krippendorff_alpha(units: list[list[str]], level: str = "nominal",
                       order: list[str] | None = None) -> float | None:
    """Krippendorff's alpha from the coincidence-matrix formulation.

    ``units`` is one list of observed values per unit (missing values simply absent).
    ``level`` is "nominal" or "ordinal"; ordinal needs a value ``order`` (defaults to the
    sorted distinct values, which is right for "1"/"2"/"3" style scales).
    """
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    values = _categories([(v, v) for u in units for v in u], order)
    k = len(values)
    if k < 2:
        return None
    index = {v: i for i, v in enumerate(values)}

    # coincidence matrix: each pair inside a unit contributes 1/(m-1)
    coincidence = [[0.0] * k for _ in range(k)]
    for unit in units:
        m = len(unit)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                coincidence[index[unit[i]]][index[unit[j]]] += 1.0 / (m - 1)

    n_c = [sum(row) for row in coincidence]
    n = sum(n_c)
    if n <= 1:
        return None

    if level == "ordinal":
        # delta^2(c,k) = ( sum_{g=c..k} n_g - (n_c + n_k)/2 )^2
        def delta(c: int, d: int) -> float:
            lo, hi = (c, d) if c <= d else (d, c)
            total = sum(n_c[g] for g in range(lo, hi + 1))
            return (total - (n_c[c] + n_c[d]) / 2.0) ** 2
    elif level == "interval":
        numeric = [float(v) for v in values]

        def delta(c: int, d: int) -> float:
            return (numeric[c] - numeric[d]) ** 2
    else:
        def delta(c: int, d: int) -> float:
            return 0.0 if c == d else 1.0

    d_o = sum(coincidence[c][d] * delta(c, d) for c in range(k) for d in range(k)) / n
    d_e = sum(n_c[c] * n_c[d] * delta(c, d) for c in range(k) for d in range(k)) / (n * (n - 1))
    if abs(d_e) < 1e-12:
        return None
    return 1.0 - d_o / d_e


def bootstrap_ci(items: list, statistic, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> dict | None:
    """Percentile bootstrap CI over items (resampled with replacement)."""
    point = statistic(items)
    if point is None or len(items) < 2:
        return {"point": point, "lo": None, "hi": None, "b": 0}
    rng = random.Random(seed)
    n = len(items)
    draws = []
    for _ in range(b):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        value = statistic(sample)
        if value is not None and not math.isnan(value):
            draws.append(value)
    if len(draws) < b // 10:
        return {"point": point, "lo": None, "hi": None, "b": len(draws)}
    draws.sort()
    lo = draws[max(0, int(math.floor((alpha / 2) * len(draws))))]
    hi = draws[min(len(draws) - 1, int(math.ceil((1 - alpha / 2) * len(draws))) - 1)]
    return {"point": point, "lo": lo, "hi": hi, "b": len(draws)}


# ======================================================================================
# Self-test against published values
# ======================================================================================


def selftest() -> int:
    ok = True

    def check(name: str, got, want, tol=1e-3):
        nonlocal ok
        good = got is not None and abs(got - want) <= tol
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got!r}, expected {want}")

    print("Cohen's kappa (Wikipedia 2x2 example: 20/5/10/15, p_o=0.70, p_e=0.50)")
    pairs = [("Yes", "Yes")] * 20 + [("Yes", "No")] * 5 + [("No", "Yes")] * 10 + [("No", "No")] * 15
    check("kappa", cohen_kappa(pairs), 0.4)
    check("linear-weighted kappa == unweighted for 2 categories",
          cohen_kappa(pairs, weights="linear"), 0.4)

    print("\nLinear-weighted kappa (3x3 [[8,2,0],[1,6,3],[0,2,8]]; p_o=26/30, p_e=5/9)")
    matrix = [[8, 2, 0], [1, 6, 3], [0, 2, 8]]
    pairs3 = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            pairs3 += [(str(i + 1), str(j + 1))] * count
    check("weighted kappa", cohen_kappa(pairs3, order=["1", "2", "3"], weights="linear"), 0.7)

    print("\nKrippendorff's alpha (Krippendorff 2011 tutorial data: 3 coders x 15 units)")
    # The canonical worked example; '*' = missing.  Published values:
    #   nominal alpha = 0.691, ordinal alpha = 0.807, interval alpha = 0.811.
    coders = [
        "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3",
        "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *",
        "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4",
    ]
    grid = [row.split() for row in coders]
    units = [[row[u] for row in grid if row[u] != "*"] for u in range(15)]
    order = ["1", "2", "3", "4"]
    check("alpha (nominal)", krippendorff_alpha(units, "nominal", order), 0.691, tol=1e-3)
    check("alpha (ordinal)", krippendorff_alpha(units, "ordinal", order), 0.807, tol=1e-3)
    check("alpha (interval)", krippendorff_alpha(units, "interval", order), 0.811, tol=1e-3)

    print("\nDegenerate cases")
    same = [("Yes", "Yes")] * 10
    print(f"  [{'PASS' if cohen_kappa(same) is None else 'FAIL'}] single-category kappa -> None")
    print(f"  [{'PASS' if percent_agreement(same) == 1.0 else 'FAIL'}] single-category agreement -> 1.0")
    ok = ok and cohen_kappa(same) is None and percent_agreement(same) == 1.0

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ======================================================================================
# Loading annotations
# ======================================================================================


def _iter_json_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("questions_*.json")) or sorted(path.glob("*.json"))
    return [path]


DATASET_FROM_NAME = re.compile(r"questions_(?P<dataset>.+)\.json$")


def load_app_output(paths: list[Path]) -> tuple[dict, dict]:
    """Read app output files (GT schema) -> ({(source, video, question): answer}, removed)."""
    answers: dict[tuple[str, str, str], str] = {}
    removed: dict[tuple[str, str], bool] = {}
    for given in paths:
        for path in _iter_json_files(Path(given)):
            match = DATASET_FROM_NAME.search(path.name)
            source = match.group("dataset") if match else None
            payload = json.loads(path.read_text(encoding="utf-8"))
            for entry in payload:
                video = entry["video"]
                src = source or _guess_source(video)
                if entry.get("remove"):
                    removed[(src, video)] = True
                    continue
                for qa in entry.get("qa_pairs", []):
                    if qa.get("answer") in (None, ""):
                        continue
                    answers[(src, video, qa["question"])] = str(qa["answer"])
    return answers, removed


def _guess_source(video: str) -> str:
    return "cosmos-predict1" if video.split("/")[0] == "artifacts" else "cosmos-drive-dreams"


# Key-file stem -> the label used in the coverage line.  Anything else keeps its own stem.
KEY_LIST_LABELS = {"reliability": "core", "reliability_ext": "extension"}
DEFAULT_RELIABILITY_KEYS = ("reliability_key.json", "reliability_ext_key.json")


def key_list_label(key_path: Path) -> str:
    stem = Path(key_path).stem
    if stem.endswith("_key"):
        stem = stem[: -len("_key")]
    return KEY_LIST_LABELS.get(stem, stem)


def default_reliability_keys() -> list[Path]:
    """The core key plus the optional extension key, whichever of them exist."""
    return [MANIFEST_DIR / name for name in DEFAULT_RELIABILITY_KEYS if (MANIFEST_DIR / name).exists()]


def load_reference(key_paths: Path | list[Path] | None, dataset_dir: Path) -> tuple[dict, dict, dict]:
    """Annotator 1 = ground truth.  Returns (answers, meta, restrict_videos).

    ``key_paths`` may be a single key file or several (core + extension); they are merged,
    and every ``restrict`` entry records which list (``key_list``) the clip came from.
    """
    answers: dict[tuple[str, str, str], str] = {}
    meta: dict[tuple[str, str, str], dict] = {}
    restrict: dict[tuple[str, str], dict] = {}

    if key_paths is None:
        key_paths = []
    elif isinstance(key_paths, (str, Path)):
        key_paths = [Path(key_paths)]
    existing = [Path(p) for p in key_paths if Path(p).exists()]
    missing = [str(p) for p in key_paths if not Path(p).exists()]
    if missing:
        print(f"[warn] key file(s) not found, ignored: {', '.join(missing)}", file=sys.stderr)

    if existing:
        for key_path in existing:
            label = key_list_label(key_path)
            payload = json.loads(key_path.read_text(encoding="utf-8"))
            for video in payload["videos"]:
                src, vid = video["source"], video["video"]
                if (src, vid) in restrict:  # the lists are disjoint; first key file wins
                    print(f"[warn] {src}/{vid} listed in more than one key file", file=sys.stderr)
                    continue
                restrict[(src, vid)] = {
                    "folder": video["folder"],
                    "weather": video.get("weather", weather_of(vid)),
                    "scenario_id": video.get("scenario_id", scenario_id(src, vid)),
                    "stratum": video.get("stratum", "unknown"),
                    "key_list": label,
                }
                for qa in video.get("qa_pairs", []):
                    answers[(src, vid, qa["question"])] = str(qa["answer"])
                    meta[(src, vid, qa["question"])] = {
                        "type": qa.get("type", "Closed"),
                        "category": qa.get("category", ""),
                        "possible_answers": qa.get("possible_answers") or [],
                        **restrict[(src, vid)],
                    }
        return answers, meta, restrict

    # fall back to the full ground-truth files
    for source, filename in GT_FILES.items():
        path = dataset_dir / filename
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            if entry.get("remove") or not entry.get("qa_pairs"):
                continue
            video = entry["video"]
            if clip_stem(video) in BLACKLISTED_STEMS:
                continue
            info = {
                "folder": video.split("/")[0],
                "weather": weather_of(video),
                "scenario_id": scenario_id(source, video),
                "stratum": "unknown",
                "key_list": "all",
            }
            for qa in entry["qa_pairs"]:
                answers[(source, video, qa["question"])] = str(qa["answer"])
                meta[(source, video, qa["question"])] = {
                    "type": qa.get("type", "Closed"),
                    "category": qa.get("category", ""),
                    "possible_answers": qa.get("possible_answers") or [],
                    **info,
                }
    return answers, meta, restrict


def load_structured_key(key_path: Path | None) -> tuple[dict, dict]:
    """Structured-state key -> (restrict_videos, question -> metadata)."""
    restrict, qmeta = {}, {}
    if key_path is None or not key_path.exists():
        return restrict, qmeta
    payload = json.loads(key_path.read_text(encoding="utf-8"))
    for video in payload["videos"]:
        restrict[(video["source"], video["video"])] = {
            "folder": video["folder"],
            "weather": video.get("weather", ""),
            "scenario_id": video.get("scenario_id", ""),
            "stratum": "n/a",
        }
    return restrict, qmeta


# ======================================================================================
# Building the joined item table
# ======================================================================================


def build_items_reliability(reference: dict, meta: dict, second: dict, restrict: dict,
                            include_reality: bool) -> tuple[list[dict], dict]:
    items, stats = [], collections.Counter()
    keys = set(reference) & set(second) if restrict else set(reference) & set(second)
    for key in sorted(keys):
        source, video, question = key
        if restrict and (source, video) not in restrict:
            stats["out_of_sample"] += 1
            continue
        if not include_reality and question == REALITY_QUESTION:
            stats["reality_excluded"] += 1
            continue
        info = meta.get(key, {})
        items.append(
            {
                "source": source,
                "video": video,
                "question": question,
                "a1": reference[key],
                "a2": second[key],
                "category": info.get("category", ""),
                "folder": info.get("folder", video.split("/")[0]),
                "weather": info.get("weather", weather_of(video)),
                "type": info.get("type", "Closed"),
                "possible_answers": info.get("possible_answers", []),
                "stratum": info.get("stratum", "unknown"),
                "key_list": info.get("key_list", restrict.get((source, video), {}).get("key_list", "all")),
            }
        )
    stats["joined"] = len(items)
    stats["only_reference"] = len(
        [k for k in reference if k not in second and (not restrict or (k[0], k[1]) in restrict)]
    )
    stats["only_second"] = len(
        [k for k in second if k not in reference and (not restrict or (k[0], k[1]) in restrict)]
    )
    return items, dict(stats)


def clip_completion(restrict: dict, second: dict, removed: dict) -> dict:
    """How far the second rater got, per key list (core / extension) and overall.

    A clip counts as *annotated* as soon as the app wrote at least one answer for it, or the
    rater flagged it unusable with the remove button -- partial completion is expected.
    """
    touched = {(src, vid) for src, vid, _ in second} | set(removed)
    lists = collections.OrderedDict()
    for (src, vid), info in restrict.items():
        label = info.get("key_list", "all")
        row = lists.setdefault(label, {"total": 0, "annotated": 0, "removed": 0})
        row["total"] += 1
        if (src, vid) in touched:
            row["annotated"] += 1
        if (src, vid) in removed:
            row["removed"] += 1
    total = sum(r["total"] for r in lists.values())
    annotated = sum(r["annotated"] for r in lists.values())
    return {"total": total, "annotated": annotated, "lists": dict(lists)}


def completion_line(completion: dict) -> str:
    order = [name for name in ("core", "extension") if name in completion["lists"]]
    order += [name for name in completion["lists"] if name not in order]
    detail = ", ".join(
        f"{name}: {completion['lists'][name]['annotated']}/{completion['lists'][name]['total']}"
        for name in order
    )
    line = f"n clips annotated by the second rater: {completion['annotated']} of {completion['total']}"
    return f"{line} ({detail})" if detail else line


def build_items_structured(first: dict, second: dict, restrict: dict,
                           question_meta: dict) -> tuple[list[dict], dict]:
    items, stats = [], collections.Counter()
    for key in sorted(set(first) & set(second)):
        source, video, question = key
        if restrict and (source, video) not in restrict:
            stats["out_of_sample"] += 1
            continue
        info = restrict.get((source, video), {})
        items.append(
            {
                "source": source,
                "video": video,
                "question": question,
                "a1": first[key],
                "a2": second[key],
                "category": question_meta.get(question, {}).get("category", ""),
                "folder": info.get("folder", video.split("/")[0]),
                "weather": info.get("weather", weather_of(video)),
                "type": question_meta.get(question, {}).get("type", "Closed"),
                "possible_answers": question_meta.get(question, {}).get("possible_answers", []),
                "stratum": "n/a",
            }
        )
    stats["joined"] = len(items)
    stats["only_annotator1"] = len([k for k in first if k not in second])
    stats["only_annotator2"] = len([k for k in second if k not in first])
    return items, dict(stats)


def load_structured_question_meta(path: Path) -> dict:
    meta = {}
    if not path.exists():
        return meta
    for block in json.loads(path.read_text(encoding="utf-8")):
        for category, questions in block.items():
            for question in questions:
                meta[question["question"]] = {
                    "category": category,
                    "type": question.get("question_type", "Closed"),
                    "possible_answers": question.get("possible_answers") or [],
                }
    return meta


def build_items_faithfulness(paths: list[Path]) -> tuple[list[dict], dict]:
    if len(paths) != 2:
        raise SystemExit("--task faithfulness needs exactly two --files (one CSV per rater)")
    tables = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            tables.append({row["item_id"]: row for row in csv.DictReader(handle)})
    first, second = tables

    items, stats = [], collections.Counter()
    for item_id in sorted(set(first) & set(second)):
        for column in FAITHFULNESS_RUBRIC_COLUMNS:
            a1 = (first[item_id].get(column) or "").strip()
            a2 = (second[item_id].get(column) or "").strip()
            if not a1 or not a2:
                stats[f"blank_{column}"] += 1
                continue
            items.append(
                {
                    "source": "faithfulness",
                    "video": item_id,
                    "question": column,
                    "a1": a1,
                    "a2": a2,
                    "category": "Faithfulness",
                    "folder": "faithfulness",
                    "weather": "",
                    "type": "MCQ" if column in FAITHFULNESS_ORDINAL_COLUMNS else "Closed",
                    "possible_answers": ["1", "2", "3"] if column in FAITHFULNESS_ORDINAL_COLUMNS else [],
                    "stratum": "n/a",
                }
            )
    stats["joined"] = len(items)
    stats["only_annotator1"] = len(set(first) - set(second))
    stats["only_annotator2"] = len(set(second) - set(first))
    return items, dict(stats)


# ======================================================================================
# Scoring
# ======================================================================================


def is_ordinal(items: list[dict]) -> bool:
    values = {item["a1"] for item in items} | {item["a2"] for item in items}
    options = {tuple(item["possible_answers"]) for item in items}
    if len(options) == 1 and set(next(iter(options))) in ORDINAL_ANSWER_SETS:
        return True
    return any(values <= s for s in ORDINAL_ANSWER_SETS) and len(values) > 1


def value_order(items: list[dict]) -> list[str] | None:
    options = {tuple(item["possible_answers"]) for item in items if item["possible_answers"]}
    if len(options) == 1:
        return list(next(iter(options)))
    return None


def score(items: list[dict], bootstrap: bool = True) -> dict:
    if not items:
        return {"n": 0}
    order = value_order(items)
    ordinal = is_ordinal(items)

    def pairs_of(rows):
        return [(r["a1"], r["a2"]) for r in rows]

    def units_of(rows):
        return [[r["a1"], r["a2"]] for r in rows]

    stats = {
        "n": len(items),
        "n_videos": len({(i["source"], i["video"]) for i in items}),
        "ordinal": ordinal,
        "categories": sorted({i["a1"] for i in items} | {i["a2"] for i in items}),
    }

    estimators = {
        "percent_agreement": lambda rows: percent_agreement(pairs_of(rows)),
        "cohen_kappa": lambda rows: cohen_kappa(pairs_of(rows), order),
        "krippendorff_alpha_nominal": lambda rows: krippendorff_alpha(units_of(rows), "nominal", order),
    }
    if ordinal:
        estimators["cohen_kappa_linear_weighted"] = lambda rows: cohen_kappa(pairs_of(rows), order, "linear")
        estimators["krippendorff_alpha_ordinal"] = lambda rows: krippendorff_alpha(units_of(rows), "ordinal", order)

    for name, fn in estimators.items():
        stats[name] = bootstrap_ci(items, fn) if bootstrap else {"point": fn(items), "lo": None, "hi": None}
    return stats


def breakdown(items: list[dict], field: str, bootstrap: bool = True) -> dict:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for item in items:
        groups[item[field] or "(none)"].append(item)
    return {
        name: score(rows, bootstrap)
        for name, rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }


# ======================================================================================
# Reporting
# ======================================================================================


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        if value.get("point") is None:
            return "n/a"
        if value.get("lo") is None:
            return f"{value['point']:.3f}"
        return f"{value['point']:.3f} [{value['lo']:.3f}, {value['hi']:.3f}]"
    return f"{value:.3f}"


def text_report(result: dict) -> str:
    lines = []
    overall = result["overall"]
    lines.append(f"annotators: {result['annotators'][0]} vs {result['annotators'][1]}")
    if result.get("completion"):
        lines.append(completion_line(result["completion"]))
    lines.append(f"joined items: {overall['n']}  (clips: {overall.get('n_videos', 0)})")
    for name in ("percent_agreement", "cohen_kappa", "cohen_kappa_linear_weighted",
                 "krippendorff_alpha_nominal", "krippendorff_alpha_ordinal"):
        if name in overall:
            lines.append(f"  {name:<32} {fmt(overall[name])}")
    for section in ("per_question", "per_category", "per_folder", "per_list"):
        if section not in result:
            continue
        lines.append("")
        lines.append(section)
        lines.append(f"{'n':>5} {'agree':>8} {'kappa':>8}  key")
        for name, stats in result[section].items():
            lines.append(
                f"{stats['n']:>5} {fmt(stats['percent_agreement'].get('point')):>8} "
                f"{fmt(stats['cohen_kappa'].get('point')):>8}  {name[:70]}"
            )
    if result.get("removed_by_annotator2"):
        lines.append("")
        lines.append(f"clips annotator 2 flagged unusable (remove): {len(result['removed_by_annotator2'])}")
        for video in result["removed_by_annotator2"][:20]:
            lines.append(f"    {video}")
    if result.get("coverage"):
        lines.append("")
        lines.append(f"coverage: {result['coverage']}")
    if result.get("vs_adjudicated"):
        lines.append("")
        lines.append("vs adjudicated labels")
        for name, stats in result["vs_adjudicated"].items():
            lines.append(f"  {name:<20} n={stats['n']:<5} accuracy={fmt(stats['accuracy'])} "
                         f"kappa={fmt(stats['cohen_kappa'])}")
    return "\n".join(lines)


def latex_table(result: dict) -> str:
    overall = result["overall"]
    rows = [("Overall", overall)] + list(result["per_category"].items())
    lines = [
        "% generated by src/annotation/agreement.py",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Subset & $N$ & \\% agree & $\\kappa$ & $\\alpha$ \\\\",
        "\\midrule",
    ]
    for name, stats in rows:
        if not stats.get("n"):
            continue
        agree = stats["percent_agreement"].get("point")
        kappa = stats["cohen_kappa"]
        alpha = stats["krippendorff_alpha_nominal"]
        kappa_cell = "n/a" if kappa.get("point") is None else (
            f"{kappa['point']:.3f}" if kappa.get("lo") is None
            else f"{kappa['point']:.3f} \\tiny[{kappa['lo']:.2f}, {kappa['hi']:.2f}]"
        )
        alpha_cell = "n/a" if alpha.get("point") is None else f"{alpha['point']:.3f}"
        safe = str(name).replace("_", "\\_").replace("&", "\\&")
        lines.append(f"{safe} & {stats['n']} & {100 * agree:.1f} & {kappa_cell} & {alpha_cell} \\\\")
        if name == "Overall":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def markdown_disagreements(items: list[dict], annotators: tuple[str, str]) -> str:
    lines = [
        f"| clip | question | {annotators[0]} | {annotators[1]} | folder | category |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in items:
        question = item["question"].replace("|", "/")
        if len(question) > 90:
            question = question[:87] + "..."
        lines.append(
            f"| `{item['source']}/{item['video']}` | {question} | {item['a1']} | {item['a2']} "
            f"| {item['folder']} | {item['category']} |"
        )
    return "\n".join(lines)


# ======================================================================================
# Adjudication
# ======================================================================================


def load_adjudicated(path: Path) -> dict:
    """results/annotation/reliability_adjudicated.jsonl: {source, video, question, answer}."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        source = row.get("source") or _guess_source(row["video"])
        key = row.get("question") or row["question_prefix"]
        out[(source, row["video"], key)] = str(row["answer"])
    return out


def _adjudicated_lookup(adjudicated: dict, item: dict) -> str | None:
    """Exact key first, then prefix match (the adjudication CSV truncates questions to 90 chars)."""
    key = (item["source"], item["video"], item["question"])
    if key in adjudicated:
        return adjudicated[key]
    for (src, vid, q), ans in adjudicated.items():
        if src == item["source"] and vid == item["video"] and item["question"].startswith(q):
            return ans
    return None


def score_vs_adjudicated(items: list[dict], adjudicated: dict, annotators: tuple[str, str]) -> dict:
    out = {}
    for label, field in ((annotators[0], "a1"), (annotators[1], "a2")):
        pairs = []
        for item in items:
            adj = _adjudicated_lookup(adjudicated, item)
            if adj is None:
                adj = item["a1"] if item["a1"] == item["a2"] else None
            if adj is not None:
                pairs.append((item[field], adj))
        if not pairs:
            continue
        out[label] = {
            "n": len(pairs),
            "accuracy": percent_agreement(pairs),
            "cohen_kappa": cohen_kappa(pairs),
        }
    return out


# ======================================================================================
# CLI
# ======================================================================================


def main(argv: list[str] | None = None) -> int:
    global BOOTSTRAP_B  # noqa: PLW0603 -- CLI override of the module default
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["reliability", "structured-state", "faithfulness"],
                        default="reliability")
    parser.add_argument("--files", type=Path, nargs="+", default=[],
                        help="reliability: annotator-2 app output (dir or questions_*.json); "
                             "structured-state: two app outputs; faithfulness: two filled CSVs")
    parser.add_argument("--key", type=Path, nargs="+", default=None, metavar="KEY_JSON",
                        help="manifests/<task>_key.json (annotator 1 / sample restriction); "
                             "reliability accepts several (core + extension) and defaults to "
                             "whichever of reliability_key.json / reliability_ext_key.json exist")
    parser.add_argument("--annotators", nargs=2, default=None, metavar=("A1", "A2"))
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--include-reality", action="store_true",
                        help="keep the always-'Generated' reality question in the headline numbers")
    parser.add_argument("--export-disagreements", action="store_true")
    parser.add_argument("--adjudicated", type=Path,
                        default=RESULTS_DIR / "reliability_adjudicated.jsonl")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--app-videos-available", type=Path, default=APP_VIDEOS_AVAILABLE)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    BOOTSTRAP_B = args.bootstrap

    removed: dict = {}
    completion: dict | None = None
    if args.task == "reliability":
        annotators = tuple(args.annotators or ("ground_truth", "annotator2"))
        keys = list(args.key) if args.key else default_reliability_keys()
        given_names = {Path(k).name for k in keys}
        for path in default_reliability_keys():
            if path.name not in given_names:
                print(f"[warn] {path.name} exists but was not passed to --key -- any annotation of "
                      f"those clips will be ignored", file=sys.stderr)
        if not keys:
            print("[warn] no reliability key file found -- falling back to the full ground truth",
                  file=sys.stderr)
        reference, meta, restrict = load_reference(keys, args.dataset_dir)
        if not args.files:
            raise SystemExit("--files is required (the annotator-2 app output dir or json files)")
        second, removed = load_app_output(args.files)
        completion = clip_completion(restrict, second, removed) if restrict else None
        items, coverage = build_items_reliability(reference, meta, second, restrict, args.include_reality)
    elif args.task == "structured-state":
        annotators = tuple(args.annotators or ("annotator1", "annotator2"))
        if len(args.files) != 2:
            raise SystemExit("--task structured-state needs exactly two --files (one per annotator)")
        key = (args.key[0] if args.key else MANIFEST_DIR / "structured_state_key.json")
        restrict, _ = load_structured_key(key)
        question_meta = load_structured_question_meta(MANIFEST_DIR / "standard_questions_structured_state.json")
        first, removed_1 = load_app_output([args.files[0]])
        second, removed_2 = load_app_output([args.files[1]])
        removed = {**removed_1, **removed_2}
        items, coverage = build_items_structured(first, second, restrict, question_meta)
    else:
        annotators = tuple(args.annotators or ("annotator1", "annotator2"))
        items, coverage = build_items_faithfulness(args.files)

    if not items:
        raise SystemExit("no items joined -- check --files / --key")

    result = {
        "task": args.task,
        "annotators": list(annotators),
        "include_reality": args.include_reality,
        "bootstrap": {"b": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
        "coverage": coverage,
        "removed_by_annotator2": [f"{src}/{vid}" for src, vid in sorted(removed)],
        "overall": score(items),
        "per_question": breakdown(items, "question"),
        "per_category": breakdown(items, "category"),
        "per_folder": breakdown(items, "folder"),
    }
    if completion is not None:
        result["completion"] = completion
        if len(completion["lists"]) > 1:
            result["per_list"] = breakdown(items, "key_list")

    adjudicated = load_adjudicated(args.adjudicated)
    if adjudicated:
        result["vs_adjudicated"] = score_vs_adjudicated(items, adjudicated, annotators)

    disagreements = [item for item in items if item["a1"] != item["a2"]]
    result["n_disagreements"] = len(disagreements)

    out = args.out or (RESULTS_DIR / f"agreement_{args.task.replace('-', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    report = text_report(result)
    latex = latex_table(result)
    tex_path = out.with_suffix(".tex")
    tex_path.write_text(latex + "\n", encoding="utf-8")

    print(report)
    print()
    print(latex)
    print()
    print(f"wrote {out}")
    print(f"wrote {tex_path}")

    if args.export_disagreements and disagreements:
        md_path = out.parent / f"{out.stem}_disagreements.md"
        md_path.write_text(markdown_disagreements(disagreements, annotators) + "\n", encoding="utf-8")
        print(f"wrote {md_path}  ({len(disagreements)} rows)")

        url_map = load_app_download_urls(args.app_videos_available)
        seen, adjudication_list = set(), []
        for item in disagreements:
            pair = (item["source"], item["video"])
            if pair in seen:
                continue
            seen.add(pair)
            url = url_map.get(pair) or BASE_DOWNLOAD_URL.format(
                relative_path=re.sub(" ", "%20", f"{item['source']}/{item['video']}")
            )
            adjudication_list.append(
                {"download_url": url, "path": item["video"], "dataset": item["source"], "processed": 0}
            )
        app_path = MANIFEST_DIR / f"{args.task.replace('-', '_')}_adjudication_videos_available.json"
        app_path.write_text(json.dumps(adjudication_list, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {app_path}  ({len(adjudication_list)} clips for the joint adjudication pass)")
        print(f"  adjudication reasons to record: {', '.join(ADJUDICATION_REASONS)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
