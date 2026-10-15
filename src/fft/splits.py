"""
Deterministic calibration / test splits for the FFT real-vs-generated study.

The agent's FFT tool declares *Generated* when the mean anisotropy of a clip is
>= 0.048.  That threshold was eyeballed on 18 real + 18 generated clips
(``in_fft_calib18``), which is both tiny and -- because the real pool is mpeg4
1280x720@10fps while the generated pools are h264 1280x704@24fps -- confounded
with the container fingerprint.  Answering the reviewers therefore needs a
*clean* split of the 5510 manifest clips into a fold the threshold may be fitted
on and folds it is only ever tested on.

Folds
-----
``calib18``    the original 18+18 clips.  Excluded from every test fold so that
               the published 0.048 is never evaluated on its own training data.
``e6_test``    the E6 balanced real-vs-generated set (1000 real / 1000
               generated, ``src/real_vs_generated/v2/groundtruth_realgen_v2.json``)
               minus any ``calib18`` member.  This is the headline test set.
``calib``      70% (by ``scenario_id`` group) of the calibration pool.
``holdout``    the complementary 30%.
``e6_sibling`` benchmark-labelled generated clips that are *weather siblings* of
               an ``e6_test`` / ``calib18`` scenario.  Neither trainable (they
               would leak into ``e6_test``) nor an independent test fold; kept
               visible rather than silently dropped.
``extra_gen``  test-only generated clips carrying no benchmark label: the
               unlabelled cosmos-drive-dreams / -predict1 folders.
``unused``     anything the rules above do not reach (empty for the default
               manifest; non-empty only if ``--include-crash`` was used when
               building it).

Grouping, and why the calibration pool is *not* the benchmark
------------------------------------------------------------
Splitting is by ``scenario_id`` (all weather / model-size re-renderings of one
underlying scene), never by clip: the eight weather variants of a scene are near
duplicates in frequency space, so a clip-level split would leak.  Real clips are
singleton groups.  ``sklearn.GroupShuffleSplit`` is run *separately* on the real
and on the generated part of the pool with the same 30% test size, which keeps
the real:generated ratio of ``calib`` and ``holdout`` matched to the pool's.

The obvious pool -- "the benchmark clips E6 did not take" -- cannot be used:
E6 v2 sampled 214 of the 250 cosmos-drive-dreams scenarios and 202 of the 212
cosmos-predict1 scenarios, so only **4** benchmark clips (3 scenarios) sit in a
scenario E6 never touches.  Calibrating on the other 584 would train on weather
re-renderings of the very scenes the headline test set is made of.  The default
pool (``--pool leakfree``) is therefore *every* dd/p1 clip whose scenario is
untouched by ``e6_test``/``calib18`` -- 275 generated clips over 43 scenarios,
which in practice means the unlabelled dd folders (``wrong_way_driving``,
``drunk_driving``, ``driving_on_wrong_lane``, ...) plus 18 p1 clips -- together
with the 1989 unused real clips.  ``--pool benchmark`` reproduces the naive
"remaining benchmark" pool instead; it prints the leak it creates and downgrades
the disjointness assertions to warnings, and exists only so the two can be
compared.

The generalisation-to-another-generator question is answered by the
leave-one-generator-out analysis in ``calibrate.py`` (train on dd + real, test on
p1 + real, and back), not by a held-out pool.

Usage
-----
    .venv/bin/python src/fft/splits.py
    .venv/bin/python src/fft/splits.py --out results/fft/splits.parquet --seed 0

Outputs ``results/fft/splits.parquet`` (``clip_id, split, in_e6, in_calib18,
label, generator, scenario_id``) and ``results/fft/splits_summary.json``.
The run is idempotent: same manifest + same seed -> byte-identical folds.

It also cross-checks ``results/fft/e6_clip_ids.txt`` -- the id list the n10
re-encode is restricted to -- against the E6 json, and writes it if absent.  The
two must agree: ``calibrate.py --mode e6cv`` cross-validates the threshold
inside exactly those clips, so a drift between the lists would put calibration
clips into the test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft.common import (  # noqa: E402
    DEFAULT_MANIFEST,
    GENERATED_SOURCES,
    REPO,
    RESULTS_FFT_DIR,
    rel_to_repo,
)

REALGEN_V2_JSON = REPO / "src" / "real_vs_generated" / "v2" / "groundtruth_realgen_v2.json"

DEFAULT_SPLITS = RESULTS_FFT_DIR / "splits.parquet"
DEFAULT_SUMMARY = RESULTS_FFT_DIR / "splits_summary.json"

#: the id list the n10 re-encode / feature extraction is restricted to.  The
#: normalised condition only exists for these 2000 clips, so ``calibrate.py
#: --condition n10`` cross-validates inside them instead of using ``calib``.
E6_IDS_TXT = RESULTS_FFT_DIR / "e6_clip_ids.txt"

#: fold names in report order.
SPLIT_ORDER = ["calib18", "calib", "holdout", "e6_test", "e6_sibling", "extra_gen", "unused"]

#: folds the threshold may never be fitted on.
TEST_SPLITS = ["holdout", "e6_test", "extra_gen"]

#: pool definitions for calib/holdout (see the module docstring).
POOL_MODES = ("leakfree", "benchmark")

TEST_FRACTION = 0.30


# --------------------------------------------------------------------------
# E6 membership
# --------------------------------------------------------------------------


def clip_id_from_rel_video(rel: str) -> str:
    """``cosmos-predict1/artifacts/x.mp4`` -> ``p1/artifacts/x``; ``real/y.mp4`` -> ``real/y``.

    Mirrors :func:`fft.common.make_clip_id` without needing the manifest: the
    leading path segment is the on-disk source directory (or ``real``), the
    trailing segment the file, and whatever sits in between is the folder.
    """
    parts = [p for p in rel.split("/") if p]
    fname = parts[-1]
    stem = fname[:-4] if fname.lower().endswith(".mp4") else fname
    gen = GENERATED_SOURCES.get(parts[0], parts[0])
    folder = "/".join(parts[1:-1])
    return f"{gen}/{folder}/{stem}" if folder else f"{gen}/{stem}"


def load_e6_ids(path: Path = REALGEN_V2_JSON) -> tuple[list[str], dict]:
    """clip_ids of the E6 v2 balanced set, derived from ``rel_video``.

    The JSON already carries a ``clip_id`` field; it is recomputed from
    ``rel_video`` anyway and the two are cross-checked, because the parquet is
    keyed on the manifest's own convention and a silent drift there would
    quietly turn the test set into a training set.
    """
    if not path.exists():
        raise SystemExit(f"E6 ground truth not found: {path}")
    records = json.loads(path.read_text())
    ids, disagree = [], 0
    for rec in records:
        derived = clip_id_from_rel_video(rec["rel_video"])
        stated = rec.get("clip_id")
        if stated and stated != derived:
            disagree += 1
        ids.append(derived)
    info = {
        "records": len(records),
        "unique_clip_ids": len(set(ids)),
        "clip_id_field_disagreements": disagree,
        "n_real": sum(1 for r in records if r.get("generator") == "real"),
        "n_generated": sum(1 for r in records if r.get("generator") != "real"),
    }
    return ids, info


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def group_split(groups: np.ndarray, test_size: float, seed: int) -> np.ndarray:
    """Boolean mask (True = test side) from a single ``GroupShuffleSplit`` draw."""
    from sklearn.model_selection import GroupShuffleSplit

    n = len(groups)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if len(np.unique(groups)) < 2:
        # Cannot split one group; keep everything on the train (calib) side.
        return np.zeros(n, dtype=bool)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(gss.split(np.zeros(n), groups=groups))
    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    return mask


def assign(man: pl.DataFrame, e6_ids: list[str], seed: int, pool_mode: str = "leakfree") -> pl.DataFrame:
    """Add the ``split`` column.  Rules are applied in priority order."""
    if pool_mode not in POOL_MODES:
        raise SystemExit(f"--pool must be one of {POOL_MODES}, got {pool_mode!r}")
    e6 = set(e6_ids)
    df = man.with_columns(
        pl.col("clip_id").is_in(list(e6)).alias("in_e6"),
        pl.col("in_fft_calib18").alias("in_calib18"),
    )

    split = np.full(df.height, "unused", dtype=object)
    is_calib18 = df["in_calib18"].to_numpy()
    is_e6 = df["in_e6"].to_numpy()
    label = df["label"].to_numpy()
    generator = df["generator"].to_numpy()
    in_bench = df["in_benchmark"].to_numpy()
    scenarios = df["scenario_id"].to_numpy()

    # 1. calib18 wins over everything: it must not appear in a test fold.
    split[is_calib18] = "calib18"

    # 2. the E6 balanced set.
    split[is_e6 & (split == "unused")] = "e6_test"

    # Scenarios claimed by a test fold that must never be trained on.
    reserved = set(scenarios[is_e6 | is_calib18].tolist())
    is_reserved = np.array([s in reserved for s in scenarios])

    # 3. calibration pool -> calib / holdout, grouped by scenario, real and
    #    generated stratified separately (see the module docstring for why the
    #    default pool is not "the remaining benchmark").
    gen_pool = (generator == "dd") | (generator == "p1")
    if pool_mode == "leakfree":
        pool = (split == "unused") & (gen_pool & ~is_reserved | (label == "real"))
    else:
        pool = (split == "unused") & (gen_pool & in_bench | (label == "real"))
    for stratum in ("real", "generated"):
        sel = np.where(pool & (label == stratum))[0]
        if not len(sel):
            continue
        test_mask = group_split(scenarios[sel], TEST_FRACTION, seed)
        split[sel[test_mask]] = "holdout"
        split[sel[~test_mask]] = "calib"

    # 4. benchmark clips that are weather siblings of a reserved scenario:
    #    unusable for calibration, not independent of e6_test either.
    split[(split == "unused") & gen_pool & in_bench & is_reserved] = "e6_sibling"

    # 5. test-only generated clips with no benchmark label.
    split[(split == "unused") & (label == "generated") & ~in_bench] = "extra_gen"

    out = df.with_columns(pl.Series("split", [str(s) for s in split]))

    # ``pool_generator``: the finer generated grouping used by the per-generator
    # TPR tables -- unlabelled dd is a different population from benchmark dd.
    out = out.with_columns(
        pl.when((pl.col("label") == "generated") & ~pl.col("in_benchmark"))
        .then(pl.col("generator") + pl.lit("_unlabelled"))
        .otherwise(pl.col("generator"))
        .alias("pool_generator")
    )
    return out


# --------------------------------------------------------------------------
# Checks / reporting
# --------------------------------------------------------------------------


def check(df: pl.DataFrame, strict: bool = True) -> tuple[list[str], dict]:
    """Fold-disjointness checks.  Returns ``(printed lines, overlap counts)``.

    With ``strict`` (the default ``--pool leakfree``) any shared ``scenario_id``
    between ``calib`` and a test fold raises; ``--pool benchmark`` only warns,
    since creating exactly that leak is the point of the comparison.
    """
    lines: list[str] = []
    overlaps: dict[str, int] = {}
    calib_scen = set(df.filter(pl.col("split") == "calib")["scenario_id"])
    for other in TEST_SPLITS + ["calib18", "e6_sibling"]:
        other_scen = set(df.filter(pl.col("split") == other)["scenario_id"])
        overlap = calib_scen & other_scen
        overlaps[other] = len(overlap)
        flag = "" if not overlap else ("  <-- LEAK" if strict else "  <-- leak (expected for --pool benchmark)")
        lines.append(f"  scenario overlap calib n {other:<11}: {len(overlap)}{flag}")
        if overlap and strict:
            raise AssertionError(f"scenario_id leak between calib and {other}: {sorted(overlap)[:5]}")

    hold_scen = set(df.filter(pl.col("split") == "holdout")["scenario_id"])
    for other in ("e6_test", "extra_gen", "calib18"):
        other_scen = set(df.filter(pl.col("split") == other)["scenario_id"])
        overlaps[f"holdout~{other}"] = len(hold_scen & other_scen)
        lines.append(f"  scenario overlap holdout n {other:<9}: {len(hold_scen & other_scen)}")

    if df["clip_id"].n_unique() != df.height:
        raise AssertionError("duplicate clip_id in the manifest")
    n_unused = int((df["split"] == "unused").sum())
    lines.append(f"  clips assigned               : {df.height} (unused: {n_unused})")
    overlaps["unused"] = n_unused
    return lines, overlaps


def summarise(df: pl.DataFrame) -> pl.DataFrame:
    # one column per *generated* generator actually present in the manifest, so
    # dropping (or adding) a generator needs no edit here.
    gens = sorted(
        df.filter(pl.col("label") == "generated")["generator"].unique().to_list()
    )
    return (
        df.group_by("split")
        .agg(
            pl.len().alias("n"),
            (pl.col("label") == "real").sum().alias("n_real"),
            (pl.col("label") == "generated").sum().alias("n_gen"),
            pl.col("scenario_id").n_unique().alias("n_scenarios"),
            *[(pl.col("generator") == g).sum().alias(g) for g in gens],
        )
        .with_columns(
            pl.when(pl.col("n_gen") > 0)
            .then(pl.col("n_real") / pl.col("n_gen"))
            .otherwise(None)
            .round(3)
            .alias("real_per_gen")
        )
        .with_columns(
            pl.col("split")
            .replace_strict({s: i for i, s in enumerate(SPLIT_ORDER)}, default=99, return_dtype=pl.Int64)
            .alias("_o")
        )
        .sort("_o")
        .drop("_o")
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--e6", type=Path, default=REALGEN_V2_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--summary", type=Path, default=None, help="default: <out dir>/splits_summary.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e6-ids", type=Path, default=E6_IDS_TXT,
                    help="id list the n10 condition is restricted to; cross-checked against the "
                         "E6 json, and written if absent")
    ap.add_argument(
        "--pool",
        choices=POOL_MODES,
        default="leakfree",
        help="calib/holdout pool: 'leakfree' (default, scenarios untouched by e6_test/calib18) "
        "or 'benchmark' (the naive 'remaining benchmark' pool, which leaks into e6_test)",
    )
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}  (run src/fft/manifest.py first)")
    man = pl.read_parquet(args.manifest)
    print(f"manifest: {rel_to_repo(args.manifest)}  ({man.height} clips)")

    e6_ids, e6_info = load_e6_ids(args.e6)
    known = set(man["clip_id"])
    matched = [c for c in e6_ids if c in known]
    print(f"E6 set  : {rel_to_repo(args.e6)}")
    print(
        f"  {e6_info['records']} records, {e6_info['unique_clip_ids']} unique clip_ids, "
        f"{e6_info['n_real']} real / {e6_info['n_generated']} generated"
    )
    print(f"  clip_id field disagreements vs rel_video: {e6_info['clip_id_field_disagreements']}")
    print(f"  matched against the manifest: {len(matched)} / {len(e6_ids)}")
    if len(matched) != 2000:
        print(f"  !! expected 2000 matches, got {len(matched)}")
    unmatched = sorted(set(e6_ids) - known)
    if unmatched:
        print(f"  !! {len(unmatched)} unmatched, e.g. {unmatched[:5]}")

    # The n10 re-encode is produced from this id list, and calibrate.py's e6cv
    # design assumes the two agree; a silent drift would put calibration clips
    # into the test set, so check rather than trust.
    id_file_status = "absent"
    if args.e6_ids.exists():
        listed = {ln.strip() for ln in args.e6_ids.read_text().splitlines() if ln.strip()}
        only_file = sorted(listed - set(matched))
        only_json = sorted(set(matched) - listed)
        if only_file or only_json:
            id_file_status = f"MISMATCH ({len(only_file)} only in the txt, {len(only_json)} only in the json)"
            print(f"  !! {rel_to_repo(args.e6_ids)} disagrees with the E6 json: {id_file_status}")
            for c in (only_file[:3] + only_json[:3]):
                print(f"     e.g. {c}")
        else:
            id_file_status = f"matches ({len(listed)} ids)"
            print(f"  {rel_to_repo(args.e6_ids)}: {id_file_status}")
    else:
        print(f"  note: {rel_to_repo(args.e6_ids)} not present; writing it")
        args.e6_ids.parent.mkdir(parents=True, exist_ok=True)
        args.e6_ids.write_text("\n".join(matched) + "\n")
        id_file_status = f"written ({len(matched)} ids)"

    df = assign(man, matched, args.seed, args.pool)
    print(f"pool mode: {args.pool}")

    n_e6_calib18 = int((df["in_e6"] & df["in_calib18"]).sum())
    if n_e6_calib18:
        print(
            f"\nNOTE: {n_e6_calib18} clips are in BOTH the E6 set and the original 18+18 "
            f"calibration set.\n      They are assigned to 'calib18' and dropped from the "
            f"'e6_test' fold, so the\n      E6 fold evaluated here has "
            f"{int((df['split'] == 'e6_test').sum())} of the {len(matched)} E6 clips."
        )

    print("\n=== split table ===")
    summary = summarise(df)
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=180):
        print(summary)

    print("\n=== per generator x split ===")
    with pl.Config(tbl_rows=60, tbl_width_chars=180):
        print(
            df.group_by("split", "pool_generator")
            .agg(pl.len().alias("n"), pl.col("scenario_id").n_unique().alias("n_scenarios"))
            .sort("split", "pool_generator")
        )

    print("\n=== assertions ===")
    lines, overlaps = check(df, strict=args.pool == "leakfree")
    for line in lines:
        print(line)

    out_df = df.select(
        "clip_id", "split", "in_e6", "in_calib18", "label", "generator", "pool_generator", "scenario_id"
    ).sort("clip_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    out_df.write_parquet(tmp)
    tmp.replace(args.out)
    print(f"\nwrote {rel_to_repo(args.out)}  ({out_df.height} rows)")

    summary_path = args.summary or args.out.parent / "splits_summary.json"
    payload = {
        "seed": args.seed,
        "pool_mode": args.pool,
        "test_fraction": TEST_FRACTION,
        "scenario_overlaps": overlaps,
        "manifest": rel_to_repo(args.manifest),
        "e6_json": rel_to_repo(args.e6),
        "e6": {**e6_info, "matched_in_manifest": len(matched), "unmatched": unmatched},
        "e6_in_calib18": n_e6_calib18,
        "e6_ids_file": rel_to_repo(args.e6_ids),
        "e6_ids_file_status": id_file_status,
        "splits": summary.to_dicts(),
        "per_generator": df.group_by("split", "pool_generator")
        .agg(pl.len().alias("n"), pl.col("scenario_id").n_unique().alias("n_scenarios"))
        .sort("split", "pool_generator")
        .to_dicts(),
    }
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(summary_path)
    print(f"wrote {rel_to_repo(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
