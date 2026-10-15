"""
Batch FFT feature extraction over the clip manifest.

Usage
-----
    # raw clips, 60 frames, all 12 cores
    .venv/bin/python src/fft/extract_features.py --condition raw --workers 12

    # just the benchmark subset of cosmos-drive-dreams, cheap variant
    .venv/bin/python src/fft/extract_features.py --condition raw --max-frames 20 \\
        --filter generator=dd,in_benchmark=true --fast

    # a perturbed condition produced by src/fft/perturb.py
    .venv/bin/python src/fft/extract_features.py --condition crf35 \\
        --filter 'pl.col("in_realgen_v1")'

    # sampling-grid sensitivity probe
    .venv/bin/python src/fft/extract_features.py --condition raw --frame-offset 3 --limit 200

    # cross-check a subset against the agent's own (slow) functions
    .venv/bin/python src/fft/extract_features.py --condition raw --impl exact --limit 50

Output: ``results/fft/features_<condition>_f<max_frames>.parquet`` (override with
``--out``).  One row per
``(clip_id, condition, max_frames, frame_offset, impl, skip_periodic)``.
Re-running appends only the missing rows, so an interrupted run resumes.

``--impl fast`` (the default) is a vectorised reimplementation of the agent's
per-frame FFT metrics that is *bit-identical* to ``--impl exact`` and roughly
14x quicker; see ``src/fft/vectorized.py``.
"""

from __future__ import annotations

# Pin the numeric backends before numpy is pulled in by anything below; pool
# workers are forked from this process and inherit the environment.
import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft.common import CONDITIONS_DIR, DEFAULT_MANIFEST, RESULTS_FFT_DIR, rel_to_repo  # noqa: E402
from fft.features import FEATURE_KEYS, IMPLS, extract_features  # noqa: E402

# Rows are keyed on every input that changes the numbers, so switching --impl
# or --fast appends a new row rather than being silently skipped on resume.
KEY_COLS = ("clip_id", "condition", "max_frames", "frame_offset", "impl", "skip_periodic")
FLUSH_EVERY = 200
PROGRESS_EVERY = 100


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def _coerce(value: str):
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def build_filter(spec: str) -> pl.Expr:
    """Turn ``--filter`` into a polars expression.

    Two accepted forms:

    * a raw polars expression string, e.g. ``pl.col("aniso_mean") > 0.05`` --
      anything containing ``pl.`` is ``eval``'d with ``pl`` in scope;
    * comma-separated ``key=value`` / ``key!=value`` clauses ANDed together,
      e.g. ``generator=dd,in_benchmark=true``.  ``key=a|b`` means "is in".
    """
    spec = spec.strip()
    if "pl." in spec:
        return eval(spec, {"pl": pl, "__builtins__": {}})  # noqa: S307 -- operator-supplied

    expr = None
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if "!=" in clause:
            key, _, raw = clause.partition("!=")
            part = pl.col(key.strip()) != _coerce(raw)
        elif "=" in clause:
            key, _, raw = clause.partition("=")
            key = key.strip()
            if "|" in raw:
                part = pl.col(key).is_in([_coerce(v) for v in raw.split("|")])
            else:
                part = pl.col(key) == _coerce(raw)
        else:
            raise SystemExit(f"--filter: cannot parse clause {clause!r}")
        expr = part if expr is None else (expr & part)
    if expr is None:
        raise SystemExit("--filter: empty expression")
    return expr


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

_CFG: dict = {}


def _init(max_frames: int, frame_offset: int, fast: bool, condition: str, impl: str) -> None:
    _CFG.update(
        max_frames=max_frames, frame_offset=frame_offset, fast=fast, condition=condition, impl=impl
    )


def _work(task: tuple[str, str]) -> dict:
    clip_id, path = task
    t0 = time.perf_counter()
    row = extract_features(
        path,
        max_frames=_CFG["max_frames"],
        frame_offset=_CFG["frame_offset"],
        fast=_CFG["fast"],
        impl=_CFG["impl"],
    )
    row.update(
        clip_id=clip_id,
        condition=_CFG["condition"],
        max_frames=_CFG["max_frames"],
        frame_offset=_CFG["frame_offset"],
        impl=_CFG["impl"],
        skip_periodic=_CFG["fast"],
        elapsed_s=round(time.perf_counter() - t0, 3),
    )
    return row


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

SCHEMA = {
    "clip_id": pl.Utf8,
    "condition": pl.Utf8,
    "max_frames": pl.Int64,
    "frame_offset": pl.Int64,
    "impl": pl.Utf8,
    "skip_periodic": pl.Boolean,
    "n_frames": pl.Int64,
    "aniso_mean": pl.Float64,
    "aniso_std": pl.Float64,
    "aniso_max": pl.Float64,
    "peak_ratio_mean": pl.Float64,
    "axis_ratio_mean": pl.Float64,
    "hf_ratio_mean": pl.Float64,
    "hf_ratio_std": pl.Float64,
    "hf_ratio_min": pl.Float64,
    "hf_ratio_max": pl.Float64,
    "spectral_slope_mean": pl.Float64,
    "spectral_slope_std": pl.Float64,
    "anomaly_ratio_mean": pl.Float64,
    "anomaly_ratio_max": pl.Float64,
    "periodic_artifact_frames": pl.Int64,
    "width": pl.Int64,
    "height": pl.Int64,
    "error": pl.Utf8,
    "elapsed_s": pl.Float64,
}
assert set(FEATURE_KEYS) <= set(SCHEMA), "features.FEATURE_KEYS drifted from the parquet schema"


def write_atomic(df: pl.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(out)


def resolve_path(clip_id: str, manifest_path: str, condition: str) -> Path:
    if condition == "raw":
        return Path(manifest_path)
    return CONDITIONS_DIR / condition / f"{clip_id}.mp4"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--condition", default="raw", help="'raw' = manifest paths; anything else reads dataset/fft_conditions/<condition>/")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--frame-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--fast", action="store_true", help="skip detect_periodic_artifacts (~2x faster)")
    ap.add_argument(
        "--impl",
        choices=IMPLS,
        default="fast",
        help="'fast' = vectorised, bit-identical reimplementation (default); 'exact' = the agent's own functions",
    )
    ap.add_argument("--filter", default=None, help="polars expr string, or key=value[,key=value]")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true", help="ignore existing rows instead of resuming")
    args = ap.parse_args()

    out = args.out or RESULTS_FFT_DIR / f"features_{args.condition}_f{args.max_frames}.parquet"

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}  (run src/fft/manifest.py first)")
    man = pl.read_parquet(args.manifest)
    print(f"manifest: {rel_to_repo(args.manifest)}  ({len(man)} clips)")

    if args.filter:
        man = man.filter(build_filter(args.filter))
        print(f"filter  : {args.filter}  -> {len(man)} clips")

    tasks = [
        (row["clip_id"], str(resolve_path(row["clip_id"], row["path"], args.condition)))
        for row in man.select("clip_id", "path").iter_rows(named=True)
    ]

    # Drop rows already computed for this exact (clip_id, condition, max_frames, offset).
    done: set[str] = set()
    existing = None
    if out.exists() and not args.overwrite:
        existing = pl.read_parquet(out)
        stale = [c for c in SCHEMA if c not in existing.columns]
        if stale:
            raise SystemExit(
                f"{rel_to_repo(out)} was written by an older schema (missing {stale}).\n"
                f"Re-run with --overwrite, or point --out at a new file."
            )
        same = existing.filter(
            (pl.col("condition") == args.condition)
            & (pl.col("max_frames") == args.max_frames)
            & (pl.col("frame_offset") == args.frame_offset)
            & (pl.col("impl") == args.impl)
            & (pl.col("skip_periodic") == args.fast)
        )
        done = set(same["clip_id"])
        print(f"resume  : {rel_to_repo(out)} has {len(existing)} rows, {len(done)} match this key -> skipped")
        tasks = [t for t in tasks if t[0] not in done]

    missing = [t for t in tasks if not Path(t[1]).exists()]
    if missing:
        print(f"WARNING : {len(missing)} clips have no file for condition '{args.condition}' -> skipped")
        print(f"          e.g. {missing[0][1]}")
        skip = {t[0] for t in missing}
        tasks = [t for t in tasks if t[0] not in skip]

    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("nothing to do")
        return 0

    print(
        f"running : {len(tasks)} clips, condition={args.condition}, max_frames={args.max_frames}, "
        f"frame_offset={args.frame_offset}, impl={args.impl}, skip_periodic={args.fast}, "
        f"workers={args.workers}"
    )

    rows: list[dict] = []
    t0 = time.perf_counter()
    n_err = 0
    with Pool(
        args.workers,
        initializer=_init,
        initargs=(args.max_frames, args.frame_offset, args.fast, args.condition, args.impl),
        maxtasksperchild=50,
    ) as pool:
        for i, row in enumerate(pool.imap_unordered(_work, tasks, chunksize=1), start=1):
            rows.append(row)
            if row.get("error"):
                n_err += 1
            if i % PROGRESS_EVERY == 0 or i == len(tasks):
                el = time.perf_counter() - t0
                rate = i / el
                eta = (len(tasks) - i) / rate if rate else 0
                print(
                    f"  {i}/{len(tasks)}  {el / 60:.1f} min elapsed  "
                    f"{rate:.2f} clip/s  ETA {eta / 60:.1f} min  errors={n_err}",
                    flush=True,
                )
            if i % FLUSH_EVERY == 0:
                write_atomic(_assemble(existing, rows), out)

    total = time.perf_counter() - t0
    df = _assemble(existing, rows)
    write_atomic(df, out)

    print(
        f"\ndone: {len(rows)} clips in {total / 60:.1f} min "
        f"({total / len(rows):.2f} s/clip wall, {args.workers * total / len(rows):.2f} s/clip cpu), "
        f"{n_err} errors"
    )
    print(f"wrote {rel_to_repo(out)}  ({len(df)} rows total)")
    if n_err:
        bad = df.filter(pl.col("error").is_not_null()).select("clip_id", "error")
        with pl.Config(tbl_rows=15, fmt_str_lengths=100):
            print(bad.head(15))
    return 0


def _assemble(existing: pl.DataFrame | None, rows: list[dict]) -> pl.DataFrame:
    new = pl.DataFrame(rows, schema=SCHEMA)
    if existing is None or not len(existing):
        return new.sort(list(KEY_COLS))
    return (
        pl.concat([existing.select(new.columns), new], how="vertical")
        .unique(subset=list(KEY_COLS), keep="last")
        .sort(list(KEY_COLS))
    )


if __name__ == "__main__":
    raise SystemExit(main())
