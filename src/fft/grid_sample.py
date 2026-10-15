"""
Draw the 600-clip subset used for the FFT robustness grid.

Re-encoding all 5510 clips under 16 perturbation conditions is ~90k ffmpeg runs;
the robustness questions (does the anisotropy signal survive compression /
rescaling / a frame-rate change / a different codec?) are answered just as well
on a stratified subset, and a subset makes the grid re-runnable in an evening.

Sample
------
300 real + 300 generated, all drawn from ``holdout`` u ``e6_test`` (never from
``calib``, so the grid can be scored with the calibrated threshold without
circularity), seed 0:

* 300 real   -- proportional to how many real clips each of the two folds has.
* 150 dd     -- at most one clip per ``scenario_id``, round-robin over the
  (folder x weather) cells so no weather or folder dominates.
* 150 p1     -- same, round-robin over p1's weather cells.

Usage
-----
    .venv/bin/python src/fft/grid_sample.py
    .venv/bin/python src/fft/grid_sample.py --n-real 300 --n-dd 150 --n-p1 150

Outputs
-------
``results/fft/grid_sample.parquet``    the sample + its strata (analysis input)
``results/fft/grid_manifest.parquet``  the same clips in *manifest schema*, so
                                       ``perturb.py --manifest`` and
                                       ``extract_features.py --manifest`` can be
                                       pointed straight at it (much saner than
                                       passing 600 ids through ``--filter``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft.common import DEFAULT_MANIFEST, RESULTS_FFT_DIR, rel_to_repo  # noqa: E402
from fft.splits import DEFAULT_SPLITS  # noqa: E402

DEFAULT_GRID = RESULTS_FFT_DIR / "grid_sample.parquet"
DEFAULT_GRID_MANIFEST = RESULTS_FFT_DIR / "grid_manifest.parquet"

SOURCE_SPLITS = ["holdout", "e6_test"]


def round_robin_by_cell(df: pl.DataFrame, n: int, cell_cols: list[str], seed: int) -> pl.DataFrame:
    """At most one clip per ``scenario_id``, cycling over the ``cell_cols`` cells.

    Deterministic given ``seed``: rows are shuffled once inside each cell with a
    seeded permutation, and the cells are visited in sorted order.
    """
    if df.is_empty() or n <= 0:
        return df.head(0)
    rng = np.random.default_rng(seed)
    d = df.with_columns(
        pl.concat_str([pl.col(c).fill_null("None") for c in cell_cols], separator="|").alias("_cell")
    )
    cells = sorted(d["_cell"].unique().to_list())
    queues: dict[str, list[dict]] = {}
    for cell in cells:
        part = d.filter(pl.col("_cell") == cell).sort("clip_id")
        order = rng.permutation(part.height)
        queues[cell] = [part.row(int(i), named=True) for i in order]

    picked: list[dict] = []
    used_scen: set[str] = set()
    while len(picked) < n and any(queues.values()):
        progressed = False
        for cell in cells:
            if len(picked) >= n:
                break
            q = queues[cell]
            while q:
                row = q.pop()
                if row["scenario_id"] in used_scen:
                    continue
                used_scen.add(row["scenario_id"])
                picked.append(row)
                progressed = True
                break
        if not progressed:
            break
    return pl.DataFrame(picked).drop("_cell") if picked else df.head(0)


def sample_real(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """``n`` real clips, allocated across the source folds in proportion to size."""
    if df.is_empty():
        return df
    rng = np.random.default_rng(seed + 1)
    parts = []
    total = df.height
    allocated = 0
    folds = sorted(df["split"].unique().to_list())
    for i, f in enumerate(folds):
        sub = df.filter(pl.col("split") == f).sort("clip_id")
        want = n - allocated if i == len(folds) - 1 else int(round(n * sub.height / total))
        want = min(want, sub.height)
        idx = rng.permutation(sub.height)[:want]
        parts.append(sub[np.sort(idx)])
        allocated += want
    return pl.concat(parts) if parts else df.head(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_GRID)
    ap.add_argument("--out-manifest", type=Path, default=DEFAULT_GRID_MANIFEST)
    ap.add_argument("--n-real", type=int, default=300)
    ap.add_argument("--n-dd", type=int, default=150)
    ap.add_argument("--n-p1", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for p in (args.manifest, args.splits):
        if not p.exists():
            raise SystemExit(f"missing {p}  (run src/fft/manifest.py / src/fft/splits.py first)")
    man = pl.read_parquet(args.manifest)
    spl = pl.read_parquet(args.splits).select("clip_id", "split", "pool_generator")
    df = man.join(spl, on="clip_id", how="inner").filter(pl.col("split").is_in(SOURCE_SPLITS))
    print(f"pool: {df.height} clips from splits {SOURCE_SPLITS}")
    with pl.Config(tbl_rows=20):
        print(df.group_by("split", "label").agg(pl.len().alias("n")).sort("split", "label"))

    real = sample_real(df.filter(pl.col("label") == "real"), args.n_real, args.seed)
    dd = round_robin_by_cell(
        df.filter(pl.col("generator") == "dd"), args.n_dd, ["folder", "weather"], args.seed
    )
    p1 = round_robin_by_cell(
        df.filter(pl.col("generator") == "p1"), args.n_p1, ["folder", "weather"], args.seed + 2
    )
    for name, got, want in (("real", real.height, args.n_real), ("dd", dd.height, args.n_dd),
                            ("p1", p1.height, args.n_p1)):
        flag = "" if got == want else f"  !! only {got} available (one clip per scenario)"
        print(f"  {name}: {got}/{want}{flag}")

    grid = pl.concat([real, dd, p1], how="vertical_relaxed").sort("clip_id")
    if grid["clip_id"].n_unique() != grid.height:
        raise SystemExit("duplicate clip_id in the grid sample")

    grid = grid.with_columns(
        pl.concat_str(
            [pl.col("pool_generator"), pl.col("folder").fill_null("-"), pl.col("weather").fill_null("-")],
            separator="|",
        ).alias("stratum")
    )
    print(f"\ngrid: {grid.height} clips, {grid['scenario_id'].n_unique()} scenarios")
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(grid.group_by("label", "pool_generator", "split").agg(pl.len().alias("n"))
              .sort("label", "pool_generator", "split"))
        print(grid.filter(pl.col("label") == "generated")
              .group_by("generator", "weather").agg(pl.len().alias("n")).sort("generator", "weather"))
        print(grid.group_by("width", "height", "fps", "codec").agg(pl.len().alias("n")).sort("n", descending=True))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("clip_id", "path", "label", "generator", "pool_generator", "folder", "weather",
                        "gen_size", "scenario_id", "split", "stratum", "codec", "width", "height", "fps",
                        "nframes", "in_benchmark") if c in grid.columns]
    tmp = args.out.with_suffix(".parquet.tmp")
    grid.select(keep).write_parquet(tmp)
    tmp.replace(args.out)
    print(f"\nwrote {rel_to_repo(args.out)}  ({grid.height} rows)")

    # manifest-schema subset, so perturb.py / extract_features.py can use --manifest
    sub_man = man.filter(pl.col("clip_id").is_in(grid["clip_id"].to_list())).sort("clip_id")
    tmp = args.out_manifest.with_suffix(".parquet.tmp")
    sub_man.write_parquet(tmp)
    tmp.replace(args.out_manifest)
    print(f"wrote {rel_to_repo(args.out_manifest)}  ({sub_man.height} rows, manifest schema)")
    print("\nnext: bash src/fft/run_grid.sh   (uses the grid manifest above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
