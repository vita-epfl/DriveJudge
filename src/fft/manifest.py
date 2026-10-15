"""
Build the clip manifest for the FFT real-vs-generated calibration study.

Enumerates every source video (cosmos-drive-dreams / -predict1 + the CarCrash
real footage), probes it with ffprobe in a process pool and writes one row per
clip to ``results/fft/manifest.parquet``.

``--exclude-generators`` drops whole source pools from the manifest, and
defaults to ``p2``: cosmos-predict2 is not part of the paper, so its clips --
which are still on disk -- must not enter any manifest-derived analysis, figure
or table.  Every downstream script reads the manifest, so excluding a generator
here removes it everywhere.

Usage
-----
    .venv/bin/python src/fft/manifest.py                    # 5510 clips (p2 excluded)
    .venv/bin/python src/fft/manifest.py --exclude-generators   # keep everything
    .venv/bin/python src/fft/manifest.py --include-crash    # + Crash-1500
    .venv/bin/python src/fft/manifest.py --out /tmp/m.parquet --workers 8

The run is idempotent: it always re-enumerates from scratch and overwrites the
output parquet.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft.common import (  # noqa: E402
    BLACKLISTED_STEMS,
    DEFAULT_MANIFEST,
    FFT_CALIB_DIR,
    FINAL_DATASET_ROOT,
    GENERATED_SOURCES,
    QUESTION_FILES,
    REALGEN_V1_JSON,
    REAL_CRASH_DIR,
    REAL_NORMAL_DIR,
    make_clip_id,
    parse_clip_name,
    rel_to_repo,
)

SCHEMA = {
    "clip_id": pl.Utf8,
    "path": pl.Utf8,
    "label": pl.Utf8,
    "generator": pl.Utf8,
    "folder": pl.Utf8,
    "scenario_id": pl.Utf8,
    "weather": pl.Utf8,
    "gen_size": pl.Utf8,
    "category": pl.Utf8,
    "n_questions": pl.Int64,
    "in_benchmark": pl.Boolean,
    "in_realgen_v1": pl.Boolean,
    "in_fft_calib18": pl.Boolean,
    "codec": pl.Utf8,
    "width": pl.Int64,
    "height": pl.Int64,
    "fps": pl.Float64,
    "nframes": pl.Int64,
    "duration": pl.Float64,
    "size_bytes": pl.Int64,
}


# --------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------


def _parse_rate(value) -> float | None:
    """``"24/1"`` / ``"30000/1001"`` -> float."""
    if not value:
        return None
    try:
        if "/" in str(value):
            num, den = str(value).split("/", 1)
            den = float(den)
            return float(num) / den if den else None
        return float(value)
    except (TypeError, ValueError):
        return None


def probe(path: str) -> dict:
    """ffprobe one file -> ``{codec,width,height,fps,nframes,duration,size_bytes}``."""
    out = {
        "codec": None,
        "width": None,
        "height": None,
        "fps": None,
        "nframes": None,
        "duration": None,
        "size_bytes": None,
        "probe_error": None,
    }
    try:
        out["size_bytes"] = Path(path).stat().st_size
    except OSError as exc:
        out["probe_error"] = f"stat: {exc}"
        return out

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        out["probe_error"] = f"ffprobe: {exc}"
        return out
    if res.returncode != 0:
        out["probe_error"] = (res.stderr or "ffprobe failed").strip()[:200]
        return out

    try:
        data = json.loads(res.stdout or "{}")
    except json.JSONDecodeError as exc:
        out["probe_error"] = f"json: {exc}"
        return out

    streams = data.get("streams") or []
    if not streams:
        out["probe_error"] = "no video stream"
        return out
    st = streams[0]

    out["codec"] = st.get("codec_name")
    out["width"] = int(st["width"]) if st.get("width") is not None else None
    out["height"] = int(st["height"]) if st.get("height") is not None else None

    fps = _parse_rate(st.get("avg_frame_rate")) or _parse_rate(st.get("r_frame_rate"))
    out["fps"] = round(fps, 6) if fps else None

    dur = st.get("duration") or (data.get("format") or {}).get("duration")
    try:
        out["duration"] = float(dur) if dur is not None else None
    except (TypeError, ValueError):
        out["duration"] = None

    nb = st.get("nb_frames")
    if nb not in (None, "N/A"):
        try:
            out["nframes"] = int(nb)
        except (TypeError, ValueError):
            out["nframes"] = None
    if out["nframes"] is None and out["fps"] and out["duration"]:
        # Containers such as webm/mkv do not carry nb_frames.
        out["nframes"] = int(round(out["duration"] * out["fps"]))

    return out


def _probe_row(row: dict) -> dict:
    row = dict(row)
    row.update(probe(row["path"]))
    return row


# --------------------------------------------------------------------------
# Membership sets
# --------------------------------------------------------------------------


def load_benchmark_keys() -> tuple[dict[tuple[str, str], tuple[str, int]], int]:
    """Benchmark lookup ``(generator, stem) -> (category, n_questions)``, + raw count.

    ``category`` is the *benchmark category label* -- the leading directory of
    the question file's ``video`` field (``negative``, ``crossing_stop``,
    ``crossing_red_lights``, ``aggressive_takeover``, ``artifacts``).  It is not
    always the on-disk folder, which the manifest keeps separately in ``folder``.

    Three dd clips are listed under *two* labels with *different* question sets
    (``agg_tk_030_Foggy`` and ``agg_tk_025_Golden hour`` as negative[1 q] +
    aggressive_takeover[7 q]; ``cross_stop_041_Morning`` as negative[4 q] +
    crossing_stop[6 q]).  For those, ``category`` is the sorted labels joined
    with ``"|"`` and ``n_questions`` is the total across both entries.

    An entry counts only when it carries no truthy ``remove`` flag, and the
    hand-blacklisted stem is dropped.

    Matching is by ``(generator, stem)`` rather than by the full relative path:
    the ``video`` field's leading directory is the *benchmark category label*,
    not the on-disk folder.  26 dd entries are labelled ``negative/`` while the
    file physically lives under ``aggressive_takeover/`` /
    ``crossing_stop/`` / ``crossing_red_lights/``.  Stems are globally unique
    within dd and within p1, so the basename is an unambiguous key.
    """
    labels: dict[tuple[str, str], set[str]] = {}
    nq: dict[tuple[str, str], int] = {}
    raw_total = 0
    for gen, jf in QUESTION_FILES.items():
        if not jf.exists():
            print(f"  WARNING: missing question file {rel_to_repo(jf)}")
            continue
        entries = json.loads(jf.read_text())
        kept = blacklisted = 0
        for entry in entries:
            if entry.get("remove"):
                continue
            kept += 1
            fname = entry["video"].rpartition("/")[2]
            stem = fname[:-4] if fname.lower().endswith(".mp4") else fname
            if stem in BLACKLISTED_STEMS:
                blacklisted += 1
                continue
            key = (gen, stem)
            labels.setdefault(key, set()).add(entry["video"].rpartition("/")[0] or "")
            nq[key] = nq.get(key, 0) + len(entry.get("qa_pairs") or [])
        raw_total += kept
        print(
            f"  benchmark {gen}: {kept} kept / {len(entries)} entries"
            + (f"  (-{blacklisted} blacklisted)" if blacklisted else "")
        )
    lookup = {k: ("|".join(sorted(v)), nq[k]) for k, v in labels.items()}
    return lookup, raw_total


def load_realgen_v1_keys() -> set[tuple[str, str]]:
    """``(generator, stem)`` keys of the previous 100-clip real-vs-generated study."""
    if not REALGEN_V1_JSON.exists():
        print(f"  WARNING: missing {rel_to_repo(REALGEN_V1_JSON)}")
        return set()
    keys = set()
    for entry in json.loads(REALGEN_V1_JSON.read_text()):
        rel = entry["rel_video"]  # "<source>/<folder>/<file>.mp4" or "real/<file>.mp4"
        parts = rel.split("/")
        gen = GENERATED_SOURCES.get(parts[0], parts[0])  # "real" stays "real"
        fname = parts[-1]
        stem = fname[:-4] if fname.lower().endswith(".mp4") else fname
        keys.add((gen, stem))
    print(f"  realgen_v1: {len(keys)} (generator, stem) keys")
    return keys


def load_calib18_keys() -> tuple[set[tuple[str, str]], set[str], set[str]]:
    """``(generator, stem)`` keys of the original 18+18 FFT calibration clips.

    Collected from every ``raw_videos/`` directory under ``dataset/fft_test/``.
    The parent directory names the source pool (``real_footage`` ->
    ``real``, ``cosmos-drive-dreams`` -> ``dd``, ...), which is what keeps the
    match unambiguous: three of the dd calibration basenames
    (``cross_red_002_{Foggy,Morning,Golden hour}``) also exist in
    cosmos-predict1, so a plain basename match would over-select.

    Returns ``(keys, real_stems, generated_stems)``.
    """
    keys: set[tuple[str, str]] = set()
    real_stems, gen_stems = set(), set()
    if not FFT_CALIB_DIR.exists():
        print(f"  WARNING: missing {rel_to_repo(FFT_CALIB_DIR)}")
        return keys, real_stems, gen_stems
    for mp4 in FFT_CALIB_DIR.rglob("raw_videos/*.mp4"):
        # .../fft_test/<pool>/raw_videos/<stem>.mp4
        pool = mp4.parent.parent.name
        if pool == "real_footage":
            gen = "real"
            real_stems.add(mp4.stem)
        else:
            gen = GENERATED_SOURCES.get(pool, pool)
            gen_stems.add(mp4.stem)
        keys.add((gen, mp4.stem))
    print(
        f"  fft_calib18: {len(real_stems)} real + {len(gen_stems)} generated "
        f"raw videos under {rel_to_repo(FFT_CALIB_DIR)}"
    )
    return keys, real_stems, gen_stems


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------


def enumerate_clips(include_crash: bool, exclude: frozenset[str] = frozenset()) -> list[dict]:
    rows: list[dict] = []

    for source, gen in GENERATED_SOURCES.items():
        if gen in exclude or source in exclude:
            print(f"  {source} ({gen}): EXCLUDED")
            continue
        root = FINAL_DATASET_ROOT / source
        if not root.is_dir():
            print(f"  WARNING: missing source dir {root}")
            continue
        n0 = len(rows)
        for mp4 in sorted(root.rglob("*.mp4")):
            folder = str(mp4.parent.relative_to(root))
            if folder == ".":
                folder = ""
            scenario_id, weather, gen_size = parse_clip_name(source, folder, mp4.name)
            rows.append(
                {
                    "clip_id": make_clip_id(gen, folder or None, mp4.stem),
                    "path": str(mp4),
                    "label": "generated",
                    "generator": gen,
                    "folder": folder or None,
                    "scenario_id": scenario_id,
                    "weather": weather,
                    "gen_size": gen_size,
                }
            )
        print(f"  {source} ({gen}): {len(rows) - n0} clips")

    real_dirs = [("real", REAL_NORMAL_DIR, "Normal")]
    if include_crash:
        real_dirs.append(("real_crash", REAL_CRASH_DIR, "Crash-1500"))
    for gen, root, folder in real_dirs:
        if gen in exclude:
            print(f"  {folder} ({gen}): EXCLUDED")
            continue
        if not root.is_dir():
            print(f"  WARNING: missing real dir {root}")
            continue
        n0 = len(rows)
        for mp4 in sorted(root.glob("*.mp4")):
            scenario_id, weather, gen_size = parse_clip_name(gen, folder, mp4.name)
            rows.append(
                {
                    "clip_id": make_clip_id(gen, None, mp4.stem),
                    "path": str(mp4),
                    "label": "real",
                    "generator": gen,
                    "folder": folder,
                    "scenario_id": scenario_id,
                    "weather": weather,
                    "gen_size": gen_size,
                }
            )
        print(f"  {folder} ({gen}): {len(rows) - n0} clips")

    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_summary(df: pl.DataFrame) -> None:
    def show(title: str, frame: pl.DataFrame) -> None:
        print(f"\n--- {title} ---")
        with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=160):
            print(frame)

    gens = sorted(df["generator"].unique().to_list())
    counts = dict(df.group_by("generator").agg(pl.len().alias("n")).iter_rows())
    print(f"\ngenerators: {', '.join(f'{g} ({counts[g]})' for g in gens)}")

    show(
        "clips per generator",
        df.group_by("generator")
        .agg(
            pl.len().alias("n"),
            pl.col("in_benchmark").sum().alias("in_benchmark"),
            pl.col("in_realgen_v1").sum().alias("in_realgen_v1"),
            pl.col("in_fft_calib18").sum().alias("in_calib18"),
            pl.col("scenario_id").n_unique().alias("n_scenarios"),
        )
        .sort("generator"),
    )

    show(
        "clips per generator/folder",
        df.group_by("generator", "folder")
        .agg(pl.len().alias("n"), pl.col("in_benchmark").sum().alias("in_benchmark"))
        .sort("generator", "folder"),
    )

    show(
        "(codec, width, height, fps, nframes) per generator",
        df.group_by("generator", "codec", "width", "height", "fps", "nframes")
        .agg(pl.len().alias("n"))
        .sort("generator", "n", descending=[False, True]),
    )

    show(
        "benchmark category (question-file label) x on-disk folder",
        df.filter(pl.col("in_benchmark"))
        .group_by("generator", "category", "folder")
        .agg(pl.len().alias("n_clips"), pl.col("n_questions").sum().alias("n_questions"))
        .sort("generator", "category", "folder"),
    )

    show(
        "weather x generator",
        df.group_by("generator", "weather").agg(pl.len().alias("n")).sort("generator", "weather"),
    )

    bad = df.filter(pl.col("codec").is_null())
    if len(bad):
        print(f"\n!! {len(bad)} clips failed ffprobe:")
        print(bad.select("clip_id", "path").head(20))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--include-crash", action="store_true", help="also index CarCrash/Crash-1500 (tag real_crash)")
    ap.add_argument("--no-probe", action="store_true", help="skip ffprobe (metadata columns left null)")
    ap.add_argument(
        "--exclude-generators",
        nargs="*",
        default=["p2"],
        metavar="GEN",
        help="generator tags (or source directory names) to leave out of the manifest entirely. "
        "Default: p2 (cosmos-predict2 is not part of the paper). Pass with no value to keep everything.",
    )
    args = ap.parse_args()

    exclude = frozenset(args.exclude_generators or ())

    print(f"final_dataset: {FINAL_DATASET_ROOT}")
    print(f"real footage : {REAL_NORMAL_DIR.parent}")
    print(f"excluded generators: {', '.join(sorted(exclude)) if exclude else '(none)'}")
    print("\n[1/4] enumerating clips")
    rows = enumerate_clips(args.include_crash, exclude)
    print(f"  total: {len(rows)} clips")

    ids = [r["clip_id"] for r in rows]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        print(f"  !! {len(dupes)} duplicate clip_ids, e.g. {dupes[:5]}")

    print("\n[2/4] membership sets")
    bench, bench_raw = load_benchmark_keys()
    realgen = load_realgen_v1_keys()
    calib, calib_real, calib_gen = load_calib18_keys()

    bench_hits: set = set()
    realgen_hits: set = set()
    calib_hits: set = set()
    for r in rows:
        stem = Path(r["path"]).stem
        key = (r["generator"], stem)
        hit = bench.get(key)
        r["in_benchmark"] = hit is not None
        r["category"] = hit[0] if hit else None
        r["n_questions"] = hit[1] if hit else 0
        r["in_realgen_v1"] = key in realgen
        r["in_fft_calib18"] = key in calib
        if r["in_benchmark"]:
            bench_hits.add(key)
        if r["in_realgen_v1"]:
            realgen_hits.add(key)
        if r["in_fft_calib18"]:
            calib_hits.add(key)

    for name, wanted, hit in (
        ("benchmark", set(bench), bench_hits),
        ("realgen_v1", realgen, realgen_hits),
        ("fft_calib18", calib, calib_hits),
    ):
        missing = wanted - hit
        if missing:
            print(f"  note: {len(missing)} {name} keys have no file on disk: {sorted(missing)[:10]}")

    print("\n[3/4] ffprobe")
    if args.no_probe:
        for r in rows:
            r.update({k: None for k in ("codec", "width", "height", "fps", "nframes", "duration", "size_bytes")})
    else:
        with Pool(args.workers) as pool:
            probed = []
            for i, row in enumerate(pool.imap(_probe_row, rows, chunksize=16), start=1):
                probed.append(row)
                if i % 500 == 0 or i == len(rows):
                    print(f"  {i}/{len(rows)}")
        rows = probed
        errs = [r for r in rows if r.get("probe_error")]
        if errs:
            print(f"  !! {len(errs)} probe errors, first: {errs[0]['clip_id']}: {errs[0]['probe_error']}")

    for r in rows:
        r.pop("probe_error", None)

    df = pl.DataFrame(rows, schema=SCHEMA).sort("clip_id")

    print("\n[4/4] writing")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(args.out)
    print(f"  {rel_to_repo(args.out)}  ({len(df)} rows, {args.out.stat().st_size / 1e6:.2f} MB)")

    print_summary(df)

    print("\n=== checks ===")
    n_black = len(BLACKLISTED_STEMS & {Path(r["path"]).stem for r in rows})
    # The question files count *entries*; in_benchmark counts *distinct video
    # files*.  The two differ because (a) a handful of clips are listed twice
    # under two category labels and (b) the blacklisted stem is dropped.
    for gen, entries_expected in (("dd", 1103), ("p1", 495)):
        got = int(df.filter((pl.col("generator") == gen) & pl.col("in_benchmark")).height)
        n_keys = sum(1 for g, _ in bench if g == gen)
        black = int(
            df.filter(
                (pl.col("generator") == gen)
                & pl.col("path").str.split("/").list.last().str.head(-4).is_in(list(BLACKLISTED_STEMS))
            ).height
        )
        dupes = entries_expected - n_keys - black
        note = "OK" if got == n_keys else f"MISMATCH: {n_keys} keys resolved to {got} files"
        print(
            f"  in_benchmark {gen}: {got} clips  "
            f"[{entries_expected} question entries - {dupes} duplicate-label entries "
            f"- {black} blacklisted = {got}]  {note}"
        )
    print(f"  benchmark entries total (remove-flag applied): {bench_raw}")
    got = int(df.filter(pl.col("in_realgen_v1")).height)
    print(f"  in_realgen_v1: {got}  {'OK' if got == 100 else 'MISMATCH (expected 100)'}")
    got = int(df.filter(pl.col("in_fft_calib18")).height)
    print(f"  in_fft_calib18: {got}  ({len(calib_real)} real + {len(calib_gen)} generated raw videos on disk)")
    unprobed = df.filter(pl.col("codec").is_null())
    if len(unprobed):
        print(f"  !! {len(unprobed)} unreadable clips (ffprobe failed): {list(unprobed['clip_id'])}")
        bad_bench = unprobed.filter(pl.col("in_benchmark"))
        if len(bad_bench):
            print(f"     of which in_benchmark: {list(bad_bench['clip_id'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
