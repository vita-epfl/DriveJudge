"""
Build perturbed copies of the clips for the FFT robustness study.

Every condition is an ffmpeg re-encode.  Outputs go to
``dataset/fft_conditions/<condition>/<clip_id>.mp4`` (clip_id contains slashes,
so the tree mirrors the manifest).  Runs are idempotent: an existing non-empty
output is skipped.

Condition registry
------------------
Normalisation conditions (input = the raw clip from the manifest):

    n10             1280x704 @ 10 fps, 50 frames, h264 crf 23   <- the common grid baseline
    n10_crf18       same, crf 18
    swap_gen2real   re-encode a *generated* clip with the real pool's codec
                    profile (mpeg4 q5, 1280x720 @ 10 fps, 50 frames)
    swap_real2gen   re-encode a *real* clip with the generated pool's profile
                    (h264 crf 23, 1280x704 @ 24 fps, 121 frames)

Grid conditions (applied ON TOP of n10 -- the n10 output is used as input when
it exists, otherwise the raw clip, and the normalisation is re-applied so the
result is the same either way):

    crf18 crf28 crf35 crf45      compression sweep
    res960 res640 res480         resolution sweep (aspect-preserving)
    fps5 fps24                   frame-rate sweep (fps24 duplicates frames)
    codec_mpeg4 codec_hevc codec_vp9
    frames25                     half the clip length

Usage
-----
    .venv/bin/python src/fft/perturb.py --condition n10 --filter in_realgen_v1=true
    .venv/bin/python src/fft/perturb.py --condition crf35 --filter generator=dd --workers 12
    .venv/bin/python src/fft/perturb.py --condition n10 --check
    .venv/bin/python src/fft/perturb.py --list
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft.common import CONDITIONS_DIR, DEFAULT_MANIFEST, rel_to_repo  # noqa: E402
from fft.extract_features import build_filter  # noqa: E402
from fft.manifest import probe  # noqa: E402

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# Baseline geometry shared by the grid conditions.
_N10_VF = "scale=1280:704:flags=bicubic,fps=10"
_X264 = ["-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p", "-an"]


def _grid_x264(crf: int, vf: str = _N10_VF, frames: int | None = 50) -> list[str]:
    args = ["-vf", vf]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    return args + _X264 + ["-crf", str(crf)]


CONDITIONS: dict[str, dict] = {
    # ---- normalisation (from raw) ----------------------------------------
    "n10": {
        "base": "raw",
        "args": ["-vf", _N10_VF, "-frames:v", "50", *_X264, "-crf", "23"],
        "encoder": "libx264",
    },
    "n10_crf18": {
        "base": "raw",
        "args": ["-vf", _N10_VF, "-frames:v", "50", *_X264, "-crf", "18"],
        "encoder": "libx264",
    },
    "swap_gen2real": {
        "base": "raw",
        "args": [
            "-vf", "scale=1280:720:flags=bicubic,fps=10", "-frames:v", "50",
            "-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p", "-an",
        ],
        "encoder": "mpeg4",
        "intended_label": "generated",
    },
    "swap_real2gen": {
        "base": "raw",
        "args": [
            "-vf", "scale=1280:704:flags=bicubic,fps=24", "-frames:v", "121",
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
        ],
        "encoder": "libx264",
        "intended_label": "real",
    },
    # ---- grid (on top of n10) --------------------------------------------
    "crf18": {"base": "n10", "args": _grid_x264(18), "encoder": "libx264"},
    "crf28": {"base": "n10", "args": _grid_x264(28), "encoder": "libx264"},
    "crf35": {"base": "n10", "args": _grid_x264(35), "encoder": "libx264"},
    "crf45": {"base": "n10", "args": _grid_x264(45), "encoder": "libx264"},
    "res960": {
        "base": "n10",
        "args": _grid_x264(23, "scale=960:528:flags=bicubic,fps=10"),
        "encoder": "libx264",
    },
    "res640": {
        "base": "n10",
        "args": _grid_x264(23, "scale=640:352:flags=bicubic,fps=10"),
        "encoder": "libx264",
    },
    "res480": {
        "base": "n10",
        "args": _grid_x264(23, "scale=480:264:flags=bicubic,fps=10"),
        "encoder": "libx264",
    },
    # No -frames:v cap for the fps sweep: the clip keeps its 5 s duration, so
    # the frame count follows the rate (25 @ 5 fps, 120 @ 24 fps).
    "fps5": {
        "base": "n10",
        "args": _grid_x264(23, "scale=1280:704:flags=bicubic,fps=5", frames=None),
        "encoder": "libx264",
    },
    "fps24": {
        "base": "n10",
        "args": _grid_x264(23, "scale=1280:704:flags=bicubic,fps=24", frames=None),
        "encoder": "libx264",
    },
    "codec_mpeg4": {
        "base": "n10",
        "args": [
            "-vf", _N10_VF, "-frames:v", "50",
            "-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p", "-an",
        ],
        "encoder": "mpeg4",
    },
    "codec_hevc": {
        "base": "n10",
        "args": [
            "-vf", _N10_VF, "-frames:v", "50",
            "-c:v", "libx265", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
            "-x265-params", "log-level=error",
            "-tag:v", "hvc1",
        ],
        "encoder": "libx265",
    },
    "codec_vp9": {
        "base": "n10",
        "args": [
            "-vf", _N10_VF, "-frames:v", "50",
            "-c:v", "libvpx-vp9", "-crf", "33", "-b:v", "0",
            "-pix_fmt", "yuv420p", "-an",
        ],
        "encoder": "libvpx-vp9",
    },
    "frames25": {"base": "n10", "args": _grid_x264(23, frames=25), "encoder": "libx264"},
}


# --------------------------------------------------------------------------
# ffmpeg capability check
# --------------------------------------------------------------------------


def available_encoders() -> set[str]:
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"cannot run ffmpeg: {exc}")
    names = set()
    for line in res.stdout.splitlines():
        parts = line.split()
        # " V....D libx264              libx264 H.264 ..."
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return names


def require_encoder(condition: str) -> None:
    enc = CONDITIONS[condition]["encoder"]
    have = available_encoders()
    if enc not in have:
        raise SystemExit(
            f"condition '{condition}' needs the ffmpeg encoder '{enc}', which this ffmpeg build "
            f"does not provide.\n"
            f"Check with `ffmpeg -encoders | grep {enc}` and install an ffmpeg built with it "
            f"(e.g. the ubuntu 'ffmpeg' package carries libx264/libx265/libvpx-vp9/mpeg4)."
        )
    print(f"encoder '{enc}' available")


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

_CFG: dict = {}


def _init(condition: str) -> None:
    _CFG["condition"] = condition
    _CFG["args"] = CONDITIONS[condition]["args"]


def _encode(task: tuple[str, str, str]) -> tuple[str, str, str | None]:
    """``(clip_id, src, dst)`` -> ``(clip_id, status, message)``."""
    clip_id, src, dst = task
    dst_p = Path(dst)
    if dst_p.exists() and dst_p.stat().st_size > 0:
        return clip_id, "skipped", None
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_name(dst_p.name + ".tmp.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-i", src, *_CFG["args"], "-threads", "1", str(tmp)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        tmp.unlink(missing_ok=True)
        return clip_id, "error", f"{type(exc).__name__}: {exc}"
    if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        msg = (res.stderr or "ffmpeg failed").strip().replace("\n", " ")[:300]
        tmp.unlink(missing_ok=True)
        return clip_id, "error", msg
    tmp.replace(dst_p)
    return clip_id, "ok", None


def _probe_out(task: tuple[str, str]) -> tuple[str, dict]:
    clip_id, path = task
    return clip_id, probe(path)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def select_clips(args) -> pl.DataFrame:
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}  (run src/fft/manifest.py first)")
    man = pl.read_parquet(args.manifest)
    print(f"manifest: {rel_to_repo(args.manifest)}  ({len(man)} clips)")
    if args.filter:
        man = man.filter(build_filter(args.filter))
        print(f"filter  : {args.filter}  -> {len(man)} clips")
    if args.limit:
        man = man.head(args.limit)
    return man


def source_for(clip_id: str, raw_path: str, condition: str) -> tuple[str, str]:
    """Input file for one clip + where it came from ('raw' or 'n10')."""
    base = CONDITIONS[condition]["base"]
    if base != "raw":
        cand = CONDITIONS_DIR / base / f"{clip_id}.mp4"
        if cand.exists() and cand.stat().st_size > 0:
            return str(cand), base
    return raw_path, "raw"


def run_encode(args) -> int:
    condition = args.condition
    require_encoder(condition)
    spec = CONDITIONS[condition]
    out_dir = CONDITIONS_DIR / condition
    print(f"condition: {condition}  (base={spec['base']})")
    print(f"ffmpeg   : ffmpeg -y -loglevel error -threads 1 -i <in> {shlex.join(spec['args'])} <out>")
    print(f"outdir   : {rel_to_repo(out_dir)}")

    man = select_clips(args)
    if not len(man):
        print("nothing selected")
        return 0

    intended = spec.get("intended_label")
    if intended:
        wrong = man.filter(pl.col("label") != intended).height
        if wrong:
            print(f"NOTE: '{condition}' is intended for label='{intended}' clips; {wrong}/{len(man)} selected clips are not")

    tasks, base_counts = [], Counter()
    for row in man.select("clip_id", "path").iter_rows(named=True):
        src, base_used = source_for(row["clip_id"], row["path"], condition)
        base_counts[base_used] += 1
        tasks.append((row["clip_id"], src, str(out_dir / f"{row['clip_id']}.mp4")))
    print(f"inputs   : {dict(base_counts)}")

    t0 = time.perf_counter()
    counts: Counter = Counter()
    errors = []
    with Pool(args.workers, initializer=_init, initargs=(condition,), maxtasksperchild=100) as pool:
        for i, (clip_id, status, msg) in enumerate(pool.imap_unordered(_encode, tasks, chunksize=1), start=1):
            counts[status] += 1
            if status == "error":
                errors.append((clip_id, msg))
            if i % 100 == 0 or i == len(tasks):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(tasks)}  {el / 60:.1f} min  {dict(counts)}", flush=True)

    print(f"\ndone in {(time.perf_counter() - t0) / 60:.1f} min: {dict(counts)}")
    for clip_id, msg in errors[:10]:
        print(f"  ERROR {clip_id}: {msg}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")
    return 0


def run_check(args) -> int:
    condition = args.condition
    out_dir = CONDITIONS_DIR / condition
    if not out_dir.is_dir():
        raise SystemExit(f"no outputs for condition '{condition}' ({out_dir})")

    files = sorted(out_dir.rglob("*.mp4"))
    files = [f for f in files if not f.name.endswith(".tmp.mp4")]
    print(f"condition '{condition}': {len(files)} files under {rel_to_repo(out_dir)}")
    if not files:
        return 0

    # clip_id = the path under the condition dir minus the trailing ".mp4"
    # (str-slice rather than with_suffix: some stems contain dots).
    tasks = [(str(f.relative_to(out_dir))[:-4], str(f)) for f in files]
    hist: Counter = Counter()
    sizes = []
    bad = []
    with Pool(args.workers) as pool:
        for clip_id, info in pool.imap_unordered(_probe_out, tasks, chunksize=8):
            if info.get("probe_error"):
                bad.append((clip_id, info["probe_error"]))
                continue
            hist[(info["codec"], info["width"], info["height"], info["fps"], info["nframes"])] += 1
            if info["size_bytes"]:
                sizes.append(info["size_bytes"])

    print("\n(codec, width, height, fps, nframes) histogram:")
    for key, n in hist.most_common():
        print(f"  {n:6d}  codec={key[0]:<8} {key[1]}x{key[2]}  fps={key[3]}  frames={key[4]}")
    if sizes:
        sizes.sort()
        print(
            f"\nsize: mean {sum(sizes) / len(sizes) / 1e6:.2f} MB, "
            f"median {sizes[len(sizes) // 2] / 1e6:.2f} MB, "
            f"total {sum(sizes) / 1e9:.2f} GB"
        )
    if bad:
        print(f"\n!! {len(bad)} unreadable outputs:")
        for clip_id, msg in bad[:10]:
            print(f"  {clip_id}: {msg}")
    return 0


def run_list() -> int:
    have = available_encoders()
    print(f"{'condition':<16} {'base':<6} {'encoder':<12} {'avail':<6} args")
    for name, spec in CONDITIONS.items():
        enc = spec["encoder"]
        print(f"{name:<16} {spec['base']:<6} {enc:<12} {'yes' if enc in have else 'NO':<6} {shlex.join(spec['args'])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", help=f"one of: {', '.join(CONDITIONS)}")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--filter", default=None, help="polars expr string, or key=value[,key=value]")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="ffprobe the condition's outputs and print a histogram")
    ap.add_argument("--list", action="store_true", help="print the condition registry + encoder availability")
    args = ap.parse_args()

    if args.list:
        return run_list()
    if not args.condition:
        raise SystemExit("--condition is required (or use --list)")
    if args.condition not in CONDITIONS:
        raise SystemExit(f"unknown condition {args.condition!r}; known: {', '.join(CONDITIONS)}")
    if args.check:
        return run_check(args)
    return run_encode(args)


if __name__ == "__main__":
    raise SystemExit(main())
