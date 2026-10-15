"""Pre-generate the counterfactual video artefacts for the E10 harness.

Three artefact families, all written under ``--out`` (default
``<final_dataset>/counterfactual``) so the canonical ``final_dataset/{raft,sam}``
caches stay untouched:

``raft_reverse/<source>/<folder>/flow_<stem>.mp4``
    the real RAFT flow video hue-rotated by 180°.  ``flow_to_image`` maps the
    flow direction onto the hue circle, so a 180° rotation swaps
    pink/magenta (ego moving forward) with green (ego moving backward) while
    leaving magnitude, texture, compression and frame count identical.

``raft_corrupt/static_gray_flow.mp4``
    a uniform mid-gray clip: a syntactically valid flow video that encodes no
    motion at all.

``sam_corrupt/object_{1..4}_clean.mp4``
    four black 512x512 clips in the exact shape SAM returns.

Idempotent (an existing output is skipped), single-threaded ffmpeg, 4 workers.

Usage::

    # on the pod
    python make_cf_videos.py \
        --raft-cache /mnt/vita/vita-students/users/scharffe/VLM-eval/final_dataset/raft \
        --out        /mnt/vita/vita-students/users/scharffe/VLM-eval/final_dataset/counterfactual

    # locally, against the ~500 cached flow clips in dataset/raft/
    .venv/bin/python src/agentic/counterfactual/make_cf_videos.py \
        --raft-cache dataset/raft --out /tmp/cf_probe --limit 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_RAFT_CACHE = REPO / "dataset" / "raft"

FLOW_SIZE = (960, 520)
FLOW_FPS = 24
FLOW_SECONDS = 5
SAM_CLIP_SIZE = 512
N_SAM_CLIPS = 4


def _run(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", *args],
                   check=True, capture_output=True)


def probe(path: Path) -> dict:
    """width / height / avg fps / nb_frames of a video (best effort)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
         "-count_packets", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, _, den = s.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return {"width": s["width"], "height": s["height"], "fps": round(fps, 3),
            "frames": int(s.get("nb_read_packets", 0))}


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------


def _reverse_one(job: tuple[str, str]) -> tuple[str, str]:
    """hue=h=180 on one flow video.  Returns (status, output path)."""
    src, dst = Path(job[0]), Path(job[1])
    if dst.exists() and dst.stat().st_size > 0:
        return ("skipped", str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    try:
        _run(["-i", str(src), "-vf", "hue=h=180",
              "-c:v", "libx264", "-crf", "18", "-preset", "fast",
              "-pix_fmt", "yuv420p", str(tmp)])
        tmp.replace(dst)  # atomic: a killed job never leaves a truncated clip
        return ("written", str(dst))
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        return (f"FAILED {e.stderr.decode('utf8', 'replace')[:200]}", str(dst))


def make_static_gray(out_root: Path, size=FLOW_SIZE, fps=FLOW_FPS,
                     seconds=FLOW_SECONDS) -> Path:
    out_dir = out_root / "raft_corrupt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "static_gray_flow.mp4"
    if not (out.exists() and out.stat().st_size > 0):
        w, h = size
        _run(["-f", "lavfi", "-i", f"color=c=0x808080:s={w}x{h}:d={seconds}:r={fps}",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
    return out


def make_black_clips(out_root: Path, fps=FLOW_FPS, seconds=FLOW_SECONDS) -> list[Path]:
    out_dir = out_root / "sam_corrupt"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(1, N_SAM_CLIPS + 1):
        out = out_dir / f"object_{i}_clean.mp4"
        if not (out.exists() and out.stat().st_size > 0):
            _run(["-f", "lavfi",
                  "-i", f"color=c=black:s={SAM_CLIP_SIZE}x{SAM_CLIP_SIZE}:"
                        f"d={seconds}:r={fps}",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
        clips.append(out)
    return clips


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raft-cache", type=Path, default=DEFAULT_RAFT_CACHE,
                    help="existing RAFT flow cache (<root>/raft on the PVC)")
    ap.add_argument("--out", type=Path, required=True,
                    help="counterfactual artefact root (--cf-root of the harness)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="only convert the first N flow videos (smoke test)")
    ap.add_argument("--skip-reverse", action="store_true")
    args = ap.parse_args()

    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    flows = sorted(p for p in args.raft_cache.rglob("*.mp4") if p.is_file())
    if args.limit:
        flows = flows[: args.limit]
    print(f"RAFT cache : {args.raft_cache}  ({len(flows)} flow videos"
          f"{' (limited)' if args.limit else ''})")
    print(f"Output root: {out_root}")

    # size the static-gray clip like the real flow videos when we can
    size, fps = FLOW_SIZE, FLOW_FPS
    if flows:
        try:
            meta = probe(flows[0])
            size = (meta["width"], meta["height"])
            fps = meta["fps"] or FLOW_FPS
            print(f"probe {flows[0].name}: {meta}")
        except Exception as e:  # ffprobe missing / odd file
            print(f"probe failed ({e}); falling back to {size}@{fps}")

    gray = make_static_gray(out_root, size=size, fps=fps)
    print(f"raft_corrupt : {gray}")
    blacks = make_black_clips(out_root, fps=fps)
    print(f"sam_corrupt  : {len(blacks)} clips in {blacks[0].parent}")

    if args.skip_reverse:
        return 0

    jobs = [(str(src), str(out_root / "raft_reverse" / src.relative_to(args.raft_cache)))
            for src in flows]
    if not jobs:
        print("no flow videos found — nothing to reverse")
        return 0

    written = skipped = failed = 0
    with Pool(processes=args.workers) as pool:
        for i, (status, dst) in enumerate(pool.imap_unordered(_reverse_one, jobs, chunksize=4), 1):
            if status == "written":
                written += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"  [FAIL] {dst}: {status}", file=sys.stderr)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  written={written} skipped={skipped} failed={failed}",
                      flush=True)

    print(f"raft_reverse : {written} written, {skipped} already present, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
