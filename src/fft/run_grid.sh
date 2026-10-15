#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# FFT robustness grid: re-encode the 600-clip grid sample under every
# perturbation condition and extract FFT features from each.
#
#   bash src/fft/run_grid.sh --dry-run          # print the commands, run nothing
#   bash src/fft/run_grid.sh                    # everything (hours)
#   bash src/fft/run_grid.sh --only crf28,fps5  # a subset of the conditions
#   bash src/fft/run_grid.sh --skip-encode      # features only, re-uses the mp4s
#   bash src/fft/run_grid.sh --workers 6        # ffmpeg / feature workers
#
# Prerequisites
#   .venv/bin/python src/fft/manifest.py
#   .venv/bin/python src/fft/splits.py
#   .venv/bin/python src/fft/grid_sample.py     -> results/fft/grid_manifest.parquet
#
# Why --manifest and not --filter: perturb.py / extract_features.py both accept a
# manifest parquet, and grid_sample.py writes one holding exactly the 600 grid
# clips in manifest schema.  Passing 600 clip ids through --filter would mean a
# ~30 kB command line; pointing --manifest at the subset is the same selection,
# readable, and re-runnable.
#
# Every step is idempotent: perturb.py skips an existing non-empty output, and
# extract_features.py resumes on (clip_id, condition, max_frames, frame_offset,
# impl, skip_periodic).  Interrupt and re-run at will.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
GRID_MANIFEST="results/fft/grid_manifest.parquet"
WORKERS=4
ENCODE_WORKERS=4
DRY=0
SKIP_ENCODE=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --skip-encode) SKIP_ENCODE=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --workers) WORKERS="$2"; ENCODE_WORKERS="$2"; shift 2 ;;
    --manifest) GRID_MANIFEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Grid conditions, in the order they are cheapest to inspect.  Names must match
# the registry in src/fft/perturb.py (check with: $PY src/fft/perturb.py --list).
CONDITIONS=(
  n10_crf18
  crf18 crf28 crf35 crf45
  res960 res640 res480
  fps5 fps24
  codec_mpeg4 codec_hevc codec_vp9
  frames25
)
# Fingerprint swaps: these re-encode from the RAW clip (not from n10) and apply
# only to one class each -- a generated clip wearing the real pool's mpeg4
# 1280x720@10 profile, and a real clip wearing the generated h264 1280x704@24
# profile.  They are the direct test of "is the detector reading the container?".
SWAPS=("swap_gen2real:label=generated" "swap_real2gen:label=real")

MAX_FRAMES=20          # the grid is scored at 20 frames/clip (5x cheaper than 60)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run() {
  printf '  $ %s\n' "$*"
  if [[ $DRY -eq 0 ]]; then "$@"; fi
}
wanted() {
  [[ -z "$ONLY" ]] && return 0
  [[ ",$ONLY," == *",$1,"* ]]
}

if [[ ! -f "$GRID_MANIFEST" ]]; then
  echo "missing $GRID_MANIFEST -- run: $PY src/fft/grid_sample.py" >&2
  exit 1
fi

say "grid manifest: $GRID_MANIFEST"
run "$PY" src/fft/perturb.py --list

# ---------------------------------------------------------------------------
# 0. the shared baseline: n10 (1280x704 @ 10 fps, 50 frames, x264 crf 23).
#    Every grid condition is applied ON TOP of this, so the codec / resolution /
#    frame-rate fingerprint is already gone before the sweep starts.
# ---------------------------------------------------------------------------
if [[ $SKIP_ENCODE -eq 0 ]] && wanted n10; then
  say "encode n10 (baseline for every grid condition)"
  run "$PY" src/fft/perturb.py --condition n10 --manifest "$GRID_MANIFEST" --workers "$ENCODE_WORKERS"
fi

# ---------------------------------------------------------------------------
# 1. encode every grid condition
# ---------------------------------------------------------------------------
if [[ $SKIP_ENCODE -eq 0 ]]; then
  for c in "${CONDITIONS[@]}"; do
    wanted "$c" || continue
    say "encode $c"
    run "$PY" src/fft/perturb.py --condition "$c" --manifest "$GRID_MANIFEST" --workers "$ENCODE_WORKERS"
  done
  for spec in "${SWAPS[@]}"; do
    c="${spec%%:*}"; filt="${spec#*:}"
    wanted "$c" || continue
    say "encode $c  (from raw, $filt only)"
    run "$PY" src/fft/perturb.py --condition "$c" --manifest "$GRID_MANIFEST" \
        --filter "$filt" --workers "$ENCODE_WORKERS"
  done
fi

# ---------------------------------------------------------------------------
# 2. features for the two references (raw and n10) at the grid's frame budget.
#    robustness.py compares every condition against features_n10_f20, so this
#    pair has to exist even though features_{raw,n10}_f60 already do.
# ---------------------------------------------------------------------------
for c in raw n10; do
  wanted "$c" || continue
  say "features $c (f$MAX_FRAMES)"
  run "$PY" src/fft/extract_features.py --manifest "$GRID_MANIFEST" --condition "$c" \
      --max-frames "$MAX_FRAMES" --impl fast --fast --workers "$WORKERS" \
      --out "results/fft/features_${c}_f${MAX_FRAMES}.parquet"
done

# ---------------------------------------------------------------------------
# 3. features for every perturbed condition
# ---------------------------------------------------------------------------
for c in "${CONDITIONS[@]}" swap_gen2real swap_real2gen; do
  wanted "$c" || continue
  say "features $c (f$MAX_FRAMES)"
  run "$PY" src/fft/extract_features.py --manifest "$GRID_MANIFEST" --condition "$c" \
      --max-frames "$MAX_FRAMES" --impl fast --fast --workers "$WORKERS" \
      --out "results/fft/features_${c}_f${MAX_FRAMES}.parquet"
done

# ---------------------------------------------------------------------------
# 4. frame-sampling sensitivity, all on n10 (no new encodes).
#    How many frames the tool averages over, and which frames it lands on, are
#    free parameters of the agent's tool -- 60 frames, grid starting at frame 0.
# ---------------------------------------------------------------------------
for f in 5 10 20 60; do
  wanted "frames$f" || wanted n10 || continue
  say "features n10 @ ${f} frames"
  run "$PY" src/fft/extract_features.py --manifest "$GRID_MANIFEST" --condition n10 \
      --max-frames "$f" --impl fast --fast --workers "$WORKERS" \
      --out "results/fft/features_n10_f${f}.parquet"
done
for f in 20 60; do
  wanted "offset$f" || wanted n10 || continue
  say "features n10 @ ${f} frames, sampling grid shifted by 1"
  run "$PY" src/fft/extract_features.py --manifest "$GRID_MANIFEST" --condition n10 \
      --max-frames "$f" --frame-offset 1 --impl fast --fast --workers "$WORKERS" \
      --out "results/fft/features_n10_f${f}_o1.parquet"
done

# ---------------------------------------------------------------------------
# 5. sanity check the encodes, then analyse
# ---------------------------------------------------------------------------
say "check the encoded outputs"
for c in n10 "${CONDITIONS[@]}" swap_gen2real swap_real2gen; do
  wanted "$c" || continue
  run "$PY" src/fft/perturb.py --condition "$c" --check --workers "$ENCODE_WORKERS"
done

say "analyse"
run "$PY" src/fft/robustness.py --grid results/fft/grid_sample.parquet \
    --features-dir results/fft --out results/fft/robustness/

echo
echo "done. Note that features_n10_f60.parquet also holds the full 5510-clip run,"
echo "so the grid rows simply resume into it; robustness.py restricts to the grid ids."
