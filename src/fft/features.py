"""
Flat FFT feature vector for one video.

Wraps the *agent's* per-frame FFT functions (``src/agentic/fft/compute_fft.py``)
and replicates the aggregation that ``process_video_fft`` performs on its
``summary`` dict -- without rendering the FFT spectrum video or writing a JSON
side-car, which is what makes ``process_video_fft`` far too slow to run over
thousands of clips.

Numerical equivalence
---------------------
With ``frame_offset=0`` the frame sampler here reproduces
``compute_fft.load_frames`` on its cv2 branch *exactly*: the same
``np.linspace(0, total-1, max_frames, dtype=int)`` index set, decoded by
sequentially walking the stream, converted with ``cv2.COLOR_BGR2GRAY`` and cast
to float64.  Frames are processed one at a time instead of being buffered in a
list (identical maths, ~60x less memory), so 12 workers fit comfortably in RAM.

The cluster runs also took the **cv2** branch (torchvision was not importable in
the agentic venv).  This was verified against
``dataset/dataset_final/results_fft/Qwen3VL_FFT_Anisotropy_Agent_results.json``:
cv2 reproduces the published anisotropy values to all four printed decimals,
whereas a torchvision-style float grayscale differs by ~2-5e-4.  There is
therefore no local-vs-cluster drift to correct for.

Two independent switches
------------------------
``impl``  selects *how* the per-frame metrics are computed:
            ``"exact"`` calls the agent's functions in ``compute_fft`` directly;
            ``"fast"`` (default) calls the bit-identical vectorised
            reimplementation in ``fft.vectorized``.  They produce the same
            numbers -- ``fast`` is simply ~14x quicker.
``fast``  (badly named, kept for CLI compatibility) selects *which* metrics are
            computed: ``True`` skips ``detect_periodic_artifacts`` entirely, so
            the ``anomaly_ratio_*`` / ``periodic_artifact_frames`` columns come
            back null.  Think of it as ``skip_periodic``.
"""

from __future__ import annotations

# Pin the numeric backends to one thread *before* numpy is imported anywhere:
# this module is the first thing a pool worker touches, and we parallelise over
# clips, not inside numpy.
import os as _os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fft.common import load_agentic_fft  # noqa: E402
from fft import vectorized  # noqa: E402

# src/agentic/fft/compute_fft.py -- the exact per-frame functions the agent's
# get_frequency_analysis tool calls, loaded by file path (see load_agentic_fft).
_agentic_fft = load_agentic_fft()
analyze_frame_fft = _agentic_fft.analyze_frame_fft
compute_anisotropy = _agentic_fft.compute_anisotropy
detect_periodic_artifacts = _agentic_fft.detect_periodic_artifacts

IMPLS = ("fast", "exact")

cv2.setNumThreads(0)

# Keys returned by extract_features(), in order.  Callers build parquet schemas
# from this, so keep it stable.
FEATURE_KEYS = (
    "n_frames",
    "aniso_mean",
    "aniso_std",
    "aniso_max",
    "peak_ratio_mean",
    "axis_ratio_mean",
    "hf_ratio_mean",
    "hf_ratio_std",
    "hf_ratio_min",
    "hf_ratio_max",
    "spectral_slope_mean",
    "spectral_slope_std",
    "anomaly_ratio_mean",
    "anomaly_ratio_max",
    "periodic_artifact_frames",
    "width",
    "height",
    "error",
)

# Threshold the agent's tool prints alongside the anisotropy value.
ANISO_THRESHOLD = 0.048


def _empty_result(error: str | None = None) -> dict:
    row = {k: None for k in FEATURE_KEYS}
    row["error"] = error
    return row


def sample_indices(total: int, max_frames: int, frame_offset: int = 0) -> set:
    """Frame indices to analyse.

    Mirrors ``compute_fft.load_frames`` (cv2 branch) and then shifts the whole
    grid by ``frame_offset`` frames, clamped to the valid range.  With
    ``frame_offset == 0`` the returned set is bit-identical to the original.

    Clamping means a non-zero offset can collapse the last few indices onto the
    final frame, so ``n_frames`` may come back 1-2 short of ``max_frames``
    (e.g. total=121, max_frames=60, offset=3 -> 59 frames).  That is the same
    de-duplication ``load_frames`` already performs via its index ``set``.
    """
    if total <= 0:
        return set()
    if total > max_frames:
        base = np.linspace(0, total - 1, max_frames, dtype=int)
    else:
        base = np.arange(total, dtype=int)
    if frame_offset:
        base = np.clip(base + int(frame_offset), 0, total - 1)
    return set(int(i) for i in base)


def iter_gray_frames(video_path, max_frames: int = 60, frame_offset: int = 0):
    """Yield ``(index, float64 grayscale frame)`` for the sampled indices."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cv2 could not open {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_indices(total, max_frames, frame_offset)
        if not indices:
            return
        last = max(indices)
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if i in indices:
                yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            if i >= last:
                break
    finally:
        cap.release()


def _frame_metrics_exact(gray: np.ndarray, skip_periodic: bool) -> dict:
    """Per-frame metrics straight from the agent's own functions."""
    m = analyze_frame_fft(gray)
    a = compute_anisotropy(gray)
    return {
        "hf_ratio": m["hf_ratio"],
        "spectral_slope": m["spectral_slope"],
        "anisotropy": a["anisotropy"],
        "peak_ratio": a["peak_ratio"],
        "axis_ratio": a["axis_ratio"],
        "anomaly_ratio": None if skip_periodic else detect_periodic_artifacts(gray)["anomaly_ratio"],
    }


def extract_features(
    path,
    max_frames: int = 60,
    frame_offset: int = 0,
    fast: bool = False,
    impl: str = "fast",
) -> dict:
    """Aggregate FFT statistics for one video.

    Parameters
    ----------
    path        video file.
    max_frames  how many frames to sample uniformly (agent default: 60).
    frame_offset shift the uniform sampling grid by this many frames; used to
                probe the sensitivity of the metrics to which frames are seen.
    fast        skip :func:`detect_periodic_artifacts` (roughly halves the cost);
                the ``anomaly_ratio_*`` / ``periodic_artifact_frames`` columns
                come back null.  Really means ``skip_periodic`` -- it is
                orthogonal to ``impl``.
    impl        ``"fast"`` (default) uses the vectorised, bit-identical
                reimplementation in :mod:`fft.vectorized`; ``"exact"`` calls the
                agent's ``compute_fft`` functions directly.  Same numbers,
                ~14x difference in speed.

    Returns a flat dict with the keys in :data:`FEATURE_KEYS`.  Any failure is
    reported through the ``error`` key rather than raised.
    """
    path = Path(path)
    if impl not in IMPLS:
        return _empty_result(f"unknown impl {impl!r}, expected one of {IMPLS}")
    if not path.exists():
        return _empty_result(f"missing file: {path}")

    per_frame = vectorized.frame_metrics if impl == "fast" else _frame_metrics_exact

    aniso, peaks, axes = [], [], []
    hf, slopes, anomalies = [], [], []
    h = w = None

    try:
        for _, gray in iter_gray_frames(path, max_frames=max_frames, frame_offset=frame_offset):
            if h is None:
                h, w = gray.shape
            m = per_frame(gray, fast)
            hf.append(m["hf_ratio"])
            slopes.append(m["spectral_slope"])
            aniso.append(m["anisotropy"])
            peaks.append(m["peak_ratio"])
            axes.append(m["axis_ratio"])
            if not fast:
                anomalies.append(m["anomaly_ratio"])
    except Exception as exc:  # noqa: BLE001 -- one bad clip must not kill the pool
        return _empty_result(f"{type(exc).__name__}: {exc}")

    n = len(aniso)
    if n < 2:
        # Same guard as process_video_fft.
        return _empty_result(f"Could not load enough frames from {path.name} (got {n})")

    row = {
        "n_frames": n,
        "aniso_mean": float(np.mean(aniso)),
        "aniso_std": float(np.std(aniso)),
        "aniso_max": float(np.max(aniso)),
        "peak_ratio_mean": float(np.mean(peaks)),
        "axis_ratio_mean": float(np.mean(axes)),
        "hf_ratio_mean": float(np.mean(hf)),
        "hf_ratio_std": float(np.std(hf)),
        "hf_ratio_min": float(np.min(hf)),
        "hf_ratio_max": float(np.max(hf)),
        "spectral_slope_mean": float(np.mean(slopes)),
        "spectral_slope_std": float(np.std(slopes)),
        "anomaly_ratio_mean": float(np.mean(anomalies)) if anomalies else None,
        "anomaly_ratio_max": float(np.max(anomalies)) if anomalies else None,
        # process_video_fft: sum(1 for a in anomaly_ratios if a > 0.001)
        "periodic_artifact_frames": int(sum(1 for a in anomalies if a > 0.001)) if anomalies else None,
        "width": int(w),
        "height": int(h),
        "error": None,
    }
    return row


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Extract flat FFT features for one video")
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--frame-offset", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="skip detect_periodic_artifacts")
    ap.add_argument("--impl", choices=IMPLS, default="fast")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = extract_features(args.video, args.max_frames, args.frame_offset, args.fast, args.impl)
    out["impl"] = args.impl
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    print(json.dumps(out, indent=2))
