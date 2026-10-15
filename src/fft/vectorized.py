"""
Bit-exact vectorised reimplementation of the agent's per-frame FFT metrics.

The reference implementation (``src/agentic/fft/compute_fft.py``) is correct but
spends almost all of its time in three Python loops that each touch the whole
frame:

* ``compute_radial_profile`` -- ``np.add.at`` (unbuffered ufunc, ~50x slower
  than the equivalent ``np.bincount``);
* ``compute_anisotropy`` -- 72 angular bins x 3 full-frame boolean ops + a
  boolean extraction, i.e. ~290 passes over ~900k pixels per frame;
* ``detect_periodic_artifacts`` -- ``for ri in range(max_r)`` with a full-frame
  ``r == ri`` compare and a masked assignment, i.e. ~700 more passes.

This module replaces all three with cached index maps + ``np.bincount`` /
gather, and additionally shares the single windowed FFT that
``analyze_frame_fft`` and ``detect_periodic_artifacts`` were each computing
separately (they build the identical ``log1p(abs(fftshift(fft2(gray*window))))``).

Bit-exactness
-------------
Every number produced here is bit-identical to the reference, not merely close.
The three properties that make that possible were verified on this numpy
(2.2.6) and are asserted by ``selftest()``:

1. ``np.bincount(idx, weights=w)`` accumulates in exactly the same order as
   ``np.add.at(out, idx, w)``, so the float64 radial profile is identical.
2. ``np.fft.fft2`` of a float32 array yields complex64, so the anisotropy
   magnitude is float32 -- and ``np.mean`` of a float32 array uses pairwise
   summation whose result depends on the element order and count.  We therefore
   do *not* accumulate the angular bins with ``bincount`` (that would promote to
   float64 and change the answer in the 6th decimal); instead we gather the ring
   pixels into one array sorted by angular bin -- a *stable* sort, so within a
   bin the pixels keep their C-order sequence -- and call ``np.mean`` on each
   contiguous slice.  ``np.mean`` on such a slice is bit-identical to
   ``np.mean`` on the freshly extracted ``magnitude[combined]`` of the
   reference, because both see the same dtype, length, order and contiguity.
3. Building the ``expected`` spectrum by a lookup-table gather ``lut[r]``
   reproduces the reference's per-radius masked assignment exactly, since both
   copy the same float64 ``radial[ri]`` value into every pixel of radius ``ri``
   (and leave ``r >= max_r`` at 0).

The geometry (window, radius map, ring/angle index maps) depends only on the
frame shape, so it is cached per ``(h, w)``.
"""

from __future__ import annotations

import numpy as np

# Angular bins used by compute_anisotropy.
N_ANGLE_BINS = 72
# detect_periodic_artifacts' default threshold_std.
DEFAULT_THRESHOLD_STD = 4.0


class Geometry:
    """Frame-shape-dependent index maps, built once per ``(h, w)``."""

    __slots__ = (
        "h", "w", "cy", "cx", "max_r",
        "window",           # float64 (h, w) hanning outer product
        "r_int",            # int32 (h, w) truncated radius map
        "prof_mask",        # bool (h, w): r < max_r
        "prof_bins",        # int32 (n_prof,) radius of each masked pixel
        "prof_counts",      # float64 (max_r,) pixels per radius, zeros -> 1
        "lut_len",          # int: r.max() + 1
        "center_mask",      # bool (h, w): r > 5
        "aniso_idx",        # int64 (n_ring,) flat indices, sorted by angular bin
        "aniso_starts",     # int64 (73,) slice bounds into aniso_idx
    )

    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        cy, cx = h // 2, w // 2
        self.cy, self.cx = cy, cx
        max_r = min(cy, cx)
        self.max_r = max_r

        # --- analyze_frame_fft / detect_periodic_artifacts window -----------
        self.window = np.outer(np.hanning(h), np.hanning(w))

        # --- compute_radial_profile ----------------------------------------
        # Reference: r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int) over ogrid.
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        self.r_int = r.astype(np.int32)
        self.lut_len = int(r.max()) + 1

        prof_mask = r < max_r
        self.prof_mask = prof_mask
        self.prof_bins = r[prof_mask].astype(np.int32)
        counts = np.bincount(self.prof_bins, minlength=max_r).astype(np.float64)
        counts[counts == 0] = 1  # reference: counts[counts == 0] = 1
        self.prof_counts = counts

        # --- detect_periodic_artifacts -------------------------------------
        self.center_mask = r > 5

        # --- compute_anisotropy ---------------------------------------------
        # Reference uses mgrid (full arrays) and an *untruncated* float radius.
        Y, X = np.mgrid[:h, :w]
        R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        ring = (R >= max_r * 0.3) & (R <= max_r * 0.9)
        angles = np.arctan2(Y - cy, X - cx)  # (-pi, pi]
        del Y, X, R

        bin_edges = np.linspace(-np.pi, np.pi, N_ANGLE_BINS + 1)
        # searchsorted(..., 'right') - 1 gives the unique i with
        # bin_edges[i] <= angle < bin_edges[i+1] -- exactly the reference's
        # `(angles >= bin_edges[i]) & (angles < bin_edges[i+1])`.  An angle of
        # exactly +pi lands at i == N_ANGLE_BINS and is dropped by both.
        bidx = np.searchsorted(bin_edges, angles, side="right") - 1
        del angles

        valid = ring & (bidx >= 0) & (bidx < N_ANGLE_BINS)
        flat = np.flatnonzero(valid)  # ascending -> C order
        bins = bidx.ravel()[flat].astype(np.int32)
        del ring, bidx, valid

        # Stable sort keeps the C-order sequence inside each bin, which is what
        # makes the per-slice np.mean bit-identical to the reference.
        order = np.argsort(bins, kind="stable")
        self.aniso_idx = flat[order]
        counts_a = np.bincount(bins, minlength=N_ANGLE_BINS)
        self.aniso_starts = np.concatenate(([0], np.cumsum(counts_a)))


_GEOM_CACHE: dict[tuple[int, int], Geometry] = {}
_GEOM_CACHE_MAX = 6


def get_geometry(h: int, w: int) -> Geometry:
    key = (h, w)
    g = _GEOM_CACHE.get(key)
    if g is None:
        if len(_GEOM_CACHE) >= _GEOM_CACHE_MAX:
            _GEOM_CACHE.clear()  # ~17 MB per entry; a clip run rarely mixes many
        g = Geometry(h, w)
        _GEOM_CACHE[key] = g
    return g


# --------------------------------------------------------------------------
# Per-frame metrics
# --------------------------------------------------------------------------


def radial_profile(log_mag: np.ndarray, g: Geometry) -> np.ndarray:
    """Vectorised ``compute_fft.compute_radial_profile`` (bit-identical)."""
    vals = log_mag[g.prof_mask]
    prof = np.bincount(g.prof_bins, weights=vals, minlength=g.max_r)
    prof /= g.prof_counts
    return prof


def spectral_metrics(radial: np.ndarray) -> tuple[float, float]:
    """The ``hf_ratio`` / ``spectral_slope`` half of ``analyze_frame_fft``.

    Copied verbatim from the reference (it is already vectorised); only the
    radial profile in front of it was the bottleneck.
    """
    n_bins = len(radial)
    high_start = int(n_bins * 0.5)

    high_freq_energy = np.mean(radial[high_start:]) if high_start < n_bins else 0
    total_energy = np.mean(radial[1:])  # skip DC
    hf_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

    valid = radial[1:] > 0
    if np.sum(valid) > 10:
        freqs = np.arange(1, n_bins)
        log_freqs = np.log(freqs[valid])
        log_power = np.log(radial[1:][valid])
        spectral_slope = np.polyfit(log_freqs, log_power, 1)[0]
    else:
        spectral_slope = 0.0

    return float(hf_ratio), float(spectral_slope)


def periodic_anomaly_ratio(
    log_mag: np.ndarray,
    radial: np.ndarray,
    g: Geometry,
    threshold_std: float = DEFAULT_THRESHOLD_STD,
) -> float:
    """Vectorised ``detect_periodic_artifacts`` -> ``anomaly_ratio``.

    Reuses the caller's ``log_mag`` / ``radial``: the reference recomputes both
    from scratch with the identical expression, so sharing them is exact.
    """
    lut = np.zeros(g.lut_len, dtype=np.float64)
    lut[: g.max_r] = radial            # radii >= max_r stay 0, as in the reference
    expected = lut[g.r_int]
    residual = log_mag - expected
    residual_valid = residual[g.center_mask]

    mean_res = np.mean(residual_valid)
    std_res = np.std(residual_valid)
    if std_res > 0:
        anomalous = np.sum(residual_valid > mean_res + threshold_std * std_res)
        return float(anomalous / len(residual_valid))
    return 0.0


def anisotropy(gray: np.ndarray, g: Geometry) -> tuple[float, float, float]:
    """Vectorised ``compute_anisotropy`` -> ``(anisotropy, peak_ratio, axis_ratio)``.

    Returns the same 4-decimal rounded floats as the reference.
    """
    # float32 in -> complex64 out -> float32 magnitude (numpy >= 2.0).
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    vals = magnitude.ravel()[g.aniso_idx]
    starts = g.aniso_starts
    radial_energy = np.zeros(N_ANGLE_BINS)
    for i in range(N_ANGLE_BINS):
        s, e = starts[i], starts[i + 1]
        if e > s:
            radial_energy[i] = vals[s:e].mean()

    if radial_energy.max() < 1e-10:
        return 0.0, 0.0, 0.0

    aniso = float(radial_energy.std() / (radial_energy.mean() + 1e-10))
    median_e = np.median(radial_energy)
    peak_ratio = float(radial_energy.max() / (median_e + 1e-10))

    def angle_to_bin(deg):
        rad = deg * np.pi / 180
        return int((rad + np.pi) / (2 * np.pi) * N_ANGLE_BINS) % N_ANGLE_BINS

    axis_bins = [angle_to_bin(a) for a in (0, 45, 90, 135, -180, -135, -90, -45)]
    non_axis_bins = [i for i in range(N_ANGLE_BINS) if i not in axis_bins]
    axis_energy_mean = np.mean(radial_energy[axis_bins]) if axis_bins else 0
    non_axis_energy_mean = np.mean(radial_energy[non_axis_bins]) if non_axis_bins else 1
    axis_ratio = float(axis_energy_mean / (non_axis_energy_mean + 1e-10))

    return round(aniso, 4), round(peak_ratio, 4), round(axis_ratio, 4)


def frame_metrics(gray: np.ndarray, skip_periodic: bool = False) -> dict:
    """All per-frame metrics for one float64 grayscale frame.

    Returns ``{hf_ratio, spectral_slope, anisotropy, peak_ratio, axis_ratio,
    anomaly_ratio}``; ``anomaly_ratio`` is ``None`` when ``skip_periodic``.
    """
    h, w = gray.shape
    g = get_geometry(h, w)

    # One windowed FFT shared by analyze_frame_fft and detect_periodic_artifacts.
    windowed = gray * g.window
    log_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(windowed))))
    radial = radial_profile(log_mag, g)

    hf_ratio, slope = spectral_metrics(radial)
    aniso, peak, axis = anisotropy(gray, g)
    anomaly = None if skip_periodic else periodic_anomaly_ratio(log_mag, radial, g)

    return {
        "hf_ratio": hf_ratio,
        "spectral_slope": slope,
        "anisotropy": aniso,
        "peak_ratio": peak,
        "axis_ratio": axis,
        "anomaly_ratio": anomaly,
    }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _reference_angular_energy(gray: np.ndarray) -> np.ndarray:
    """The reference's raw (unrounded) 72-bin angular energy vector.

    Lifted verbatim out of ``compute_fft.compute_anisotropy`` so the self-test
    can compare the quantity *before* the 4-decimal rounding that would
    otherwise hide sub-5e-5 drift.
    """
    h, w = gray.shape
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[:h, :w]
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = min(cy, cx)
    ring_mask = (R >= max_r * 0.3) & (R <= max_r * 0.9)
    angles = np.arctan2(Y - cy, X - cx)
    bin_edges = np.linspace(-np.pi, np.pi, N_ANGLE_BINS + 1)
    out = np.zeros(N_ANGLE_BINS)
    for i in range(N_ANGLE_BINS):
        angle_mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        combined = ring_mask & angle_mask
        if combined.any():
            out[i] = magnitude[combined].mean()
    return out


def _fast_angular_energy(gray: np.ndarray) -> np.ndarray:
    g = get_geometry(*gray.shape)
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))))
    vals = magnitude.ravel()[g.aniso_idx]
    out = np.zeros(N_ANGLE_BINS)
    for i in range(N_ANGLE_BINS):
        s, e = g.aniso_starts[i], g.aniso_starts[i + 1]
        if e > s:
            out[i] = vals[s:e].mean()
    return out


def selftest(shape=(180, 320), seed=0, verbose=True) -> bool:
    """Assert bit-equality against the reference on one synthetic frame."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fft.common import load_agentic_fft

    ref = load_agentic_fft()
    rng = np.random.default_rng(seed)
    gray = (rng.random(shape) * 255).astype(np.uint8).astype(np.float64)

    # Unrounded angular energies -- the strictest of the checks.
    ae_ref = _reference_angular_energy(gray)
    ae_fast = _fast_angular_energy(gray)
    ae_same = np.array_equal(ae_ref, ae_fast)
    if verbose:
        print(
            f"  {'angular_energy':16s} 72 bins, unrounded, maxabs diff="
            f"{np.abs(ae_ref - ae_fast).max():.3e} {'exact' if ae_same else 'DIFFER'}"
        )

    # Unrounded radial profile.
    win = np.outer(np.hanning(shape[0]), np.hanning(shape[1]))
    lm = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray * win))))
    rp_ref = ref.compute_radial_profile(lm)
    rp_fast = radial_profile(lm, get_geometry(*shape))
    rp_same = np.array_equal(rp_ref, rp_fast)
    if verbose:
        print(
            f"  {'radial_profile':16s} {len(rp_ref)} bins, unrounded, maxabs diff="
            f"{np.abs(rp_ref - rp_fast).max():.3e} {'exact' if rp_same else 'DIFFER'}"
        )

    r_ref = ref.analyze_frame_fft(gray)
    r_ani = ref.compute_anisotropy(gray)
    r_per = ref.detect_periodic_artifacts(gray)
    got = frame_metrics(gray)

    checks = [
        ("hf_ratio", r_ref["hf_ratio"], got["hf_ratio"]),
        ("spectral_slope", r_ref["spectral_slope"], got["spectral_slope"]),
        ("anisotropy", r_ani["anisotropy"], got["anisotropy"]),
        ("peak_ratio", r_ani["peak_ratio"], got["peak_ratio"]),
        ("axis_ratio", r_ani["axis_ratio"], got["axis_ratio"]),
        ("anomaly_ratio", r_per["anomaly_ratio"], got["anomaly_ratio"]),
    ]
    ok = ae_same and rp_same
    for name, a, b in checks:
        same = a == b
        ok &= same
        if verbose:
            print(f"  {name:16s} ref={a!r:24s} fast={b!r:24s} {'exact' if same else 'DIFFER'}")
    return bool(ok)


if __name__ == "__main__":
    allok = True
    # Cover even/odd dimensions and the real frame shapes in the study.
    for shape in [(180, 320), (181, 321), (128, 128), (432, 768), (704, 1280), (720, 1280)]:
        print(f"vectorized.selftest(shape={shape}):")
        r = selftest(shape=shape)
        allok &= r
        print("  ->", "PASS" if r else "FAIL")
    print("\nALL PASS" if allok else "\nFAILURES PRESENT")
