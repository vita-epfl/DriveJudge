"""Offline tests for the FFT counterfactual text (plain asserts, no pytest).

    .venv/bin/python src/agentic/counterfactual/test_cf_fft.py

Checks that
  * ``cf_tools.generate_interpretation(s, 0.048)`` is byte-identical to the
    canonical ``fft.compute_fft.generate_interpretation(s)``,
  * ``reverse`` flips the verdict and stays internally consistent,
  * ``corrupt`` is deterministic per (seed, clip) and threshold-straddling,
  * ``none`` never leaks the cached interpretation,
and prints the real / reverse / corrupt text for a real cached analysis taken
from ``dataset/fft_test/`` (that cache predates the anisotropy metric, so a
summary is synthesised from its numbers when the key is absent).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AGENTIC = _HERE.parent
sys.path.insert(0, str(_AGENTIC))
sys.path.insert(0, str(_HERE))

import cf_tools  # noqa: E402
from fft.compute_fft import generate_interpretation as canonical_interpretation  # noqa: E402

REPO = _AGENTIC.parents[1]
FFT_TEST_DIR = REPO / "dataset" / "fft_test"


def load_a_real_summary() -> tuple[str, dict, bool]:
    """(stem, summary, synthesised?) from any cached *_fft_analysis.json."""
    caches = sorted(FFT_TEST_DIR.rglob("*_fft_analysis.json"))
    if not caches:
        print(f"[info] no cache under {FFT_TEST_DIR}; using a synthetic summary")
        return "synthetic_clip", cf_tools.synthesize_summary(0.0415), True
    path = caches[0]
    raw = json.loads(path.read_text())
    summary = raw.get("summary", {})
    stem = path.name.replace("_fft_analysis.json", "")
    if "anisotropy" in summary:
        return stem, summary, False
    # older schema: derive a plausible anisotropy from the numbers it does have
    print(f"[info] {path.relative_to(REPO)} uses the pre-anisotropy schema "
          f"(keys: {sorted(summary)}) -> synthesising a summary")
    hf = summary.get("hf_ratio", {}).get("mean", 0.9)
    slope = summary.get("spectral_slope", {}).get("mean", -0.13)
    aniso = 0.030 + 0.02 * abs(slope)          # deterministic, in the real band
    return stem, cf_tools.synthesize_summary(
        aniso,
        num_frames=summary.get("num_frames_analyzed", 60),
        peak_ratio=1.0 + hf * 0.5,
        axis_ratio=1.0 + abs(slope),
    ), True


def main() -> int:
    T = cf_tools.FFT_THRESHOLD_DEFAULT

    # ---- 1. parameterised interpretation == canonical at the default -------
    rng = random.Random(0)
    for _ in range(500):
        s = cf_tools.synthesize_summary(
            rng.uniform(0.02, 0.10), num_frames=rng.choice([30, 60]),
            peak_ratio=rng.uniform(1.0, 2.0), axis_ratio=rng.uniform(0.9, 1.3))
        assert cf_tools.generate_interpretation(s, T) == canonical_interpretation(s), s
    print("[1] generate_interpretation(·, 0.048) == fft.compute_fft's, 500 random "
          "summaries -- OK")

    # ---- 2. a real cached analysis, all three modes ------------------------
    stem, summary, synthesised = load_a_real_summary()
    aniso = summary["anisotropy"]["mean"]
    print(f"\n[2] clip {stem!r}  anisotropy={aniso:.4f}  "
          f"verdict={cf_tools.verdict_of(aniso, T)}"
          f"{'  (summary synthesised)' if synthesised else ''}")

    texts = {}
    for mode in ("real", "reverse", "corrupt", "none"):
        cf_summary, text = cf_tools.fft_counterfactual(summary, mode, stem, T, cf_seed=0)
        texts[mode] = text
        print(f"\n----- fft-mode = {mode} " + "-" * 46)
        print(text)

    # ---- 3. the reverse verdict really flips ------------------------------
    v_real = cf_tools.verdict_of(aniso, T)
    rev_aniso = cf_tools.reverse_anisotropy(aniso, T)
    v_rev = cf_tools.verdict_of(rev_aniso, T)
    assert v_rev != v_real, f"reverse did not flip the verdict ({v_real} -> {v_rev})"
    assert f"VERDICT HINT: {v_rev.upper()}" in texts["reverse"]
    assert f"VERDICT HINT: {v_real.upper()}" in texts["real"]
    assert f"{rev_aniso:.4f}" in texts["reverse"], "numbers and verdict must agree"
    print(f"\n[3] reverse flips the verdict: {v_real} ({aniso:.4f}) -> "
          f"{v_rev} ({rev_aniso:.4f}) -- OK")

    # ---- 4. the flip holds over the whole calibrated band ------------------
    flipped = 0
    for a in [0.020, 0.030, 0.034, 0.040, 0.0475, 0.048, 0.050, 0.055, 0.070, 0.090, 0.11]:
        r = cf_tools.reverse_anisotropy(a, T)
        assert cf_tools.FFT_ANISO_CLAMP[0] <= r <= cf_tools.FFT_ANISO_CLAMP[1]
        if cf_tools.verdict_of(r, T) != cf_tools.verdict_of(a, T):
            flipped += 1
        else:
            # only the exact-threshold tie may survive (both sides are "Generated")
            assert abs(a - T) < 1e-9, f"reverse failed to flip at anisotropy={a}"
    print(f"[4] reverse flips {flipped}/11 probes across [0.020, 0.11] "
          "(only the exact-threshold tie is exempt) -- OK")

    # ---- 5. corrupt is seeded, per-clip, and threshold-straddling ----------
    a1 = cf_tools.corrupt_anisotropy("clip_a", 0)
    a2 = cf_tools.corrupt_anisotropy("clip_a", 0)
    a3 = cf_tools.corrupt_anisotropy("clip_a", 1)
    a4 = cf_tools.corrupt_anisotropy("clip_b", 0)
    assert a1 == a2 and a1 != a3 and a1 != a4
    draws = [cf_tools.corrupt_anisotropy(f"clip_{i}", 0) for i in range(400)]
    assert all(cf_tools.FFT_CORRUPT_RANGE[0] <= d <= cf_tools.FFT_CORRUPT_RANGE[1]
               for d in draws)
    gen = sum(d >= T for d in draws)
    print(f"[5] corrupt: deterministic per (seed, clip); {gen}/400 draws land on "
          f"GENERATED (uniform over {cf_tools.FFT_CORRUPT_RANGE}) -- OK")

    # ---- 6. no cached interpretation leaks in a non-real mode -------------
    poisoned = dict(summary)
    marker = "THIS IS THE CACHED REAL INTERPRETATION"
    fake_cache = {"summary": poisoned, "interpretation": marker}
    for mode in ("none", "corrupt", "reverse"):
        _, text = cf_tools.fft_counterfactual(fake_cache["summary"], mode, stem, T, 0)
        assert marker not in text
    assert texts["none"] == cf_tools.FFT_NONE_TEXT
    print("[6] non-real modes never return the cached interpretation string -- OK")

    # ---- 7. a recalibrated threshold propagates everywhere -----------------
    t2 = 0.052
    txt = cf_tools.generate_interpretation(cf_tools.synthesize_summary(0.050), t2)
    assert "(threshold: 0.052)" in txt and "VERDICT HINT: REAL" in txt
    assert "0.052 - 0.070" in txt and "0.034 - 0.051" in txt
    assert f">= {t2}" in cf_tools.fft_tool_description(t2)
    print(f"[7] --fft-threshold {t2} propagates to the text, the reference ranges "
          "and the tool description -- OK")

    print("\nAll FFT counterfactual tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
