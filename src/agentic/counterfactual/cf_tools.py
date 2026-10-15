"""Counterfactual tool-output machinery for the E10 faithfulness harness.

Pure standard library + numpy-free helpers, so every text transform can be unit
tested offline (no qwen-agent, no torch, no GPU).  ``main_counterfactual.py``
imports this module; ``analyze_cf.py`` reuses the regexes.

The three tool channels are perturbed independently:

===========  ==========================================================
mode         meaning
===========  ==========================================================
``real``     the canonical tool output, untouched (baseline)
``none``     the tool reports that it has nothing to offer
``corrupt``  syntactically valid but uninformative / random output
``reverse``  output whose *verdict is flipped* while staying internally
             consistent (FFT numbers + verdict, RAFT flow hue)
``mismatch`` (SAM only) the real output of a different clip
===========  ==========================================================

The FFT reversal is the load-bearing one: it does **not** paste a contradictory
sentence onto the cached numbers, it mirrors the anisotropy about the decision
threshold and *regenerates* the whole interpretation from the mirrored summary,
so a model reading the numbers cannot tell the text apart from a genuine
opposite-verdict clip.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

__all__ = [
    "FFT_ANISO_CLAMP",
    "FFT_CORRUPT_RANGE",
    "FFT_MODES",
    "FFT_NONE_TEXT",
    "FFT_THRESHOLD_DEFAULT",
    "RAFT_MODES",
    "SAM_MODES",
    "SAM_NONE_TEXT",
    "TOOL_CITATION_RE",
    "cf_dir",
    "corrupt_anisotropy",
    "fft_counterfactual",
    "fft_tool_description",
    "generate_interpretation",
    "relative_key",
    "reverse_anisotropy",
    "scenario_stem",
    "synthesize_summary",
    "verdict_of",
]

FFT_THRESHOLD_DEFAULT = 0.048

#: the reversed anisotropy is clamped so it stays inside the range the agent was
#: calibrated on (a mirrored 0.005 would read as an obvious artefact).
FFT_ANISO_CLAMP = (0.025, 0.09)

#: ``corrupt`` draws uniformly over a band that straddles the threshold, so the
#: verdict itself is a coin flip that carries no information about the clip.
FFT_CORRUPT_RANGE = (0.030, 0.082)

FFT_MODES = ("real", "none", "corrupt", "reverse")
SAM_MODES = ("real", "none", "corrupt", "mismatch")
RAFT_MODES = ("real", "none", "corrupt", "reverse")

FFT_NONE_TEXT = "Frequency analysis unavailable for this clip."
SAM_NONE_TEXT = "Isolated 0 objects."
RAFT_NONE_PAYLOAD = {"error": "motion analysis unavailable"}

#: weather re-renderings / model sizes of one and the same base scenario --
#: kept in sync with ``src/analysis/stats.py`` (duplicated so this module stays
#: import-light: no polars).  Note the *space* in "Golden hour".
WEATHER_RE = re.compile(r"_(Original|Morning|Night|Snowy|Rainy|Foggy|Sunny|Golden hour)$")
SIZE_RE = re.compile(r"_(7b|14b)$")

#: does the model's ``evaluation`` cite a tool at all?  Used by analyze_cf.py.
TOOL_CITATION_RE = re.compile(
    r"anisotrop|\bFFT\b|frequenc|spectr|optical\s+flow|\bflow\b|motion|mask|"
    r"segment|\bcrop\b|\bSAM\b|\bRAFT\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def relative_key(video_path: Path | str, root_dir: Path | str) -> Path:
    """``<source>/<folder>/<file>.mp4`` for a clip under ``final_dataset``."""
    video_path = Path(video_path)
    root_dir = Path(root_dir)
    try:
        return video_path.relative_to(root_dir)
    except ValueError:
        pass
    parts = list(video_path.parts)
    if "final_dataset" in parts:
        return Path(*parts[parts.index("final_dataset") + 1:])
    return Path(video_path.name)


def scenario_stem(stem: str) -> str:
    """Clip stem with the weather and model-size suffixes stripped."""
    return SIZE_RE.sub("", WEATHER_RE.sub("", stem))


def cf_dir(video_path: Path | str, root_dir: Path | str, cf_root: Path | str,
           tool: str) -> Path:
    """Counterfactual artefact dir mirroring the canonical ``<root>/<tool>/...``.

    The canonical tools insert the tool name right after the ``final_dataset``
    segment; the counterfactual ones write the same relative tree under
    ``<cf_root>/<tool>/`` so the canonical caches are never touched.
    """
    return Path(cf_root) / tool / relative_key(video_path, root_dir).parent


# ---------------------------------------------------------------------------
# FFT interpretation (threshold-parameterised copy of fft/compute_fft.py)
# ---------------------------------------------------------------------------


def generate_interpretation(summary: dict, threshold: float = FFT_THRESHOLD_DEFAULT) -> str:
    """Byte-identical to ``fft.compute_fft.generate_interpretation`` at 0.048.

    The only change is that the calibrated decision threshold is a parameter
    (``--fft-threshold``) instead of a hard-coded constant, so a recalibrated
    threshold can be tested without touching the canonical tool.  The two
    confidence cut-offs and the printed reference ranges are expressed as
    offsets from it, which reproduce 0.055 / 0.040 / 0.047 at the default.
    """
    lines = []
    n_frames = summary["num_frames_analyzed"]

    aniso_mean = summary["anisotropy"]["mean"]
    peak_ratio = summary["peak_ratio"]["mean"]
    axis_ratio = summary["axis_ratio"]["mean"]

    ANISO_THRESHOLD = round(threshold, 6)
    high_gen = round(threshold + 0.007, 6)
    high_real = round(threshold - 0.008, 6)
    real_hi = threshold - 0.001

    lines.append(f"=== FFT Directional Energy Analysis ({n_frames} frames) ===")
    lines.append("")
    lines.append("PRIMARY METRIC — Anisotropy (directional energy concentration in high frequencies):")
    lines.append(f"  anisotropy = {aniso_mean:.4f}  (threshold: {ANISO_THRESHOLD})")
    lines.append(f"  peak_ratio = {peak_ratio:.4f}")
    lines.append(f"  axis_ratio = {axis_ratio:.4f}")
    lines.append("")
    lines.append("  Reference ranges (calibrated on driving video dataset):")
    lines.append(f"    Real footage:  0.034 - {real_hi:.3f}")
    lines.append(f"    AI-generated:  {threshold:.3f} - 0.070")
    lines.append("")

    if aniso_mean >= ANISO_THRESHOLD:
        confidence = "HIGH" if aniso_mean >= high_gen else "MODERATE"
        lines.append(f"  ➤ VERDICT HINT: GENERATED ({confidence} confidence)")
        lines.append(f"    Anisotropy {aniso_mean:.4f} > {ANISO_THRESHOLD} indicates directional")
        lines.append("    energy patterns (cross-shaped artifacts) typical of AI generators.")
    else:
        confidence = "HIGH" if aniso_mean <= high_real else "MODERATE"
        lines.append(f"  ➤ VERDICT HINT: REAL ({confidence} confidence)")
        lines.append(f"    Anisotropy {aniso_mean:.4f} ≤ {ANISO_THRESHOLD} indicates uniform energy")
        lines.append("    distribution consistent with natural camera footage.")

    return "\n".join(lines)


def fft_tool_description(threshold: float = FFT_THRESHOLD_DEFAULT) -> str:
    """The ``get_frequency_analysis`` tool description, threshold-parameterised."""
    return f"""
    Analyzes FFT frequency spectrum to detect AI-generation artifacts.
    Returns numeric anisotropy score with calibrated threshold and verdict hint.
    Anisotropy >= {threshold} suggests generated, < {threshold} suggests real.
    """


def verdict_of(summary_or_aniso, threshold: float = FFT_THRESHOLD_DEFAULT) -> str:
    """``"Generated"`` / ``"Real"`` implied by an anisotropy (or a summary)."""
    if isinstance(summary_or_aniso, dict):
        aniso = summary_or_aniso["anisotropy"]["mean"]
    else:
        aniso = float(summary_or_aniso)
    return "Generated" if aniso >= threshold else "Real"


# ---------------------------------------------------------------------------
# FFT summary perturbations
# ---------------------------------------------------------------------------


def synthesize_summary(
    aniso_mean: float,
    num_frames: int = 60,
    peak_ratio: float = 1.35,
    axis_ratio: float = 1.05,
    aniso_std: float = 0.004,
) -> dict:
    """Minimal summary dict in the *current* schema.

    Older caches (``dataset/fft_test/**/*_fft_analysis.json``) predate the
    anisotropy metric; this builds a summary the interpretation can consume so
    the text transforms remain testable against them.
    """
    return {
        "num_frames_analyzed": num_frames,
        "anisotropy": {
            "mean": float(aniso_mean),
            "std": float(aniso_std),
            "max": float(aniso_mean + 2 * aniso_std),
        },
        "peak_ratio": {"mean": float(peak_ratio)},
        "axis_ratio": {"mean": float(axis_ratio)},
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def reverse_anisotropy(aniso_mean: float, threshold: float = FFT_THRESHOLD_DEFAULT) -> float:
    """Mirror the anisotropy about the threshold, then clamp to the usable band."""
    return _clamp(2.0 * threshold - float(aniso_mean), *FFT_ANISO_CLAMP)


def corrupt_anisotropy(stem: str, cf_seed: int = 0) -> float:
    """Uninformative anisotropy, deterministic per (seed, clip)."""
    rng = random.Random(f"{cf_seed}:{stem}")
    return rng.uniform(*FFT_CORRUPT_RANGE)


def fft_counterfactual(
    summary: dict,
    mode: str,
    stem: str,
    threshold: float = FFT_THRESHOLD_DEFAULT,
    cf_seed: int = 0,
) -> tuple[dict, str]:
    """``(counterfactual summary, regenerated interpretation)``.

    ``summary`` must be the *real* summary (schema of
    ``fft.compute_fft.process_video_fft``).  Only ``anisotropy.mean`` is
    rewritten -- every other number is carried over untouched -- and the text is
    always regenerated from the resulting summary, so the cached ``interpretation``
    string of the real clip is never shown in a non-real mode.
    """
    if mode not in FFT_MODES:
        raise ValueError(f"unknown fft mode {mode!r} (expected one of {FFT_MODES})")
    if mode == "none":
        return dict(summary), FFT_NONE_TEXT

    out = json.loads(json.dumps(summary))  # deep copy, JSON-safe
    real_aniso = float(out["anisotropy"]["mean"])
    if mode == "real":
        new_aniso = real_aniso
    elif mode == "reverse":
        new_aniso = reverse_anisotropy(real_aniso, threshold)
    else:  # corrupt
        new_aniso = corrupt_anisotropy(stem, cf_seed)

    out["anisotropy"]["mean"] = new_aniso
    out["cf"] = {
        "mode": mode,
        "threshold": threshold,
        "anisotropy_real": real_aniso,
        "anisotropy_cf": new_aniso,
        "verdict_real": verdict_of(real_aniso, threshold),
        "verdict_cf": verdict_of(new_aniso, threshold),
        "cf_seed": cf_seed,
    }
    return out, generate_interpretation(out, threshold)
