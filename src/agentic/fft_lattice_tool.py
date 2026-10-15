"""Lattice score add-on for the agent's FFT tool (opt-in via --fft-lattice).

Appends to the tool text a second metric -- the strength of the periodic grid peaks that the
generator's decoder leaves at (+-W/4, +-H/4) of the 2-D spectrum -- and a combined verdict from a
logistic rule fitted on 1000 real / 1000 generated clips (raw files; see
report/reports/fft_revision_findings.md).  Pure numpy + cv2; no qwen-agent dependency.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# score = A*(aniso-MA)/SA + B*(lattice-ML)/SL + C ;  generated <=> score >= 0
LATTICE_RULE = {"A": 1.87, "MA": 0.0484, "SA": 0.0107, "B": 4.09, "ML": 1.585, "SL": 1.324, "C": 1.58}
_lattice_mod = None


def _load_lattice_module():
    global _lattice_mod
    if _lattice_mod is None:
        here = Path(__file__).resolve()
        for cand in (here.parents[1] / "fft" / "lattice.py", here.parents[2] / "src" / "fft" / "lattice.py"):
            if cand.is_file():
                spec = importlib.util.spec_from_file_location("fft_lattice", cand)
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                _lattice_mod = mod
                break
        if _lattice_mod is None:
            raise FileNotFoundError("src/fft/lattice.py not found next to the agent code")
    return _lattice_mod


def lattice_score(video_path: Path, cache_dir: Path | None = None, n_frames: int = 4) -> dict:
    """Return the lattice features for a clip, cached as <stem>_lattice.json in cache_dir."""
    cache = (Path(cache_dir) / f"{Path(video_path).stem}_lattice.json") if cache_dir else None
    if cache and cache.is_file():
        return json.loads(cache.read_text())
    r = _load_lattice_module().lattice_features(str(video_path), n_frames)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(r, indent=2))
    return r


def combined_score(aniso_mean: float, lattice_primary: float, intercept: float | None = None) -> float:
    r = LATTICE_RULE
    c = r["C"] if intercept is None else intercept
    return r["A"] * (aniso_mean - r["MA"]) / r["SA"] + r["B"] * (lattice_primary - r["ML"]) / r["SL"] + c


def lattice_block(video_path: Path, aniso_mean: float, cache_dir: Path | None = None,
                  intercept: float | None = None) -> str:
    """Text appended to the FFT tool output when --fft-lattice is on."""
    feats = lattice_score(video_path, cache_dir)
    if feats.get("error") or feats.get("lattice_primary") is None:
        return "\nSECONDARY METRIC — Lattice score: unavailable for this clip."
    lat = float(feats["lattice_primary"])
    s = combined_score(aniso_mean, lat, intercept)
    r = LATTICE_RULE
    c = r["C"] if intercept is None else intercept
    verdict = "GENERATED" if s >= 0 else "REAL"
    return "\n".join([
        "",
        "SECONDARY METRIC — Lattice score (periodic grid peaks at (±W/4, ±H/4) left by the generator's decoder):",
        f"  lattice = {lat:.2f}   (reference: real footage ≈ 0.6; generated with a visible grid ≥ 1.0, drive-dreams ≈ 2.5)",
        "",
        "COMBINED VERDICT (anisotropy + lattice, logistic rule calibrated on 1000 real / 1000 generated clips):",
        f"  score = {r['A']:.2f}·(anisotropy − {r['MA']:.4f})/{r['SA']:.4f} + {r['B']:.2f}·(lattice − {r['ML']:.3f})/{r['SL']:.3f} + {c:.2f} = {s:+.2f}",
        f"  ➤ COMBINED VERDICT: {verdict}  (score ≥ 0 → generated; this verdict overrides the anisotropy-only hint)",
    ])


LATTICE_PROMPT_NOTE = (
    "\nNOTE: get_frequency_analysis also returns a lattice score and a COMBINED VERDICT "
    "(anisotropy + lattice). When present, trust the COMBINED VERDICT over the anisotropy threshold alone.\n"
)
LATTICE_DESC_NOTE = " Also returns a lattice score and a combined verdict; trust the combined verdict when present."
