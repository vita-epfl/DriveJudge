"""
Shared constants + helpers for the standalone FFT feature-extraction package
(``src/fft/``), used by the real-vs-generated calibration / robustness study.

This module is deliberately dependency-free (stdlib only, no numpy / cv2 /
polars) so that it can be imported *before* numpy in order to pin the BLAS /
OpenMP thread counts -- see :func:`set_single_thread_env`.

Nomenclature
------------
generator   short tag for the clip's origin:
              ``dd``          cosmos-drive-dreams   (generated)
              ``p1``          cosmos-predict1       (generated)
              ``p2``          cosmos-predict2       (generated; excluded from
                                                    the manifest by default --
                                                    see ``manifest.py
                                                    --exclude-generators``)
              ``real``        CarCrash/videos/Normal
              ``real_crash``  CarCrash/videos/Crash-1500  (optional)
clip_id     ``<generator>/<folder>/<stem>`` for the generated pools
            (e.g. ``dd/negative/agg_tk_030_Foggy``, ``p1/artifacts/<stem>``,
            ``p2/negative/<stem>``) and ``<generator>/<stem>`` for real clips
            (e.g. ``real/000012``).  Unique across the whole manifest --
            Normal/ and Crash-1500/ share *all* 1500 basenames, hence the
            distinct ``real_crash`` prefix.
scenario_id cluster key grouping the weather / model-size variants of one
            underlying scene, e.g. ``dd/negative/agg_tk_030``.  Real clips are
            singleton clusters (``real/000012``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Repo layout
# --------------------------------------------------------------------------

# src/fft/common.py -> src/fft -> src -> <repo root>
REPO = Path(__file__).resolve().parents[2]

DATASET_DIR = REPO / "dataset"
RESULTS_FFT_DIR = REPO / "results" / "fft"
CONDITIONS_DIR = DATASET_DIR / "fft_conditions"

DEFAULT_MANIFEST = RESULTS_FFT_DIR / "manifest.parquet"

# --------------------------------------------------------------------------
# Video sources (overridable via environment)
# --------------------------------------------------------------------------

FINAL_DATASET_ROOT = Path(
    os.environ.get(
        "FINAL_DATASET_ROOT",
        "/home/matthieu/Documents/Projet de master/final_dataset",
    )
)

# Directory holding the ``Normal/`` and ``Crash-1500/`` sub-folders.
REAL_ROOT = Path(
    os.environ.get(
        "REAL_ROOT",
        "/home/matthieu/Documents/Projet de master/Dataset/CarCrash/videos",
    )
)

REAL_NORMAL_DIR = REAL_ROOT / "Normal"
REAL_CRASH_DIR = REAL_ROOT / "Crash-1500"

# source directory name -> short generator tag
GENERATED_SOURCES = {
    "cosmos-drive-dreams": "dd",
    "cosmos-predict1": "p1",
    "cosmos-predict2": "p2",
}
# and back
SOURCE_OF_GENERATOR = {v: k for k, v in GENERATED_SOURCES.items()}

# The four folders that make up the labelled benchmark within cosmos-drive-dreams.
DD_BENCHMARK_FOLDERS = (
    "negative",
    "crossing_stop",
    "crossing_red_lights",
    "aggressive_takeover",
)

# Ground-truth question files (benchmark membership).
QUESTION_FILES = {
    "dd": DATASET_DIR / "dataset_final" / "questions_cosmos-drive-dreams.json",
    "p1": DATASET_DIR / "dataset_final" / "questions_cosmos-predict1.json",
}

# Previous real-vs-generated study membership.
REALGEN_V1_JSON = REPO / "src" / "real_vs_generated" / "groundtruth_realgen.json"

# Original 18+18 FFT calibration clips.
FFT_CALIB_DIR = DATASET_DIR / "fft_test"

# Clip excluded from the benchmark by hand.
BLACKLISTED_STEMS = frozenset({"nOP1blfMCTg_48624_14b"})

# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------

# NOTE: "Golden hour" contains a space -- longest-first matching is not needed
# because none of these is a suffix of another, but keep the tuple explicit.
WEATHERS = (
    "Original",
    "Morning",
    "Night",
    "Snowy",
    "Rainy",
    "Foggy",
    "Sunny",
    "Golden hour",
)

GEN_SIZES = ("7b", "14b")


def parse_clip_name(source: str, folder: str, filename: str):
    """Split a clip filename into ``(scenario_id, weather, gen_size)``.

    ``source`` accepts either the on-disk directory name
    (``cosmos-drive-dreams``) or the short generator tag (``dd``); ``real`` /
    ``real_crash`` are also accepted.

    * cosmos-drive-dreams -- ``<scenario>_<Weather>.mp4``; the weather suffix is
      optional (exactly one clip has none).
    * cosmos-predict1 / -predict2 -- weather rule first, then a trailing
      ``_7b`` / ``_14b`` model-size suffix is stripped; whatever is left is the
      scenario.
    * real footage -- singleton clusters, no weather, no size.

    ``scenario_id`` is ``<generator>/<folder>/<scenario>`` for the generated
    pools and ``<generator>/<scenario>`` for real clips.
    """
    gen = GENERATED_SOURCES.get(source, source)
    stem = filename[:-4] if filename.lower().endswith(".mp4") else filename

    if gen in ("real", "real_crash"):
        return f"{gen}/{stem}", None, None

    weather = None
    for w in WEATHERS:
        if stem.endswith("_" + w):
            weather = w
            stem = stem[: -(len(w) + 1)]
            break

    gen_size = None
    if gen in ("p1", "p2"):
        for s in GEN_SIZES:
            if stem.endswith("_" + s):
                gen_size = s
                stem = stem[: -(len(s) + 1)]
                break

    scenario_id = f"{gen}/{folder}/{stem}" if folder else f"{gen}/{stem}"
    return scenario_id, weather, gen_size


def make_clip_id(generator: str, folder: str | None, stem: str) -> str:
    """``dd/negative/agg_tk_030_Foggy`` / ``real/000012``."""
    return f"{generator}/{folder}/{stem}" if folder else f"{generator}/{stem}"


# --------------------------------------------------------------------------
# Runtime helpers
# --------------------------------------------------------------------------


AGENTIC_DIR = REPO / "src" / "agentic"
AGENTIC_COMPUTE_FFT = AGENTIC_DIR / "fft" / "compute_fft.py"


def add_agentic_path() -> None:
    """Put ``src/agentic`` on ``sys.path`` -- the trick used by
    ``src/real_vs_generated/scripts/main_multi_tools_v4.py`` so that
    ``from utils.raft import ...`` / ``from fft.compute_fft import ...``
    resolve against the agent tree.

    CAVEAT: the agent's FFT module lives in a package *also* called ``fft``,
    which collides with this package (``src/fft``).  Whichever is imported
    first wins for the whole process, so prefer :func:`load_agentic_fft` for
    the FFT module itself; this helper is only useful for the agent's other
    sub-packages (``utils`` etc.).
    """
    agentic = str(AGENTIC_DIR)
    if agentic in sys.path:
        sys.path.remove(agentic)
    sys.path.insert(0, agentic)


def load_agentic_fft():
    """Import ``src/agentic/fft/compute_fft.py`` under an unambiguous name.

    Loaded straight off its file path as ``agentic_compute_fft`` so that it
    cannot be shadowed by -- and cannot shadow -- this ``fft`` package.  The
    module has no relative imports, so loading it in isolation is safe.
    """
    import importlib.util

    name = "agentic_compute_fft"
    if name in sys.modules:
        return sys.modules[name]
    if not AGENTIC_COMPUTE_FFT.exists():
        raise FileNotFoundError(f"agent FFT module not found: {AGENTIC_COMPUTE_FFT}")
    spec = importlib.util.spec_from_file_location(name, AGENTIC_COMPUTE_FFT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_single_thread_env() -> None:
    """Pin every BLAS / OpenMP backend to one thread.

    MUST be called before numpy (and therefore before cv2 / compute_fft) is
    imported, otherwise the runtimes have already read the variables.  We rely
    on ``fork`` semantics so the parent's settings are inherited by pool
    workers -- the point of this package is to parallelise over *clips*, not
    inside numpy.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"


def rel_to_repo(p) -> str:
    """Pretty-print a path relative to the repo root when possible."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)
