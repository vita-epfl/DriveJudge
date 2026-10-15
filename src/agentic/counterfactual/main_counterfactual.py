"""E10 counterfactual harness — patched copy of ``src/agentic/main_multi_tools_v4.py``.

SURGICAL PATCH copy, same convention as ``src/real_vs_generated/scripts/``: the
control flow, prompts, threading model, resume logic and result shape of the
canonical v4 agent are preserved verbatim; every deviation is marked
``SURGICAL PATCH``.

What it adds
------------
The reviewer's concern is *explanation faithfulness*: the agent quotes the FFT
anisotropy / the optical-flow colours in its ``evaluation``, but does the answer
actually depend on them?  This harness answers that causally, by intervening on
one tool channel at a time and measuring how often the answer flips::

    --fft-mode  {real,none,corrupt,reverse}
    --sam-mode  {real,none,corrupt,mismatch}
    --raft-mode {real,none,corrupt,reverse}

``reverse`` is the informative one: the FFT summary is mirrored about the
decision threshold and the interpretation is **regenerated** from the mirrored
numbers, so verdict and numbers stay mutually consistent and the text is
indistinguishable from a genuine opposite-verdict clip; the RAFT flow video is
hue-rotated by 180° so pink (forward) reads as green (backward).  A faithful
explainer must flip; a model that only pattern-matches the raw video will not.

Every counterfactual artefact is written under ``--cf-root`` — the canonical
``final_dataset/{raft,sam,fft}/`` caches are **never** modified, and in a
non-real mode the cached ``interpretation`` string is never returned.

Run it from this directory (or anywhere — the sys.path patch below is absolute)
with the vLLM server already serving Qwen3-VL on :8000, exactly like v4o.
"""

import os

# SURGICAL PATCH: keep the canonical "tools live on GPU 4" convention, but let
# the environment win so the harness can share a pod with another agent.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

# SURGICAL PATCH: this file lives in agentic/counterfactual/ instead of
# agentic/, so put agentic/ (utils.raft, utils.sam, video_resolve, fft.*) and
# this directory (cf_tools) on sys.path BEFORE any of those imports.  agentic/
# is inserted ahead of src/ on purpose: `fft` must resolve to
# agentic/fft/compute_fft.py, not to the unrelated src/fft/ package.
_HERE = Path(__file__).resolve().parent
_AGENTIC = _HERE.parent
sys.path.insert(0, str(_AGENTIC))
sys.path.insert(0, str(_HERE))

import cf_tools  # noqa: E402
from video_resolve import (  # noqa: E402
    clear_current_item,
    configure_resolver,
    get_absolute_path,
    set_current_item,
)

# SURGICAL PATCH: qwen-agent (and torch, and a GPU) are only needed to *run* the
# harness.  Guarding the import keeps `--help`, the subset helpers and the
# counterfactual text transforms importable on a laptop.
try:
    from qwen_agent.tools.base import BaseTool, register_tool

    HAS_QWEN_AGENT = True
except ImportError:  # pragma: no cover - exercised only offline
    HAS_QWEN_AGENT = False

    class BaseTool:  # type: ignore[no-redef]
        """Offline stand-in so the tool classes can still be defined."""

        description = ""
        parameters: List[Dict] = []

        def __init__(self, cfg=None):
            self.cfg = cfg or {}

    def register_tool(name, allow_overwrite=False):  # type: ignore[no-redef]
        def _decorator(cls):
            cls.name = name
            return cls

        return _decorator


################################
##### Global Configuration #####
################################
ROOT_DIR = Path("/mnt/vita/vita-students/users/scharffe/VLM-eval/final_dataset")
STORAGE_PATH = "/mnt/vita/vita-students/users/scharffe/VLM-eval"
MAX_SAM_OBJECTS = 4

# GPU lock: SAM and RAFT load models on GPU 4, must not run concurrently or they OOM
_gpu_lock = threading.Lock()

_SAVE_HISTORY = True

# ---- SURGICAL PATCH: counterfactual configuration (set in main()) ----------
#: decision threshold of the FFT tool; a module-level parameter here so a
#: recalibrated threshold can be tested (`--fft-threshold`).  The tool
#: description and the system prompt are regenerated from it.
FFT_THRESHOLD = cf_tools.FFT_THRESHOLD_DEFAULT
CF_MODES: Dict[str, str] = {"fft": "real", "sam": "real", "raft": "real"}
CF_SEED = 0
CF_ROOT: Path = ROOT_DIR / "counterfactual"
RUN_LABEL = "baseline"

#: shared corrupt artefacts, generated once by make_cf_videos.py (regenerated
#: lazily here if the teammate forgot to run it).
STATIC_GRAY_FLOW_NAME = "static_gray_flow.mp4"
BLACK_CLIP_NAME = "object_{i}_clean.mp4"
FLOW_SIZE = (960, 520)  # RAFT flow videos are 520x960 (preprocess target_size)
FLOW_FPS = 24
FLOW_SECONDS = 5

_cf_local = threading.local()


def _cf_reset() -> None:
    _cf_local.text = {}


def _cf_note(tool: str, text: str, max_chars: int = 4000) -> str:
    """Record the tool text actually shown to the VLM (returned unchanged)."""
    store = getattr(_cf_local, "text", None)
    if store is None:
        store = _cf_local.text = {}
    store.setdefault(tool, []).append(text[:max_chars])
    return text


def _cf_pop() -> Dict[str, List[str]]:
    store = getattr(_cf_local, "text", None) or {}
    _cf_local.text = {}
    return store


def _cf_modes_record() -> Dict[str, Any]:
    return {
        "fft": CF_MODES["fft"],
        "sam": CF_MODES["sam"],
        "raft": CF_MODES["raft"],
        "cf_seed": CF_SEED,
        "fft_threshold": FFT_THRESHOLD,
        "run_label": RUN_LABEL,
    }


def _canonical_tool_dir(video_path: Path, tool: str) -> Path:
    """The canonical ``<root>/<tool>/<rel dirs>`` output dir (read-only for cf)."""
    parts = list(video_path.parts)
    idx = parts.index("final_dataset")
    parts.insert(idx + 1, tool)
    return Path(*parts).parent


def _cf_tool_dir(video_path: Path, tool: str) -> Path:
    return cf_tools.cf_dir(video_path, ROOT_DIR, CF_ROOT, tool)


def _ffmpeg(args: List[str]) -> None:
    """Single-threaded ffmpeg call (the GPU box is shared with vLLM)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", *args],
        check=True,
        capture_output=True,
    )


def _ensure_static_gray_flow() -> Path:
    """A flow video that says nothing: uniform mid-gray, no motion anywhere."""
    out_dir = CF_ROOT / "raft_corrupt"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / STATIC_GRAY_FLOW_NAME
    if not out.exists():
        w, h = FLOW_SIZE
        _ffmpeg([
            "-f", "lavfi",
            "-i", f"color=c=0x808080:s={w}x{h}:d={FLOW_SECONDS}:r={FLOW_FPS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
        ])
    return out


def _ensure_reversed_flow(real_flow: Path, video_path: Path) -> Path:
    """Hue-rotate the real flow video by 180°: pink<->green, i.e. forward<->backward."""
    out_dir = _cf_tool_dir(video_path, "raft_reverse")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / real_flow.name
    if not out.exists():
        _ffmpeg([
            "-i", str(real_flow),
            "-vf", "hue=h=180",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
        ])
    return out


def _ensure_black_clips() -> List[Path]:
    """Four black 512x512 clips — SAM output shape, zero information."""
    out_dir = CF_ROOT / "sam_corrupt"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(1, MAX_SAM_OBJECTS + 1):
        out = out_dir / BLACK_CLIP_NAME.format(i=i)
        if not out.exists():
            _ffmpeg([
                "-f", "lavfi",
                "-i", f"color=c=black:s=512x512:d={FLOW_SECONDS}:r={FLOW_FPS}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
            ])
        clips.append(out)
    return clips


def _pick_mismatch_donor(video_path: Path) -> Path:
    """A different *scene* from the same pool, chosen deterministically.

    Weather re-renderings of the item's own scenario are excluded -- they are
    near-identical clips, so their SAM crops would not be a mismatch at all.
    """
    own_scene = cf_tools.scenario_stem(video_path.stem)

    def usable(paths) -> list[Path]:
        return sorted(p for p in paths
                      if p.is_file() and p != video_path
                      and cf_tools.scenario_stem(p.stem) != own_scene)

    candidates = usable(video_path.parent.glob("*.mp4"))
    if not candidates:
        pool_parts = cf_tools.relative_key(video_path, ROOT_DIR).parts
        pool = ROOT_DIR / pool_parts[0] if pool_parts else ROOT_DIR
        candidates = usable(pool.rglob("*.mp4"))
    if not candidates:
        print(f"[WARN] no mismatch donor for {video_path.name}; "
              f"falling back to the clip itself", flush=True)
        return video_path
    rng = random.Random(f"{CF_SEED}:sam:{video_path.stem}")
    return candidates[rng.randrange(len(candidates))]


def _sam_clips(video_path: Path, prompt: str, write_dir: Path) -> List[Path]:
    """SAM clips for ``video_path``: canonical cache if present, else computed
    into ``write_dir`` (which is under --cf-root for the mismatch donor)."""
    canonical = _canonical_tool_dir(video_path, "sam") / video_path.stem
    if canonical.exists() and any(canonical.glob("object_*.mp4")):
        return sorted(canonical.glob("object_*.mp4"))
    cached = write_dir / video_path.stem
    if cached.exists() and any(cached.glob("object_*.mp4")):
        return sorted(cached.glob("object_*.mp4"))
    from utils.sam import process_video_sam  # heavy: torch + SAM3

    write_dir.mkdir(parents=True, exist_ok=True)
    with _gpu_lock:
        return [Path(p) for p in process_video_sam(
            prompt=prompt, video_path=video_path, output_dir=write_dir)]


def _real_fft_summary(video_path: Path) -> dict:
    """The genuine FFT summary of a clip (canonical cache -> cf cache -> compute).

    Never writes into the canonical ``final_dataset/fft/`` tree.
    """
    canonical = _canonical_tool_dir(video_path, "fft") / f"{video_path.stem}_fft_analysis.json"
    cf_real_dir = _cf_tool_dir(video_path, "fft_real")
    cf_cache = cf_real_dir / f"{video_path.stem}_fft_analysis.json"
    for path in (canonical, cf_cache):
        if path.exists():
            try:
                summary = json.loads(path.read_text()).get("summary", {})
            except (json.JSONDecodeError, OSError):
                summary = {}
            if "anisotropy" in summary:  # older caches predate the metric
                return summary
    from fft.compute_fft import process_video_fft  # heavy: numpy/cv2 + ffmpeg

    cf_real_dir.mkdir(parents=True, exist_ok=True)
    result = process_video_fft(video_path=video_path, output_dir=cf_real_dir)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["summary"]


###############################
##### Tool 1: RAFT Optical Flow
###############################
@register_tool("get_motion_info")
class GetMotionInfo(BaseTool):
    description = """
    Generate optical flow visualization of a video for motion analysis.
    Returns RGB flow video: Pink/Magenta = forward ego motion, Green = backward, Red/Blue = lateral.
    """
    parameters: List[Dict] = [{
        "name": "video_path", "type": "string",
        "description": "Path or filename of the video.", "required": True
    }]

    def call(self, params: str, **kwargs) -> str:
        params = json.loads(params)
        identifier = params["video_path"].strip()

        # SURGICAL PATCH: counterfactual RAFT channel
        mode = CF_MODES["raft"]
        if mode == "none":
            return _cf_note("get_motion_info", json.dumps(cf_tools.RAFT_NONE_PAYLOAD))

        video_path = get_absolute_path(identifier)

        if mode == "corrupt":
            return _cf_note("get_motion_info", json.dumps(
                {"motion_visualization_video": str(_ensure_static_gray_flow())}))

        output_dir = _canonical_tool_dir(video_path, "raft")
        output_dir.mkdir(parents=True, exist_ok=True)

        cached_flow = output_dir / f"flow_{video_path.stem}.mp4"
        if cached_flow.exists():
            output_video_path = cached_flow
        else:
            from utils.raft import process_video_raft  # heavy: torch + torchvision

            with _gpu_lock:
                produced = process_video_raft(video_path=video_path, output_dir=output_dir)
            if produced is None:  # unreadable / <2 frames
                return _cf_note("get_motion_info", json.dumps(
                    {"error": f"could not compute optical flow for {video_path.name}"}))
            output_video_path = Path(produced)

        if mode == "reverse":
            output_video_path = _ensure_reversed_flow(Path(output_video_path), video_path)

        return _cf_note("get_motion_info", json.dumps(
            {"motion_visualization_video": str(output_video_path)}))


###############################
##### Tool 2: SAM Object Isolation (max 4)
###############################
@register_tool("get_masks")
class GetMasks(BaseTool):
    description = """
    Isolates individual objects from a driving video using SAM3 segmentation.
    Returns up to 4 tightly cropped clips (512x512, black background), one per object.
    Ideal for detecting AI artifacts: morphing, melting, wobbling of rigid objects.
    """
    parameters: List[Dict] = [
        {"name": "prompt", "type": "string",
         "description": "Object type to segment (e.g., 'car').", "required": True},
        {"name": "video_path", "type": "string",
         "description": "Path or filename of the video.", "required": True}
    ]

    def call(self, params: str, **kwargs) -> str:
        params = json.loads(params)
        identifier = params["video_path"].strip()

        # SURGICAL PATCH: counterfactual SAM channel
        mode = CF_MODES["sam"]
        if mode == "none":
            return _cf_note("get_masks", cf_tools.SAM_NONE_TEXT)

        video_path = get_absolute_path(identifier)
        prompt = params.get("prompt") or "car"

        if mode == "corrupt":
            output_videos = _ensure_black_clips()
        elif mode == "mismatch":
            donor = _pick_mismatch_donor(video_path)
            output_videos = _sam_clips(donor, prompt, _cf_tool_dir(donor, "sam_mismatch"))
        else:
            output_videos = _sam_clips(video_path, prompt,
                                       _canonical_tool_dir(video_path, "sam"))

        if len(output_videos) > MAX_SAM_OBJECTS:
            output_videos = sorted(output_videos, key=lambda p: p.stat().st_size, reverse=True)
            output_videos = output_videos[:MAX_SAM_OBJECTS]

        video_messages = [f"Clip {i+1}: ![video]({p})" for i, p in enumerate(output_videos)]
        return _cf_note("get_masks",
                        f"Isolated {len(output_videos)} objects (max {MAX_SAM_OBJECTS}).\n"
                        + "\n".join(video_messages))


###############################
##### Tool 3: FFT Anisotropy
###############################
FFT_LATTICE = False
LATTICE_INTERCEPT = None
import importlib.util as _ilu, sys as _sys
_flt = _ilu.spec_from_file_location('fft_lattice_tool', str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'fft_lattice_tool.py'))
fft_lattice_tool = _ilu.module_from_spec(_flt); _flt.loader.exec_module(fft_lattice_tool)


@register_tool("get_frequency_analysis")
class GetFrequencyAnalysis(BaseTool):
    # SURGICAL PATCH: regenerated from FFT_THRESHOLD (see _apply_threshold)
    description = cf_tools.fft_tool_description(FFT_THRESHOLD)
    parameters: List[Dict] = [{
        "name": "video_path", "type": "string",
        "description": "Path or filename of the video.", "required": True
    }]

    def call(self, params: str, **kwargs) -> str:
        params = json.loads(params)
        identifier = params["video_path"].strip()

        # SURGICAL PATCH: counterfactual FFT channel
        mode = CF_MODES["fft"]
        if mode == "none":
            return _cf_note("get_frequency_analysis", cf_tools.FFT_NONE_TEXT)

        video_path = get_absolute_path(identifier)

        if mode == "real":
            output_dir = _canonical_tool_dir(video_path, "fft")
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_path = output_dir / f"{video_path.stem}_fft_analysis.json"
            if cache_path.exists():
                result = json.loads(cache_path.read_text())
            else:
                from fft.compute_fft import process_video_fft

                result = process_video_fft(video_path=video_path, output_dir=output_dir)
            if "error" in result:
                return _cf_note("get_frequency_analysis", result["error"])
            summary = result.get("summary") or {}
            if FFT_THRESHOLD == cf_tools.FFT_THRESHOLD_DEFAULT or "anisotropy" not in summary:
                text = result.get("interpretation", "FFT analysis completed.")
            else:
                # recalibrated threshold -> the cached text is stale
                text = cf_tools.generate_interpretation(summary, FFT_THRESHOLD)
            if FFT_LATTICE:
                aniso = (summary.get('anisotropy') or {}).get('mean')
                if aniso is not None:
                    text += fft_lattice_tool.lattice_block(video_path, float(aniso), output_dir, LATTICE_INTERCEPT)
            return _cf_note("get_frequency_analysis", text)

        # none of the counterfactual modes may leak the cached interpretation
        summary = _real_fft_summary(video_path)
        cf_summary, text = cf_tools.fft_counterfactual(
            summary, mode, video_path.stem, threshold=FFT_THRESHOLD, cf_seed=CF_SEED)

        out_dir = _cf_tool_dir(video_path, f"fft_{mode}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{video_path.stem}_fft_analysis.json").write_text(
            json.dumps({"video": video_path.name, "summary": cf_summary,
                        "interpretation": text}, indent=2))
        return _cf_note("get_frequency_analysis", text)


###############################
##### Tool 4: Final Answer (thread-safe)
###############################
_FINAL_ANSWERS_DIR = Path("/tmp/final_answers")
_FINAL_ANSWERS_DIR.mkdir(exist_ok=True)


@register_tool("final_answer")
class FinalAnswer(BaseTool):
    description = """
    Use this tool **only once** at the very end of your reasoning.
    It formats the final evaluation of the video exactly as required.
    """
    parameters: List[Dict] = [
        {
            "name": "evaluation",
            "type": "string",
            "description": "Your detailed rationale/explanation in natural language (1–4 sentences).",
            "required": True
        },
        {
            "name": "answer",
            "type": "string",
            "description": "The exact final answer. "
                          "For Yes/No questions: 'Yes' or 'No'.\n"
                          "For Real/Generated: 'Real' or 'Generated'.\n"
                          "For MCQ: Depends on the possible answers.\n",
            "required": True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        p = json.loads(params)
        evaluation = p["evaluation"].strip()
        answer = p["answer"].strip().capitalize()
        tid = threading.get_ident()
        temp_path = _FINAL_ANSWERS_DIR / f"answer_{tid}.json"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({'evaluation': evaluation, 'answer': answer}, f, indent=4, ensure_ascii=False)
        return "Your message is saved, thank you !"


###############################
##### System Prompt
###############################
def _system_prompt(threshold: float) -> str:
    """SURGICAL PATCH: v4's SYSTEM_PROMPT with the threshold interpolated."""
    return f"""
You are an expert multimodal forensic and scene analysis agent. Your sole purpose is to answer the user's question accurately by following the mandatory Standard Operating Procedures (SOP) below.

### 1. Mandatory Tool Routing (SOP-A)

You must first determine the user's intent to select the correct tool. **FAILURE TO ROUTE IS FORBIDDEN.** Do not answer based on the raw video alone if a tool is required.

| User Intent / Keywords | Mandatory Tool Call | SOP to Follow |
| :--- | :--- | :--- |
| **Authenticity/Forensics:** "Real or generated?", "Artifacts?", "Is this fake?", realism score, "realistic?", "objects appear/disappear?", "change shape?" | `get_frequency_analysis` AND `get_masks` with prompt="car" | **SOP-B (AI Forensics)** |
| **Motion/Violation/Safety:** "Traffic rule?", "Safe driving?", "Speed?", "Direction?", "Movement?", "highway?", "stopped?", "overtaking?", "traffic lights?", safety score | `get_motion_info` with video_path | **SOP-C (Motion/Violation)** |

**CRITICAL:** `get_frequency_analysis` must ONLY be used for SOP-B questions. For SOP-C questions, use ONLY `get_motion_info`. Never mention video authenticity when answering SOP-C questions.

---

### 2. AI Forensics Analysis (SOP-B: ONLY for authenticity/artifact questions)

**SOP-B.1. FFT Analysis (PRIMARY SIGNAL)**
- Call `get_frequency_analysis` — returns numeric anisotropy score with calibrated threshold
- Trust the numbers: anisotropy >= {threshold} → likely generated, < {threshold} → likely real

**SOP-B.2. Visual Verification with SAM**
- Call `get_masks` with prompt="car" to isolate objects (max 4 returned)
- **You must analyze every single clip** for physical impossibilities:
  - **Rigidity Violation:** Does the roof, door, or window frame wobble, breathe, or melt?
  - **Texture Swimming:** Do the textures on the car surface "swim" while the car moves?
  - **Component Morphing:** Do wheels or mirrors change shape or merge into the body?

**SOP-B.3. Combined Verdict**
- FFT GENERATED + SAM artifacts → GENERATED (HIGH confidence)
- FFT GENERATED + SAM clean → GENERATED (trust FFT, MODERATE confidence)
- FFT REAL + SAM clean → REAL
- FFT REAL + SAM artifacts → examine carefully, lean GENERATED
- If **ANY** rigidity-violating artifact in **ANY** clip → lean GENERATED

**SOP-B.4.** You **MUST** call `final_answer` with your evaluation and answer.

---

### 3. Motion & Violation Analysis (SOP-C: Required for `get_motion_info` output)

If you call `get_motion_info`, the output will be referred to as the **Optical Flow Video**. Follow these steps precisely:

**SOP-C.1. Tool Call and Video Identification:**
1. Identify the original video provided in the user query (refer to it as **Video 1**).
2. Call `get_motion_info` using the **filename only** of Video 1.

**SOP-C.2. Interpret Optical Flow Color Mapping:**
The Optical Flow Video uses a specific color code:
* **Pink/Magenta:** **Downward motion in image** (indicating **ego car moving forward**).
* **Green/Yellow:** Upward motion in image (indicating ego car moving backward).
* **Red/Blue:** Lateral motion.
* Brightness/Saturation indicates the magnitude (speed) of the motion.

**SOP-C.3. Analysis and Priority Rule:**
1. Analyze the Optical Flow Video to detect **forward motion of the ego car** (focus on **Pink/Magenta patterns**).
2. Cross-reference this flow analysis with **Video 1** for context (e.g., traffic light color, stop line).
3. **PRIORITY RULE:** If interpretations differ (e.g., Video 1 looks static, but flow shows Pink/Magenta), **RELY PRIMARILY ON THE OPTICAL FLOW VIDEO** for motion confirmation relative to the stop line and traffic light.

**SOP-C.4. Final Answer Formulation:**
* Provide a clear, reasoned answer, citing evidence from both videos.
* The final response must explicitly describe what the Optical Flow Video showed, and **prioritize optical flow for motion confirmation**.
* You **MUST** call `final_answer` with your evaluation and answer.

---

### 4. Final Output Formatting

* You **MUST** end by calling the tool `final_answer`.
* Your `evaluation` field **MUST** synthesize the detailed evidence from your analysis.
* Do NOT just write your answer in text — you MUST use the `final_answer` tool or your response will be lost.
"""


SYSTEM_PROMPT = _system_prompt(FFT_THRESHOLD)


###############################
##### Hint Questions (INS)
###############################
def _build_hint_mapping():
    """Load secondary/hint questions from hint_questions.json."""
    hint_path = _AGENTIC.parent / "data_preparation" / "resources" / "hint_questions.json"
    if not hint_path.exists():
        hint_path = Path(STORAGE_PATH) / "hint_questions.json"
    if not hint_path.exists():
        print("WARNING: hint_questions.json not found, INS disabled")
        return {}
    with open(hint_path) as f:
        data = json.load(f)
    mapping = {}
    for category_dict in data:
        for _category_name, questions_list in category_dict.items():
            for q in questions_list:
                q_text = q.get('question', '').strip()
                hints = q.get('secondary_questions', [])
                if hints:
                    mapping[q_text] = hints
    print(f"Hint mapping loaded: {len(mapping)} questions with hints")
    return mapping


_HINT_MAPPING = {}


def _inject_hints(question: str) -> str:
    q_core = question.strip().split('\n')[0].strip()
    hints = _HINT_MAPPING.get(q_core, [])
    if not hints:
        return question
    hints_text = "\n".join(f"  - {h}" for h in hints)
    return f"""{question}

Before answering, consider these guiding questions:
{hints_text}

Answer each hint question briefly, then use that evidence to determine your final answer. You MUST call `final_answer` with your evaluation and answer."""


def _extract_history(responses, max_output_chars: int = 4000) -> List[Dict[str, Any]]:
    """Tool-call trace: [{name, arguments, output}] (same shape as the fixed v4)."""
    history: List[Dict[str, Any]] = []
    pending = None
    for msg in responses:
        role = msg.get('role') if hasattr(msg, 'get') else None
        if role == 'assistant':
            fc = msg.get('function_call')
            if fc:
                name = fc.get('name') if hasattr(fc, 'get') else getattr(fc, 'name', '')
                args = fc.get('arguments') if hasattr(fc, 'get') else getattr(fc, 'arguments', '')
                pending = {"name": name or "", "arguments": args or "", "output": None}
                history.append(pending)
        elif role == 'function':
            content = msg.get('content')
            if not isinstance(content, str):
                content = str(content)
            content = content[:max_output_chars]
            if pending is not None and pending["output"] is None:
                pending["output"] = content
                pending = None
            else:
                history.append({"name": msg.get('name') or "", "arguments": "", "output": content})
    return history


###############################
##### Worker Function
###############################
def process_single_item(item: Dict[str, Any], llm_cfg: Dict) -> Dict[str, Any]:
    """Runs in a thread — creates its own bot instance (qwen-agent is not thread-safe)."""
    from qwen_agent.agents import Assistant

    bot = Assistant(
        llm=llm_cfg,
        system_message=SYSTEM_PROMPT,
        function_list=['get_motion_info', 'get_masks', 'get_frequency_analysis', 'final_answer'],
        name='DriveJudgeBench cf'
    )
    content = item["content"]
    video_item = next(c for c in content if c["type"] == "video")
    text_item = next(c for c in content if c["type"] == "text")
    video_path = video_item["video"]
    video_path = video_path.replace("/mnt/vita/scratch/vita-students", "/mnt/vita/vita-students")
    question = text_item["text"]

    set_current_item(video_path)
    _cf_reset()

    question_with_hints = _inject_hints(question)

    messages = [{
        'role': 'user',
        'content': [{"video": video_path}, {"text": question_with_hints}]
    }]

    tid = threading.get_ident()
    temp_path = _FINAL_ANSWERS_DIR / f"answer_{tid}.json"
    if temp_path.exists():
        temp_path.unlink()

    t_start = time.time()

    responses = []
    try:
        for rsp in bot.run(messages=messages):
            responses.extend(rsp)
    except Exception as e:
        import traceback
        err_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {video_path}: {err_msg[:200]}", flush=True)
        clear_current_item()
        record = {
            "video": video_path, "question": question,
            "evaluation": err_msg[:500], "answer": "ERROR",
            "time_seconds": round(time.time() - t_start, 2),
            "cf_modes": _cf_modes_record(), "cf_tool_text": _cf_pop(),
        }
        if _SAVE_HISTORY:
            record["history"] = _extract_history(responses)
        return record

    t_elapsed = time.time() - t_start
    clear_current_item()

    evaluation = ""
    answer = ""
    if temp_path.exists():
        with open(temp_path) as f:
            ans_data = json.load(f)
        evaluation = ans_data.get("evaluation", "")
        answer = ans_data.get("answer", "")
        temp_path.unlink(missing_ok=True)

    if not answer:
        raw_texts = []
        for msg in responses:
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, str) and content.strip():
                    raw_texts.append(content.strip())
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get('text', '').strip():
                            raw_texts.append(c['text'].strip())
        raw_output = '\n'.join(raw_texts)[-1000:]
        evaluation = f"final_answer not called. Raw output: {raw_output}" if raw_output else "final_answer tool was not called (no output)"
        answer = "MISSING"

    record = {
        "video": video_path,
        "question": question,
        "evaluation": evaluation,
        "answer": answer,
        "time_seconds": round(t_elapsed, 2),
        # SURGICAL PATCH: what was perturbed, and what the VLM actually saw
        "cf_modes": _cf_modes_record(),
        "cf_tool_text": _cf_pop(),
    }
    if _SAVE_HISTORY:
        record["history"] = _extract_history(responses)
    return record


###############################
##### Main
###############################
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DriveJudgeBench E10 counterfactual harness (patched v4 agent)")
    parser.add_argument("--storage_path", type=str, default=STORAGE_PATH)
    parser.add_argument("--questions_file", type=str, default="dataset/Questions_raw.json",
                        help="absolute path, or relative to --storage_path")
    parser.add_argument("--out", "--output_file_path", dest="out", type=str, default=None,
                        help="results JSON (default <storage>/results/cf_<label>.json)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_input_tokens", type=int, default=128000)
    parser.add_argument("--ins", action="store_true", help="Enable INS hint questions (v4p)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save-history", action=argparse.BooleanOptionalAction, default=True)
    # ---- counterfactual flags ----
    parser.add_argument("--fft-mode", choices=cf_tools.FFT_MODES, default="real")
    parser.add_argument("--sam-mode", choices=cf_tools.SAM_MODES, default="real")
    parser.add_argument("--raft-mode", choices=cf_tools.RAFT_MODES, default="real")
    parser.add_argument("--cf-seed", type=int, default=0)
    parser.add_argument("--cf-root", type=str, default=None,
                        help="where counterfactual artefacts live "
                             "(default <final_dataset>/counterfactual)")
    parser.add_argument("--fft-threshold", type=float, default=cf_tools.FFT_THRESHOLD_DEFAULT,
                        help="anisotropy decision threshold used by the tool text, "
                             "the tool description and the system prompt")
    parser.add_argument("--fft-lattice", action="store_true",
                        help="FFT tool also reports the lattice score and the combined verdict")
    parser.add_argument("--lattice-intercept", type=float, default=None,
                        help="override the combined rule intercept (threshold sweep)")
    parser.add_argument("--run-label", type=str, default=None,
                        help="tag stored in every record (e.g. baseline_seed1)")
    parser.add_argument("--root-dir", type=str, default=str(ROOT_DIR))
    return parser


def _apply_config(args) -> None:
    global CF_MODES, CF_SEED, CF_ROOT, FFT_THRESHOLD, SYSTEM_PROMPT, RUN_LABEL
    global ROOT_DIR, _SAVE_HISTORY, FFT_LATTICE, LATTICE_INTERCEPT

    ROOT_DIR = Path(args.root_dir)
    _SAVE_HISTORY = args.save_history
    CF_MODES = {"fft": args.fft_mode, "sam": args.sam_mode, "raft": args.raft_mode}
    CF_SEED = args.cf_seed
    CF_ROOT = Path(args.cf_root) if args.cf_root else ROOT_DIR / "counterfactual"
    FFT_THRESHOLD = args.fft_threshold
    RUN_LABEL = args.run_label or (
        "baseline" if set(CF_MODES.values()) == {"real"}
        else "_".join(f"{k}-{v}" for k, v in CF_MODES.items() if v != "real"))

    # regenerate every threshold-dependent string
    GetFrequencyAnalysis.description = cf_tools.fft_tool_description(FFT_THRESHOLD)
    SYSTEM_PROMPT = _system_prompt(FFT_THRESHOLD)
    FFT_LATTICE = args.fft_lattice
    LATTICE_INTERCEPT = args.lattice_intercept
    if FFT_LATTICE:
        GetFrequencyAnalysis.description = GetFrequencyAnalysis.description.rstrip() + fft_lattice_tool.LATTICE_DESC_NOTE
        SYSTEM_PROMPT = SYSTEM_PROMPT + fft_lattice_tool.LATTICE_PROMPT_NOTE


def main():
    args = build_parser().parse_args()
    if not HAS_QWEN_AGENT:
        raise SystemExit(
            "qwen-agent is not installed — this harness only runs inside the pod venv "
            "(see commands.txt). The module itself imports fine for --help / offline tests.")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from filelock import FileLock
    from tqdm import tqdm

    _apply_config(args)

    global _HINT_MAPPING
    if args.ins:
        _HINT_MAPPING = _build_hint_mapping()
    else:
        print("INS disabled (use --ins to enable hint questions)")

    print(f"Counterfactual modes: {CF_MODES} | seed={CF_SEED} | "
          f"threshold={FFT_THRESHOLD} | label={RUN_LABEL}")
    print(f"Counterfactual artefact root: {CF_ROOT}")
    CF_ROOT.mkdir(parents=True, exist_ok=True)

    if args.out is None:
        os.makedirs(os.path.join(args.storage_path, "results", "counterfactual"), exist_ok=True)
        args.out = os.path.join(args.storage_path, "results", "counterfactual",
                                f"cf_{RUN_LABEL}.json")

    print("Building video cache...")
    cache = configure_resolver(ROOT_DIR)
    n_ambiguous = sum(1 for v in cache.values() if len(v) > 1)
    print(f"Video cache built: {len(cache)} videos ({n_ambiguous} ambiguous basenames)")

    lock_file = args.out + ".lock"
    output_lock = FileLock(lock_file)

    def safe_locked_write(result_entry=None):
        nonlocal output_lock, lock_file
        for attempt in range(3):
            try:
                with output_lock:
                    if os.path.exists(args.out):
                        with open(args.out, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            current_data = json.loads(content) if content else []
                    else:
                        current_data = []
                    if result_entry is not None:
                        current_data.append(result_entry)
                        with open(args.out, 'w', encoding='utf-8') as f:
                            json.dump(current_data, f, indent=2, ensure_ascii=False)
                    return current_data
            except OSError as e:
                if e.errno == 116:
                    print(f"[WARN] Stale file handle on lock (attempt {attempt+1}), recreating...")
                    output_lock = FileLock(lock_file)
                else:
                    raise
        raise OSError("Failed to acquire lock after 3 retries")

    # SURGICAL PATCH: allow an absolute --questions_file (cf subsets live outside storage)
    questions_path = args.questions_file if os.path.isabs(args.questions_file) \
        else os.path.join(args.storage_path, args.questions_file)
    with open(questions_path) as f:
        all_questions = json.load(f)
    print(f"{len(all_questions)} total questions")

    if os.path.exists(args.out):
        existing = safe_locked_write()
        done_pairs = {(o["video"], o["question"]) for o in existing
                      if isinstance(o, dict) and "answer" in o}
        questions = [q for q in all_questions
                     if (q["content"][0]["video"], q["content"][1]["text"]) not in done_pairs]
        print(f"Resuming: {len(existing)} done, {len(questions)} remaining")
    else:
        questions = all_questions
        print(f"Starting fresh: {len(questions)} questions")

    if not questions:
        print("All questions already processed!")
        return

    llm_cfg = {
        'model_type': 'qwenvl_oai',
        'model': f'{args.storage_path}/models/Qwen3-VL-30B-A3B-Instruct',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': args.top_p,
            'top_k': args.top_k,
            'temperature': args.temperature,
            'max_input_tokens': args.max_input_tokens,
        }
    }

    print(f"Starting with {args.num_workers} threads (per-thread bot instances)...")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_item, item, llm_cfg): item
                   for item in questions}

        pbar = tqdm(total=len(futures), desc=f"cf/{RUN_LABEL}")
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            safe_locked_write(result)
            pbar.update(1)
            if completed % 50 == 0:
                pbar.set_postfix({"last_time": f"{result.get('time_seconds', '?')}s",
                                  "last_answer": result.get('answer', '?')[:10]})
        pbar.close()

    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        times = [r["time_seconds"] for r in results if "time_seconds" in r]
        if times:
            print("\n=== Timing Summary ===")
            print(f"Total: {len(times)} questions")
            print(f"Total wall time: {sum(times):.0f}s ({sum(times)/60:.1f}min)")
            print(f"Mean: {sum(times)/len(times):.1f}s | Min: {min(times):.1f}s | Max: {max(times):.1f}s")


if __name__ == "__main__":
    main()
