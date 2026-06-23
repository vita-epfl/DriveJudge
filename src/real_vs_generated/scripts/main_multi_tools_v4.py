"""
DriveJudgeBench Multi-Tool Agent v4 (ThreadPool)
Tools: RAFT (optical flow), SAM (max 4 objects), FFT (anisotropy)
Timing: per-question wall-clock time logged in results

Uses ThreadPoolExecutor (not ProcessPoolExecutor) because:
- Tools (SAM/RAFT/FFT) need GPU 4 — threads share the process and load models once
- vLLM requests are HTTP/IO-bound, perfect for threading
- Avoids the multiprocessing + CUDA deadlock issue
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import gc
import json
import time
import threading
import argparse
import sys
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock

# SURGICAL PATCH: this file lives in realgen_ablation/scripts/ instead of
# agentic/, so add <pod_home>/agentic (for utils.raft, utils.sam) and
# <pod_home> (for fft.compute_fft) to sys.path BEFORE those imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agentic"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from utils.raft import process_video_raft
from utils.sam import process_video_sam

from fft.compute_fft import process_video_fft

################################
##### Global Configuration #####
################################
# SURGICAL PATCH: mount root is /mnt/vita/scratch/... on this cluster (PVC
# claimname=vita-scratch). v4 was written against a different mount layout.
ROOT_DIR = Path("/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset")
STORAGE_PATH = "/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval"
MAX_SAM_OBJECTS = 4

# GPU lock: SAM and RAFT load models on GPU 4, must not run concurrently or they OOM
_gpu_lock = threading.Lock()

print("Building video cache...")
_VIDEO_CACHE: dict[str, Path] = {}
for ext in ("*.mp4", "*.MP4"):
    for video_path in ROOT_DIR.rglob(ext):
        if video_path.is_file():
            _VIDEO_CACHE[video_path.name] = video_path
print(f"Video cache built: {len(_VIDEO_CACHE)} videos")


def get_absolute_path(identifier):
    # SURGICAL PATCH: legacy strip-of-`scratch` removed — this cluster mounts at
    # /mnt/vita/scratch/vita-students.
    if Path(identifier).exists() and Path(identifier).is_file():
        return Path(identifier)
    if '/' in identifier or '\\' in identifier:
        video_path = ROOT_DIR / identifier
        if video_path.is_file():
            return video_path
    if identifier in _VIDEO_CACHE:
        return _VIDEO_CACHE[identifier]
    raise FileNotFoundError(f"Video '{identifier}' not found in {ROOT_DIR}")


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
        video_path = get_absolute_path(identifier)

        parts = list(video_path.parts)
        idx = parts.index('final_dataset')
        parts.insert(idx + 1, 'raft')
        output_dir = Path(*parts).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        cached_flow = output_dir / f"flow_{video_path.stem}.mp4"
        if cached_flow.exists():
            return json.dumps({"motion_visualization_video": str(cached_flow)})

        with _gpu_lock:
            output_video_path = process_video_raft(video_path=video_path, output_dir=output_dir)
        return json.dumps({"motion_visualization_video": str(output_video_path)})


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
        video_path = get_absolute_path(identifier)

        parts = list(video_path.parts)
        idx = parts.index('final_dataset')
        parts.insert(idx + 1, 'sam')
        output_dir = Path(*parts).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        sam_output_dir = output_dir / video_path.stem
        if sam_output_dir.exists() and any(sam_output_dir.glob("object_*.mp4")):
            output_videos = sorted(sam_output_dir.glob("object_*.mp4"))
        else:
            with _gpu_lock:
                output_videos = process_video_sam(
                    prompt=params.get('prompt'), video_path=video_path, output_dir=output_dir)
                output_videos = [Path(p) for p in output_videos]

        if len(output_videos) > MAX_SAM_OBJECTS:
            output_videos = sorted(output_videos, key=lambda p: p.stat().st_size, reverse=True)
            output_videos = output_videos[:MAX_SAM_OBJECTS]

        video_messages = [f"Clip {i+1}: ![video]({p})" for i, p in enumerate(output_videos)]
        return f"Isolated {len(output_videos)} objects (max {MAX_SAM_OBJECTS}).\n" + "\n".join(video_messages)


###############################
##### Tool 3: FFT Anisotropy
###############################
@register_tool("get_frequency_analysis")
class GetFrequencyAnalysis(BaseTool):
    description = """
    Analyzes FFT frequency spectrum to detect AI-generation artifacts.
    Returns numeric anisotropy score with calibrated threshold and verdict hint.
    Anisotropy >= 0.048 suggests generated, < 0.048 suggests real.
    """
    parameters: List[Dict] = [{
        "name": "video_path", "type": "string",
        "description": "Path or filename of the video.", "required": True
    }]

    def call(self, params: str, **kwargs) -> str:
        params = json.loads(params)
        identifier = params["video_path"].strip()
        video_path = get_absolute_path(identifier)

        parts = list(video_path.parts)
        idx = parts.index('final_dataset')
        parts.insert(idx + 1, 'fft')
        output_dir = Path(*parts).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_path = output_dir / f"{video_path.stem}_fft_analysis.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                result = json.load(f)
        else:
            result = process_video_fft(video_path=video_path, output_dir=output_dir)

        if 'error' in result:
            return result['error']
        return result.get('interpretation', 'FFT analysis completed.')


###############################
##### Tool 4: Final Answer (thread-safe)
###############################
# Thread-local storage for final answers
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
        # Write to thread-specific temp file (thread id ensures no collisions)
        tid = threading.get_ident()
        temp_path = _FINAL_ANSWERS_DIR / f"answer_{tid}.json"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({'evaluation': evaluation, 'answer': answer}, f, indent=4, ensure_ascii=False)
        return "Your message is saved, thank you !"


###############################
##### System Prompt
###############################
SYSTEM_PROMPT = """
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
- Trust the numbers: anisotropy >= 0.048 → likely generated, < 0.048 → likely real

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


###############################
##### Hint Questions (INS)
###############################
def _build_hint_mapping():
    """Load secondary/hint questions from hint_questions.json."""
    hint_path = Path(__file__).parent.parent / "data_preparation" / "resources" / "hint_questions.json"
    if not hint_path.exists():
        # Try cluster path
        hint_path = Path(STORAGE_PATH) / "hint_questions.json"
    if not hint_path.exists():
        print("WARNING: hint_questions.json not found, INS disabled")
        return {}
    with open(hint_path) as f:
        data = json.load(f)
    mapping = {}
    for category_dict in data:
        for category_name, questions_list in category_dict.items():
            for q in questions_list:
                q_text = q.get('question', '').strip()
                hints = q.get('secondary_questions', [])
                if hints:
                    mapping[q_text] = hints
    print(f"Hint mapping loaded: {len(mapping)} questions with hints")
    return mapping

_HINT_MAPPING = {}  # Populated in main() if --ins flag is set

def _inject_hints(question: str) -> str:
    """Append hint questions to the main question if available."""
    # Strip the question to its core (remove trailing whitespace/newlines)
    q_core = question.strip().split('\n')[0].strip()
    hints = _HINT_MAPPING.get(q_core, [])
    if not hints:
        return question
    hints_text = "\n".join(f"  - {h}" for h in hints)
    return f"""{question}

Before answering, consider these guiding questions:
{hints_text}

Answer each hint question briefly, then use that evidence to determine your final answer. You MUST call `final_answer` with your evaluation and answer."""


###############################
##### Worker Function
###############################
def process_single_item(item: Dict[str, Any], llm_cfg: Dict) -> Dict[str, Any]:
    """Runs in a thread — creates its own bot instance (qwen-agent is not thread-safe)."""
    bot = Assistant(
        llm=llm_cfg,
        system_message=SYSTEM_PROMPT,
        function_list=['get_motion_info', 'get_masks', 'get_frequency_analysis', 'final_answer'],
        name='DriveJudgeBench v4'
    )
    content = item["content"]
    video_item = next(c for c in content if c["type"] == "video")
    text_item = next(c for c in content if c["type"] == "text")
    video_path = video_item["video"]
    # SURGICAL PATCH: legacy path strip removed — mount is /mnt/vita/scratch/vita-students.
    question = text_item["text"]

    # Inject hint questions (INS paradigm)
    question_with_hints = _inject_hints(question)

    messages = [{
        'role': 'user',
        'content': [{"video": video_path}, {"text": question_with_hints}]
    }]

    # Clear thread answer file
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
        return {
            "video": video_path, "question": question,
            "evaluation": err_msg[:500], "answer": "ERROR",
            "time_seconds": round(time.time() - t_start, 2)
        }

    t_elapsed = time.time() - t_start

    # Read answer from temp file
    evaluation = ""
    answer = ""
    if temp_path.exists():
        with open(temp_path) as f:
            ans_data = json.load(f)
        evaluation = ans_data.get("evaluation", "")
        answer = ans_data.get("answer", "")
        temp_path.unlink(missing_ok=True)

    if not answer:
        # Capture raw VLM output for post-processing
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
        raw_output = '\n'.join(raw_texts)[-1000:]  # last 1000 chars
        evaluation = f"final_answer not called. Raw output: {raw_output}" if raw_output else "final_answer tool was not called (no output)"
        answer = "MISSING"

    return {
        "video": video_path,
        "question": question,
        "evaluation": evaluation,
        "answer": answer,
        "time_seconds": round(t_elapsed, 2)
    }


###############################
##### Main
###############################
def main():
    parser = argparse.ArgumentParser(description="DriveJudgeBench v4 — Threaded RAFT+SAM+FFT")
    parser.add_argument("--storage_path", type=str, default=STORAGE_PATH)
    parser.add_argument("--questions_file", type=str, default="dataset/Questions_raw.json")
    parser.add_argument("--output_file_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_input_tokens", type=int, default=128000)
    parser.add_argument("--ins", action="store_true", help="Enable INS hint questions")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel threads")
    args = parser.parse_args()

    # Load INS hint questions if enabled
    global _HINT_MAPPING
    if args.ins:
        _HINT_MAPPING = _build_hint_mapping()
    else:
        print("INS disabled (use --ins to enable hint questions)")

    if args.output_file_path is None:
        os.makedirs(os.path.join(args.storage_path, "results"), exist_ok=True)
        args.output_file_path = os.path.join(args.storage_path, "results", "v4_results.json")

    lock_file = args.output_file_path + ".lock"
    output_lock = FileLock(lock_file)

    def safe_locked_write(result_entry=None):
        """Read-modify-write results with stale handle recovery."""
        nonlocal output_lock, lock_file
        for attempt in range(3):
            try:
                with output_lock:
                    if os.path.exists(args.output_file_path):
                        with open(args.output_file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            current_data = json.loads(content) if content else []
                    else:
                        current_data = []
                    if result_entry is not None:
                        current_data.append(result_entry)
                        with open(args.output_file_path, 'w', encoding='utf-8') as f:
                            json.dump(current_data, f, indent=2, ensure_ascii=False)
                    return current_data
            except OSError as e:
                if e.errno == 116:  # Stale file handle
                    print(f"[WARN] Stale file handle on lock (attempt {attempt+1}), recreating...")
                    output_lock = FileLock(lock_file)
                else:
                    raise
        raise OSError("Failed to acquire lock after 3 retries")

    # Load questions
    # SURGICAL PATCH: allow absolute --questions_file (for realgen ablation)
    questions_path = args.questions_file if os.path.isabs(args.questions_file) \
        else os.path.join(args.storage_path, args.questions_file)
    with open(questions_path) as f:
        all_questions = json.load(f)
    print(f"{len(all_questions)} total questions")

    # Resume logic
    if os.path.exists(args.output_file_path):
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
        'model': f'{STORAGE_PATH}/models/Qwen3-VL-30B-A3B-Instruct',
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

    # Process with ThreadPoolExecutor
    completed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_item, item, llm_cfg): item
                   for item in questions}

        pbar = tqdm(total=len(futures), desc="DriveJudgeBench v4")
        for future in as_completed(futures):
            result = future.result()
            completed += 1

            safe_locked_write(result)

            pbar.update(1)
            if completed % 50 == 0:
                pbar.set_postfix({"last_time": f"{result.get('time_seconds', '?')}s",
                                  "last_answer": result.get('answer', '?')[:10]})

        pbar.close()

    # Timing summary
    if os.path.exists(args.output_file_path):
        with open(args.output_file_path) as f:
            results = json.load(f)
        times = [r["time_seconds"] for r in results if "time_seconds" in r]
        if times:
            print(f"\n=== Timing Summary ===")
            print(f"Total: {len(times)} questions")
            print(f"Total wall time: {sum(times):.0f}s ({sum(times)/60:.1f}min)")
            print(f"Mean: {sum(times)/len(times):.1f}s | Min: {min(times):.1f}s | Max: {max(times):.1f}s")


if __name__ == "__main__":
    main()
