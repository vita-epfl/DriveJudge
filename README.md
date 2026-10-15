<div align="center">

# 🚗🔍 DriveJudge

### Can Vision-Language Models reliably judge generated driving simulations?

**A benchmark and tool-augmented evaluation framework for testing whether VLMs can judge safety, rule compliance, realism, artifacts, and temporal behavior in generated driving videos.**

[🤗 Dataset](https://huggingface.co/datasets/SeagullofLeman/DriveJudgeBench)

</div>

---

## 🧠 TL;DR

Generative driving simulators and visual world models can produce realistic-looking driving videos, but visual realism alone does not guarantee that a rollout is behaviorally correct. A generated video may contain unsafe maneuvers, traffic-rule violations, physically inconsistent motion, or subtle generation artifacts.

We introduce **DriveJudgeBench**, a benchmark of **1,597 curated synthetic driving clips** and **7,371 manually annotated video--question pairs** across six categories:

- reality detection
- artifact recognition
- safety assessment
- traffic-law compliance
- spatio-temporal reasoning
- visual understanding

Across a broad set of open- and closed-source VLMs, we find that off-the-shelf models are often unreliable judges of generated driving videos. They can recognize static scene cues such as traffic lights and road layout, but often fail to verify temporal behavior, detect artifacts, or identify traffic-rule violations.

To address this, we propose **DriveJudge**, a training-free tool-augmented VLM judge built on Qwen3-VL. DriveJudge grounds model decisions using optical flow, segmentation-based crop-and-zoom, and FFT-based frequency analysis, substantially improving reliability on safety-relevant and synthetic-content evaluation tasks.

<div align="center">

| 🏆 Model | Accuracy |
|:---|:---:|
| 🔒 **Gemini-3-Flash** | **64.0 %** |
| 🤖 **DriveJudge** | **60.3%** |
| 🤖 **DriveJudge + CoT** | 50.9 % |
| 🔒 GPT-5.6-sol | 39.8 % |
| 🔒 GPT-5.4-mini | 39.1 % |
| InternVL3.5-30B | 30.9 % |
| Qwen3-Omni-30B | 27.4 % |
| Cosmos-Reason-7B | 26.5 % |
| InternVL3.5-8B | 25.6 % |
| LLaVA-OneVision-7B | 23.9 % |
| Qwen3-VL-30B | 21.9 % |

*Overall accuracy on the 7,371 video–question pairs. 🔒 = closed-source API model (scored on the
7,240–7,282 pairs their APIs answered); 🤖 = our tool-augmented judge. Per-model outputs are in
[`results/`](results/); see [`analysis_notebooks/`](analysis_notebooks/) for the breakdowns.
The DriveJudge row comes from a rerun with a corrected system prompt; `results/v4o_analyzed.parquet`
is the original run (60.1 %).*

</div>

---

## 🧭 What's inside

```
DriveJudge/
├── 📂 src/                              the pipeline code
│   ├── data_preparation/                ground-truth question files → chat-formatted (video, question) JSONs
│   ├── evaluation/
│   │   ├── open_source/                 one inference script per local VLM (vLLM / transformers / lmdeploy)
│   │   └── closed_source/               GPT / Gemini / Claude via the providers' Batch APIs
│   ├── agentic/                         DriveJudge, the Qwen3-VL tool-using agent (+ utils/ RAFT·SAM, fft/,
│   │                                    counterfactual/ ablation)
│   ├── analysis/                        regex-clean raw outputs → join ground truth → accuracy (polars)
│   ├── annotation/                      inter-annotator reliability + explanation-faithfulness studies
│   └── real_vs_generated/               real-vs-generated study (100-clip v1, 2000-clip balanced v2)
├── 📓 analysis_notebooks/               the story in charts (benchmark, timing, questions, agent failures)
├── 📊 results/                          final per-model analyzed parquets + study outputs
├── 📝 report/                           LaTeX write-up, figures, accuracy summaries
└── 🗂️ dataset/dataset_final/            ground-truth question files (the rest of dataset/ is gitignored)
```

### 📚 Documentation

Each part of the pipeline has its own README:

| README | Covers |
|:---|:---|
| [`src/data_preparation/`](src/data_preparation/README.md) | building the chat-formatted question files |
| [`src/evaluation/open_source/`](src/evaluation/open_source/README.md) | local VLMs: one script per model + the settings we used |
| [`src/evaluation/closed_source/`](src/evaluation/closed_source/README.md) | API models: [GPT](src/evaluation/closed_source/gpt/README.md), [Gemini](src/evaluation/closed_source/gemini/README.md), [Claude](src/evaluation/closed_source/claude/README.md) |
| [`src/agentic/`](src/agentic/README.md) | DriveJudge, the tool-augmented agent |
| [`src/analysis/`](src/analysis/README.md) | parsing raw outputs and scoring against the ground truth |
| [`src/real_vs_generated/`](src/real_vs_generated/README.md) | the real-vs-generated study |
| [`src/annotation/`](src/annotation/README.md) | annotator agreement and explanation-faithfulness studies |
| [`analysis_notebooks/`](analysis_notebooks/README.md) | the analysis notebooks |

---

## 🚀 Getting started

### 1. Install

```bash
git clone https://github.com/vita-epfl/DriveJudge.git && cd DriveJudge
```

Each part of the pipeline has its own environment, fully described by one requirements file:

| Environment | File | For | Hardware |
|:---|:---|:---|:---|
| core | [`requirements.txt`](requirements.txt) | data prep, scoring, notebooks, dataset download | CPU |
| GPU | [`requirements-gpu.txt`](requirements-gpu.txt) | open-source models + the `vllm serve` behind DriveJudge | CUDA 12 |
| GPU + InternVL | [`requirements-internvl.txt`](requirements-internvl.txt) | the GPU env plus lmdeploy, for InternVL3.5 | CUDA 12 |
| agent | [`requirements-agent.txt`](requirements-agent.txt) | the DriveJudge agent and its tools (RAFT, SAM 3, FFT) | 1 GPU |
| API | [`requirements-closed-source.txt`](requirements-closed-source.txt) | GPT / Gemini / Claude batch runs | CPU |

All of them target Linux x86_64 and Python 3.12.

**GPU work, with Docker (recommended).** [`docker/Dockerfile`](docker/Dockerfile) builds the GPU (with
InternVL) and agent environments in one image (CUDA 12.8, Python 3.12); the repo is mounted, not
copied:

```bash
docker build -f docker/Dockerfile -t drivejudge .
docker run --gpus all -it -v "$PWD":/workspace/DriveJudge -v <STORAGE>:<STORAGE> drivejudge
# inside: python (GPU env) · /opt/agent-venv/bin/python (agent env)
```

**Without Docker**, one venv per environment:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # core; on a GPU box use
                                                                         # requirements-gpu.txt or -internvl.txt
python3.12 -m venv .venv-agent && .venv-agent/bin/pip install -r requirements-agent.txt
python3.12 -m venv .venv-api && .venv-api/bin/source .venv-api/bin/activate          # requirements-closed-source.txt, see Install
```

### 2. Get the dataset

The clips are on the Hugging Face Hub: [`SeagullofLeman/DriveJudgeBench`](https://huggingface.co/datasets/SeagullofLeman/DriveJudgeBench).
The scripts assume one **storage root** laid out like this:

```
<STORAGE>/
├── final_dataset/                 ← the HF dataset (cosmos-drive-dreams/, cosmos-predict1/, …)
├── dataset/Questions*.json        ← built by src/data_preparation/ (step 3)
├── models/                        ← local weights: Qwen3-VL-30B-A3B-Instruct/, sam3/, …
└── results/                       ← eval scripts write their raw results JSON here
```

```bash
huggingface-cli download SeagullofLeman/DriveJudgeBench --repo-type dataset \
    --local-dir <STORAGE>/final_dataset
```

The benchmark uses the `cosmos-drive-dreams/` and `cosmos-predict1/` subsets. Their **ground truth
ships with this repo**: [`dataset/dataset_final/questions_cosmos-drive-dreams.json`](dataset/dataset_final/questions_cosmos-drive-dreams.json)
and [`questions_cosmos-predict1.json`](dataset/dataset_final/questions_cosmos-predict1.json), one entry
per clip: `{video, qa_pairs: [{question, type, category, answer, possible_answers}]}`, with `video`
relative to the subset folder. Entries flagged `remove` were dropped from the benchmark. (The
`questions.json` files on the Hub are an early draft; use these.)

> ⚠️ **Paths are hardcoded to our cluster** (`/mnt/vita/…`). Point them at your `<STORAGE>`:
> the eval scripts and the agent take `--storage_path` (and the open-source ones `--model_path`),
> but a few module-level constants must be edited by hand:
>
> | File | Constant |
> |:---|:---|
> | `src/data_preparation/prep_data*.py` | `storage_path` (→ `<STORAGE>/final_dataset`) |
> | `src/agentic/main_multi_tools_v4.py` | `ROOT_DIR`, `STORAGE_PATH` |
> | `src/agentic/utils/sam.py` | `MODEL_PATH` (SAM 3 weights) |
> | `src/analysis/answer_analysis*.py` | `path_to_remove` (prefix stripped before the ground-truth join) |

### 3. Build the prompts

From the repo root, turn the ground truth into chat-formatted `(video, question)` messages, then
copy them to `<STORAGE>/dataset/` where the eval scripts look for them:

```bash
python src/data_preparation/prep_data.py                    # Questions.json      (open-source models)
python src/data_preparation/prep_data_direct_prompting.py   # Questions_raw.json  (DriveJudge, API models)
mkdir -p <STORAGE>/dataset && cp dataset/dataset_final/Questions*.json <STORAGE>/dataset/
```

### 4. Run a model

**Open-source VLM** (one script per model, see [`src/evaluation/open_source/`](src/evaluation/open_source/README.md)
for all of them and the settings we used), e.g. Qwen3-VL on 4 GPUs:

```bash
python src/evaluation/open_source/eval_qwen_3_VL.py --storage_path <STORAGE> \
    --model_path <STORAGE>/models/Qwen3-VL-30B-A3B-Instruct --tensor_parallel_size 4 \
    --batch_size 150 --max_new_tokens 2048 --num_frames 15
```

The raw answers land in `<STORAGE>/results/Qwen3VL_results.json`.

**DriveJudge** → [next section](#-drivejudge-the-tool-augmented-vlm-judge).
**API models** (GPT, Gemini, Claude) → [Closed-source evaluation](#-closed-source-evaluation).
Scoring the raw answers → [Results](#-results--notebooks).

---

## 🏗️ The pipeline

```
                   ┌────────────────────────┐
  GT questions +   │ src/data_preparation/  │   →  dataset/ … (video, question) message JSONs
  MP4 clips   ───► │     prep_data*.py      │
                   └────────────────────────┘
                            │
         ┌──────────────────┼─────────────────────────────────┐
         ▼                  ▼                                  ▼
 ┌──────────────────┐ ┌────────────────┐        ┌──────────────────────────┐
 │ evaluation/      │ │  src/agentic   │        │ evaluation/closed_source │
 │ open_source      │ │ Qwen3-VL +     │        │ GPT-5.4-mini /           │
 │ (vLLM/tfm)       │ │ tools @ :8000  │        │ Gemini (Batch API)       │
 └──────────────────┘ └────────────────┘        └──────────────────────────┘
         │                  │                                  │
         └──────────────────┴──────────────┬───────────────────┘
                                            ▼
                                  ┌───────────────────┐
                                  │   src/analysis/   │  regex-clean + ground-truth join (polars)
                                  └───────────────────┘
                                            ▼
                          📓 analysis_notebooks/  +  📝 report/
```

Every model is prompted to emit a fixed shape so the analysis regexes can parse it:

```
Feedback:::
Evaluation: <free-form reasoning>
Answer: <the actual answer>
```

> ⚠️ The output shape and the parsing regex in `src/analysis/answer_analysis.py` are coupled —
> change one and you change the other.

---

## 🤖 DriveJudge: the tool-augmented VLM judge

`src/agentic/main_multi_tools_v4.py` implements **DriveJudge**, the training-free tool-augmented judge reported in the paper.
It uses Qwen3-VL as the backbone and augments it with optical flow, segmentation, and FFT frequency analysis. 
It talks to the OpenAI-compatible API of a **local vLLM server that must already be running on
port 8000**. The server runs in the GPU environment, the agent in the agent environment (SAM 3
needs transformers 5, vLLM needs 4.57); see [Install](#1-install).

```bash
# 1. serve the backbone, GPU env (we used 4 GPUs for the server, a 5th one for the tools)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <STORAGE>/models/Qwen3-VL-30B-A3B-Instruct \
     --tensor-parallel-size 4 --max-model-len 32768 --port 8000 --trust-remote-code \
     --media-io-kwargs '{"video": {"num_frames": -1}}' --mm-processor-kwargs '{"fps": 24.0}' \
     --limit-mm-per-prompt '{"video": 1}'

# 2. once the port answers, run the agent, agent env (cwd must be src/agentic so utils/ and fft/ resolve)
cd src/agentic && CUDA_VISIBLE_DEVICES=4 <agent-python> main_multi_tools_v4.py \
    --storage_path <STORAGE> --questions_file dataset/Questions_raw.json --num_workers 4
```

`<agent-python>` is `/opt/agent-venv/bin/python` in the Docker image, `../../.venv-agent/bin/python`
otherwise.
For the paper we ran a pre-release transformers build from `main`; `requirements-agent.txt` pins
5.0.0, the first release with the same SAM 3 video API.

Tool weights: RAFT comes from `torchvision` (downloaded automatically); SAM 3 is loaded from
`<STORAGE>/models/sam3` (download [`facebook/sam3`](https://huggingface.co/facebook/sam3) there, or
edit `MODEL_PATH` in `utils/sam.py`). Raw results go to `<STORAGE>/results/v4_results.json`;
`postprocess_missing.py` recovers answers when the model never called `final_answer`. More in
[`src/agentic/README.md`](src/agentic/README.md).

It registers three perception tools and forces a structured verdict:

| Tool | What it gives the model | Source |
|:---|:---|:---|
| 🌀 `get_motion_info` | optical-flow / motion summary | `agentic/utils/raft.py` (RAFT) |
| 🎭 `get_masks` | object segmentation masks | `agentic/utils/sam.py` (SAM 3) |
| 🔬 `get_frequency_analysis` | 2-D FFT power spectrum | `agentic/fft/compute_fft.py` |
| ✅ `final_answer` | the structured `{ evaluation, answer }` verdict | — |

The FFT spectrum is the key tool that exposes generation fingerprints. `main_multi_tools_timing.py`
is the latency-instrumented variant used for the timing study.

<div align="center">
<img src="report/figures/raft_flow.jpg" width="240">&nbsp;&nbsp;
<img src="report/figures/sam_artifact_f1.jpg" width="240">&nbsp;&nbsp;
<img src="report/figures/hough_generated.jpg" width="240">
<br><i>optical flow · segmentation · spectral / Hough cues</i>
</div>

---

## 🔒 Closed-source evaluation

[`src/evaluation/closed_source/`](src/evaluation/closed_source/) runs the API models (**GPT**,
**Gemini**, **Claude**) through their providers' **Batch APIs** with structured outputs. Every
backend sees the same 5 frames (1 fps, base64 JPEG), the same system prompt and the same answer
schema. The flow is fully scripted (chunk → upload → poll → merge → retry), and runs from any
machine, no GPU needed:

```bash
source .venv-api/bin/activate          # requirements-closed-source.txt, see Install
cp src/evaluation/closed_source/gpt/example.env src/evaluation/closed_source/gpt/.env   # fill in OPENAI_API_KEY
export LOCAL_DATASET_ROOT=<STORAGE>/final_dataset    # where the clips are on this machine

# from the repo root (the default --questions is ./dataset/dataset_final/Questions_raw.json)
python src/evaluation/closed_source/gpt/run_gpt.py --mode live --limit 10   # cheap pilot first
python src/evaluation/closed_source/gpt/run_gpt.py --mode batch             # full run, async
```

`LOCAL_DATASET_ROOT` maps the cluster paths stored in `Questions_raw.json` to your local copy
(default: `../final_dataset` next to the repo checkout).

See [`src/evaluation/closed_source/README.md`](src/evaluation/closed_source/README.md) for the
three backends and [`gpt/README.md`](src/evaluation/closed_source/gpt/README.md) for the full
playbook (batch-size caps, token limits, billing gotchas). 🔑 Each backend reads its key from a
gitignored `.env` (copy the `example.env` template in its folder).

---

## 📊 Results & notebooks

**Scoring a run.** Copy the raw results JSONs from `<STORAGE>/results/` into the input directory
below. Each analysis script cleans every raw results JSON in that directory, joins
it to the ground truth on `(video, question)` and saves one parquet per model (details in
[`src/analysis/`](src/analysis/README.md)):

```bash
python src/analysis/answer_analysis.py          # free-text outputs (open-source models): dataset/dataset_final/results/
python src/analysis/answer_analysis_agent.py    # structured outputs (DriveJudge, API models): dataset/dataset_final/results_test/
```

The API runners also ship their own `analysis/run_analysis.py`, which writes straight to `results/`.

**Our results.** The notebooks below read the scored parquets in `results/`; open them with
Jupyter (core env) from inside `analysis_notebooks/`.

| Notebook | What it shows |
|:---|:---|
| [`analysis_notebooks/benchmark_analysis.ipynb`](analysis_notebooks/benchmark_analysis.ipynb) | overall + per-category accuracy across all models & agents |
| [`analysis_notebooks/benchmark_timing.ipynb`](analysis_notebooks/benchmark_timing.ipynb) | accuracy ↔ inference-time trade-off |
| [`analysis_notebooks/agentic_failure_analysis.ipynb`](analysis_notebooks/agentic_failure_analysis.ipynb) | where the agent goes wrong (tool misuse, no answer) |
| [`analysis_notebooks/questions_analysis.ipynb`](analysis_notebooks/questions_analysis.ipynb) | question diversity & difficulty |

Cleaned per-model outputs live in `results/*_analyzed.parquet`, one per model (`v4o` = DriveJudge,
`v4p` = DriveJudge + CoT, `Gemini` = Gemini-3-Flash, …); the LaTeX write-up, figures and
failure dumps live in [`report/`](report/). Run the notebooks **from `analysis_notebooks/`** so
their relative paths (`../dataset/…`, `../report/figures/…`) resolve.

---

## 🗂️ Data, secrets & git hygiene

- **Videos & model weights** are not in the repo: videos come from the
  [HF dataset](https://huggingface.co/datasets/SeagullofLeman/DriveJudgeBench). `dataset/` is
  **gitignored** except the two ground-truth files (see [Get the dataset](#2-get-the-dataset)).
- **Heavy closed-source artifacts** (base64 request batches, raw results, frame caches) and bulky
  regenerable analysis blobs are gitignored too; only the *code* and small summaries are tracked.
- **Secrets** — `secrets.env`, `robot_secret.json`, and every `.env` are gitignored. Use the
  `example.env` templates.

---

<div align="center">

*VITA lab.*

</div>
