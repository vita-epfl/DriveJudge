# Handoff — balanced real/generated benchmark (v2)

Everything needed to run the GPU sweep is already built and committed under
`src/real_vs_generated/v2/`. **No dataset building is needed on the cluster** —
the question JSONs are static files with absolute pod paths baked in.

TL;DR: upload 1000 real clips → run 9 jobs → send back 9 JSONs.

---

## 0. Why

Reviewers rejected the "Reality understanding" numbers because that category of
the benchmark contains **only generated clips** — a model that always answers
*Generated* tops it without looking at the video. The 100-clip v1 ablation
showed the opposite failure (base VLMs answer *Real* ~100 % of the time) but is
far too small to quote. v2 is the balanced, properly-powered version:
**1000 real / 1000 generated**, 1416 scenario clusters, with trivial baselines
and cluster-bootstrap CIs.

---

## 1. What to upload

**Only the real clips.** Every generated clip in v2 is already on the PVC under
`final_dataset/cosmos-drive-dreams/` and `final_dataset/cosmos-predict1/`.

Upload the 1000 files listed in `v2/realgen_v2_real_clips.txt` from the CarCrash
`Normal` split to

```
/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/real/
```

**Keep the CarCrash numbering** (`000001.mp4`, `001711.mp4`, …) — the question
files reference `final_dataset/real/NNNNNN.mp4` verbatim. The 50 clips of the v1
study (`000001..000050`) are a subset of this list, so if `final_dataset/real/`
already holds them the upload is incremental.

Locally the source is
`/home/matthieu/Documents/Projet de master/Dataset/CarCrash/videos/Normal/`
(`local_path` in the ground truth points at each file). Rough size: ~3 GB.

```bash
# example — adjust to however you reach the PVC
SRC="$HOME/Documents/Projet de master/Dataset/CarCrash/videos/Normal"
DST=/mnt/vita/scratch/vita-students/users/scharffe/VLM-eval/final_dataset/real
mkdir -p "$DST"
rsync -av --files-from=src/real_vs_generated/v2/realgen_v2_real_clips.txt \
      "$SRC/" "$DST/"
```

Sanity check after the upload — all three must print 2000:

```bash
python - <<'PY'
import json, os
gt = json.load(open('.../v2/groundtruth_realgen_v2.json'))
print(len(gt), sum(os.path.isfile(r['video']) for r in gt))
PY
```

Also upload the repo as usual (`code/real_vs_generated/` must ship with it, the
`v2/` sub-folder included).

---

## 2. What to run

Exact invocations: the **"v2"** section at the bottom of
`src/real_vs_generated/commands.txt`. Nine runs, all reusing the patched
scripts already in `src/real_vs_generated/scripts/` with the same batch sizes /
TP sizes as the v1 sweep:

| # | System | question file | output |
|---|---|---|---|
| 1 | Qwen3-VL | `questions_realgen_v2.json` | `Qwen3VL_realgen_v2.json` |
| 2 | Qwen3-VL + hints | `questions_realgen_v2_hints.json` | `Qwen3VL_hints_realgen_v2.json` |
| 3 | Qwen3-Omni | `questions_realgen_v2.json` | `Qwen_omni_realgen_v2.json` |
| 4 | Cosmos-Reason | `questions_realgen_v2.json` | `Cosmos_Reason_realgen_v2.json` |
| 5 | LLaVA-OneVision | `questions_realgen_v2.json` | `llava_onevision_realgen_v2.json` |
| 6 | InternVL3.5-30B | `questions_realgen_v2.json` | `InternVL3_5-30B_realgen_v2.json` |
| 7 | InternVL3.5-8B | `questions_realgen_v2.json` | `InternVL3_5-8B_realgen_v2.json` |
| 8 | Qwen3-VL + agent (v4o) | `questions_realgen_v2_agent.json` | `v4o_realgen_v2.json` |
| 9 | Qwen3-VL + agent + hints (v4p) | `questions_realgen_v2_agent_hints.json` | `v4p_realgen_v2.json` |

Launch the pod with `cluster/run_job_realgen.sh` as for v1.

### Short on GPU time? Run the pilot first

The message lists are ordered **pilot first**: the first 1000 entries are a
nested, label-balanced (500 real / 500 generated) random half. A
`questions_realgen_v2_pilot*.json` copy of every question file is shipped
because none of the eval scripts has a `--limit` flag. Run the `_pilot`
variants, write to `<Model>_realgen_v2_pilot.json`, and the analysis reruns
unchanged with `--pilot-only`. Pilot results are a strict subset of the full
run, so nothing is wasted if you later run the full 2000.

### Agent runs — read this first

The v4 agent used to build its video cache by `rglob`-ing `final_dataset` and
keying on the **basename**, last writer wins. 190 generated basenames occur in
more than one pool (dd / p1 / p2), and the VLM hands the tools a *bare
filename*, so `get_motion_info`, `get_masks` and `get_frequency_analysis` could
silently analyse a *different* clip than the one the question was about.

* **Fixed.** `src/agentic/video_resolve.py` maps a basename to *every* match and
  disambiguates against the clip the worker thread is currently on. Both
  `src/agentic/main_multi_tools_v4.py` and the copy in `scripts/` now use it,
  so runs 8 and 9 are safe as written. The pod-side `agentic/` folder on the
  PVC **must be refreshed** so `agentic/video_resolve.py` is there —
  `startup_realgen_agent.sh` copies `agentic/` next to `realgen_ablation/`, and
  the script imports `video_resolve` from it. It prints
  `Video cache built: N videos (190 ambiguous basenames)` at start-up; if that
  line is missing you are running the old copy.
* Belt and braces: the v2 sample also **hard-excludes** the 31 ambiguous
  cosmos-drive-dreams clips. 120 cosmos-predict1 clips remain ambiguous —
  cosmos-predict1 only has 494 benchmark clips and 337 unambiguous ones, so
  excluding them would have made the target count unreachable. They carry
  `basename_collides: true` in `groundtruth_realgen_v2.json`, so the analysis
  can report with and without them.
* Real clips are safe: `000001.mp4` … `003000.mp4` collide with nothing in
  `final_dataset`.

---

## 3. What to send back

The nine result JSONs (whatever subset finished), unmodified, into

```
src/real_vs_generated/v2/results/
```

plus a one-line note per run saying whether it was the full 2000 or the 1000
pilot, and which agent script was used. Result shape is unchanged from v1 —
baseline `{video, question, answer}` with the `Feedback::: / Answer:` text, and
agent `{video, question, evaluation, answer, history}`.

Then, locally:

```bash
.venv/bin/python src/real_vs_generated/analyze_realgen.py --mode v2 \
    --gt src/real_vs_generated/v2/groundtruth_realgen_v2.json \
    --results-dir src/real_vs_generated/v2/results \
    --out results/realgen
```

which writes `results/realgen/realgen_metrics_v2.parquet`,
`realgen_summary_v2.json`, `realgen_v2.tex` and
`report/figures/fig_realgen_balanced.pdf`. Add `--pilot-only` for the pilot
subset (outputs get a `_pilot` suffix).

---

## 4. What is in the box

```
v2/
  questions_realgen_v2.json              2000  baseline prompt (Feedback::: template)
  questions_realgen_v2_hints.json        2000  baseline + 3 reality hints
  questions_realgen_v2_agent.json        2000  bare question           (v4o)
  questions_realgen_v2_agent_hints.json  2000  bare question + hints   (v4p)
  questions_realgen_v2_raw.json          2000  Questions_raw.json shape (GPT / Gemini batch)
  questions_realgen_v2_pilot*.json       1000  the same five, pilot only
  groundtruth_realgen_v2.json            2000  one record per clip
  realgen_v2_real_clips.txt              1000  CarCrash files to upload
  realgen_v2_summary.md                        stratum tables
  results/                                     drop the result JSONs here
```

Composition (`realgen_v2_summary.md` has the full tables):

| | n | clusters |
|---|---|---|
| real (CarCrash Normal, 50 carried over from v1) | 1000 | 1000 |
| cosmos-drive-dreams (4 benchmark folders x 8 weathers) | 566 | 214 |
| cosmos-predict1 (7 weathers + 7b / 14b) | 434 | 202 |

`cosmos-predict1` has only 494 benchmark clips and 434 of them survive the
`<=3 clips per scenario_id` cap, so an exact 500/500 generated split is
impossible; the builder keeps the cluster cap and moves the 66-clip deficit to
`cosmos-drive-dreams`. Rebuild with `--shortfall relax` to trade the cap for the
source balance.

Rebuilding is deterministic (`--seed 42`) and needs
`results/fft/manifest.parquet`:

```bash
.venv/bin/python src/real_vs_generated/build_realgen_v2.py \
    --n-real 1000 --n-gen 1000 --pilot 500 --seed 42 \
    --max-per-cluster 3 --out src/real_vs_generated/v2/
```
