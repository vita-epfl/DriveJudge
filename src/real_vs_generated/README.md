# 🧮 real_vs_generated/

A self-contained **real-vs-AI-generated** ablation — the controlled experiment behind the headline
"models just say *real*" finding. 100 clips: **50 real 🟢 / 50 generated 🔴** (deterministic, seed 42).

| File / dir | What it is |
|:---|:---|
| `build_realgen_dataset.py` | builds the 100-clip subset + its question JSONs |
| `groundtruth_realgen.json` | the Real / Generated label per clip |
| `scripts/` | surgical copies of the eval + agent scripts (patched to accept an absolute `--questions_file`) + `startup_realgen_agent.sh` |
| `analyze_realgen.py` | computes per-class (Real vs Generated) accuracy |
| `results/` | one JSON of model answers per run |
| `realgen_accuracy_summary.json` | the final per-model Real/Generated accuracy table |
| `commands.txt` | the exact RunAI pod commands to reproduce the whole sweep |

**The result:** base VLMs score ~96–100 % on real clips but **0–22 % on generated ones**; the
Qwen3-VL **agent** reaches **64–70 %** on generated clips. See the [root README](../../README.md).
`commands.txt` refers to our cluster launcher (`cluster/run_job_realgen.sh`, not published); any
machine with the GPU setup from the root README can run the same commands.

---

## v2 — the balanced 1000 / 1000 extension

v1 is too small to quote and the paper's "Reality understanding" category is all-generated
(so *"always Generated"* wins it for free). **v2** is the properly-powered answer:
**1000 real 🟢 / 1000 generated 🔴**, stratified, ≤ 3 clips per scenario cluster, with a nested
500 + 500 pilot as the first half of every message list.

| File / dir | What it is |
|:---|:---|
| `build_realgen_v2.py` | builds the 2000-clip subset from `results/fft/manifest.parquet` |
| `v2/` | the question JSONs (5 prompt variants × full + pilot), ground truth, upload list, stratum tables |
| `HANDOFF_v2.md` | what a teammate has to upload / run / send back |
| `commands.txt` | the "v2" section holds the exact pod commands |
| `analyze_realgen.py --mode v2` | balanced accuracy / per-class recall / F1 / MCC with cluster-bootstrap CIs, plus the trivial and FFT-threshold baselines |

```bash
.venv/bin/python src/real_vs_generated/build_realgen_v2.py \
    --n-real 1000 --n-gen 1000 --pilot 500 --seed 42 \
    --max-per-cluster 3 --out src/real_vs_generated/v2/

.venv/bin/python src/real_vs_generated/analyze_realgen.py --mode v2 \
    --gt src/real_vs_generated/v2/groundtruth_realgen_v2.json \
    --results-dir src/real_vs_generated/v2/results --out results/realgen
```

`analyze_realgen.py` with no flags still reproduces the v1 table byte-for-byte.
