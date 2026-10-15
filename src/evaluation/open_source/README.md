# 🧪 src/evaluation/open_source/

One inference script per **open-source** VLM in the benchmark. Each loads a prepared
`(video, question)` message JSON (from [`../../data_preparation/`](../../data_preparation/)), runs
**batched** inference, and writes a results JSON to the cluster `results/` dir.

| Script | Model | Backend |
|:---|:---|:---|
| `eval_qwen_3_VL.py` | Qwen3-VL-30B-A3B | vLLM |
| `eval_qwen_omni.py` | Qwen3-Omni-30B-A3B | transformers |
| `eval_cosmos_reason.py` | Cosmos-Reason-7B | transformers |
| `eval_InternVL3_5.py` | InternVL3.5-8B / 30B | lmdeploy |
| `eval_LLaVa.py` | LLaVA-OneVision-7B | vLLM |

The `*_timing.py` variants add latency instrumentation (they feed
`../../analysis/answer_analysis_timing.py` and `../../../analysis_notebooks/benchmark_timing.ipynb`).

## Settings we used

Run from the repo root, with `--storage_path <STORAGE>` (and `--model_path` / `--output_file_path`
as needed, see the [root README](../../../README.md#2-get-the-data)). `--batch_size` was tuned for
our cluster GPUs; lower it if you run out of memory, the results do not depend on it.

| Model | Flags |
|:---|:---|
| Qwen3-VL | `--batch_size 150 --max_new_tokens 2048 --tensor_parallel_size 4 --num_frames 15` |
| Qwen3-Omni | `--batch_size 50 --max_new_tokens 2048` |
| Cosmos-Reason | `--batch_size 10 --max_tokens 4096 --tensor_parallel_size 2` |
| LLaVA-OneVision | `--batch_size 5 --max_new_tokens 1024 --tensor_parallel_size 4` |
| InternVL3.5-8B | `--batch_size 4 --max_new_tokens 1024 --num_segments 8 --tp 4` |
| InternVL3.5-30B | `--batch_size 100 --max_new_tokens 1024 --num_segments 16 --tp 8 --session_len 65536 --cache_max_entry_count 0.8 --pytorch_backend True` |

InternVL needs the lmdeploy extras in [`requirements-internvl.txt`](../../../requirements-internvl.txt). Frame ablations change `--num_frames`
/ `--num_segments` (32 frames: Qwen3-VL `--batch_size 20`; InternVL-30B `--batch_size 75
--session_len 131072`). Hinted runs add `--questions_file dataset/Questions_with_hint.json`.
