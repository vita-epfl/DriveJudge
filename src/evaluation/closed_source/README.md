# 🔒 src/evaluation/closed_source/

The API models in the benchmark, each run through its provider's **Batch API** with structured
outputs. Videos are sampled at 1 fps (5 frames), base64-encoded, and sent inline. The flow is the
same for all three: **chunk → upload → poll → merge → retry**.

| Folder | Model | API |
|:---|:---|:---|
| [`gpt/`](gpt/) | GPT-5.6 Sol (`gpt-5.6-sol`; earlier run: GPT-5.4-mini) | OpenAI Batch API |
| [`gemini/`](gemini/) | Gemini 3 Flash (`gemini-3-flash-preview`) | Google Batch API |
| [`claude/`](claude/) | Claude Opus 5.5 (`claude-opus-5-5`) | Anthropic Message Batches API |

`claude/` reuses `gpt/`'s frame cache, system prompt and schemas verbatim (via its `gpt_helpers.py`
shim) so the three arms are directly comparable — see [`claude/README.md`](claude/README.md).
Two cross-cutting options exist on all three runners:

* **model-aware output names** — `--out-name` / `--results`. The historical runs (gpt-5.4-mini,
  gemini-3-flash-preview) keep their bare `gpt` / `gemini` filenames; every other model uses its
  sanitised model id (`results/gpt-5.6-sol_batch_results.json`,
  `batches/batch_state_gpt-5.6-sol.json`, …), so a new model never clobbers the old one. The
  `cycle.sh` loops pick the run with `STEM=<model id>` (default: the new model; `STEM=gpt` /
  `STEM=gemini` for the old runs).
* **`--reflect --from <round-1 results json>`** — a second round in which the model sees its own
  round-1 JSON and is asked to re-examine the frames. `custom_id` gains a `_r2` suffix; records gain
  `answer_r1`, `evaluation_r1`, `changed` and `history=[round-1 record]`.

Extra dependencies for these runs are pinned in [`requirements-closed-source.txt`](../../../requirements-closed-source.txt).

```bash
python src/evaluation/closed_source/gpt/run_gpt.py --mode batch --prepare-only   # write the parts
bash src/evaluation/closed_source/gpt/cycle.sh --loop 600   # submit one at a time → merge → retry
```

Each folder holds the runner (`run_gpt.py` / `run_gemini.py` / `run_claude.py`), batch helpers
(`retrieve_batch.py`, `merge_results.py`, `retry_failed.py`, `cycle.sh`), the shared request
builders, and an `analysis/` subfolder that writes the scored results to the top-level `results/`
(`GPT-5.4-mini_analyzed.parquet` / `Gemini_analyzed.parquet` for the paper's runs,
`<model>_analyzed.parquet` for newer ones). Run the runners from the repo root, and set
`LOCAL_DATASET_ROOT` to your copy of the clips (see [`gpt/README.md`](gpt/README.md#setup)).

🔑 API keys live in a gitignored `.env` in each backend folder (copy `example.env`):
`gpt/.env` → `OPENAI_API_KEY`, `gemini/.env` → `GEMINI_API_KEY`, `claude/.env` → `ANTHROPIC_API_KEY`. Heavy artifacts — request batches,
raw results, frame caches — are gitignored; only code + small summaries are tracked. See
[`gpt/README.md`](gpt/README.md) for the full playbook (token limits, the `detail=low` billing
gotcha, etc.).
