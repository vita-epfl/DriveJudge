# GPT evaluation

OpenAI-powered VQA for the `final_dataset` clips, with structured outputs
and prompt caching on the shared image prefix (many questions share the same
video). Default model: `gpt-5.6-sol` (`--model`, or `OPENAI_MODEL` in `.env`);
the earlier paper run used `gpt-5.4-mini`.

## Setup

1. `cp example.env .env` (in this folder) and fill in `OPENAI_API_KEY`.
2. Install the extras into your venv, from the repo root:
   `pip install -r requirements-closed-source.txt`.
3. Build `dataset/dataset_final/Questions_raw.json` with
   `src/data_preparation/prep_data_direct_prompting.py`, and point
   `LOCAL_DATASET_ROOT` at your local copy of the clips (default:
   `../final_dataset` next to the repo checkout; see `path_fix.py`).

**Run every command from the repo root**: the default `--questions` path
(`./dataset/dataset_final/Questions_raw.json`) is relative to the working
directory. Batch state, results and retry dirs always land in this folder,
whatever the working directory.

## 10-question pilot (live mode, results in seconds)

```bash
python src/evaluation/closed_source/gpt/run_gpt.py --mode live --limit 10
```

Output goes to `results/<stem>_live_results.json` (`<stem>` = sanitised model
id, or `gpt` for the historical gpt-5.4-mini run). Each item has `video`,
`question`, `evaluation`, `answer`, `history`, `latency_s`, `usage`
(input / cached / output tokens), and `custom_id`, the shape
`src/analysis/answer_analysis_agent.py` already parses.

**Check the tokens per request before going further.** Mean
`usage.input_tokens` tells you which image resolution the model actually got
(see *Image detail* below).

## Full run (batch mode, async: nothing keeps running locally)

The org's enqueued-token limit (2 M) rejects a burst of batches, so the
parts are submitted **one at a time** by `cycle.sh`:

```bash
python src/evaluation/closed_source/gpt/run_gpt.py --mode batch --prepare-only
bash src/evaluation/closed_source/gpt/cycle.sh --loop 600   # submit → poll → merge → retry, every 10 min
```

`--prepare-only` writes the JSONL parts to `retry_<stem>/jsonl/` (auto-chunked
so each stays under OpenAI's 200 MB input-file cap; `--max-bytes-per-file`,
default 150 MB) plus `retry_<stem>/retry_state.json`. `cycle.sh` (pick the
run with `STEM=<stem>`) submits the next part once the previous one is done,
retrieves results, merges them by `custom_id` into
`results/<stem>_all_results.json`, and resubmits failed requests. It runs
Python through `uv run`, so it needs [uv](https://docs.astral.sh/uv/) and the
repo `.venv`. Log: `cycle.log` in this folder.

Without `--prepare-only`, `run_gpt.py --mode batch` uploads and submits every
part at once and tracks them in `batches/batch_state_<stem>.json`; then
`python src/evaluation/closed_source/gpt/retrieve_batch.py [--wait]` merges
the outputs. That is only safe when the whole run fits under the
enqueued-token limit.

Individual batches finish in minutes to a few hours (SLA 24 h).

## Image detail

`--image-detail low` (default) asks for one 85-token thumbnail per frame;
`high` keeps native-resolution tiles, which fine-grained artifact questions
need.

> ⚠️ **Not every model honours `detail`.** `gpt-5.4-mini` ignored
> `detail=low` and was billed at high detail (~1058 tokens per 1280×704
> frame, ~5.5 k input tokens per request). `gpt-5.6-sol` honours it
> (~1.1 k tokens per request at `low`). Compare the mean input tokens per
> request with earlier runs before letting a full run continue: it is the only
> place where a silent resolution change shows up.

## Prompt caching

System prompt + the 5 video frames come first in every request; the
question text comes last. Since ~4.6 questions share each video, OpenAI's
automatic prompt cache hits on the image prefix for the 2nd–Nth question
per video. The `usage.cached_input_tokens` field in the results shows how
much actually hit cache.

## Scoring

```bash
python src/evaluation/closed_source/gpt/analysis/run_analysis.py --help
```

writes `results/<model>_analyzed.parquet` at the repo root (same 10-column
schema as the other closed-source arms).
