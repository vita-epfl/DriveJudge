# Gemini evaluation

Mirrors [`../gpt/`](../gpt/) on Google's Batch API (file-based JSONL upload): same 5 frames, same
system prompt, same structured-output schema, same result-record shape, so
`src/analysis/answer_analysis_agent.py` and `analysis/run_analysis.py` work unchanged. Pick the
model with `--model` (or `GEMINI_MODEL` in `.env`); the paper's **Gemini-3-Flash** row is the
`gemini-3-flash-preview` run (`--model gemini-3-flash-preview`, `results/Gemini_analyzed.parquet`).

## Setup

`cp example.env .env` (in this folder), fill in `GEMINI_API_KEY`, and install
`requirements-closed-source.txt`. `LOCAL_DATASET_ROOT` points at your local copy of the clips, as
for GPT. Run from the repo root (the default `--questions` path is relative to it).

## Run

```bash
python src/evaluation/closed_source/gemini/run_gemini.py --limit 2      # smoke test
python src/evaluation/closed_source/gemini/run_gemini.py                # full run: build + submit
bash   src/evaluation/closed_source/gemini/cycle.sh 600                 # poll + merge until done
```

Unlike the GPT flow there is no separate `--mode`: `run_gemini.py` always submits batches, tracked in
`batches/batch_state_<stem>.json`. `cycle.sh` (pick the run with `STEM=<stem>`; `STEM=gemini` for the
legacy run) calls `retrieve_batch.py`, merges finished batches into
`results/<stem>_batch_results.json` (latest answer wins per `custom_id`), and exits once every batch
is terminal; rerun it to resume. `--skip-existing <results json>` tops up a partial run without
resubmitting finished questions; `--reflect --from <results json>` runs the self-reflection round
(see [`../README.md`](../README.md)).

## Scoring

```bash
python src/evaluation/closed_source/gemini/analysis/run_analysis.py --help
```

writes `results/<model>_analyzed.parquet` at the repo root.
