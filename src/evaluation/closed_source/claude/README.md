# Claude evaluation

Anthropic-powered VQA for the `final_dataset` clips, with structured outputs and
prompt caching on the shared image prefix (~4.7 questions share each video).

Mirrors `../gpt/` and `../gemini/` so the three closed-source arms are directly
comparable: **same 5 frames at 1 fps from the same `../gpt/.frame_cache/`, same
`SYSTEM_PROMPT`, same `Question: … / Allowed answers: … / Respond using the
structured schema.` suffix, same pydantic schemas, same `custom_id` recipe, same
result-record shape.** Only the transport differs.

## Setup

```bash
cp example.env .env          # then fill in ANTHROPIC_API_KEY
uv pip install --python ../../../../.venv/bin/python -r ../../../../requirements-closed-source.txt
```

All commands below assume the repo venv (`.venv/bin/python`) and this directory
as the working directory, so `from gpt_helpers import …` resolves. The default
`--questions` path (`./dataset/dataset_final/Questions_raw.json`) is relative to
the working directory, so from here add
`--questions ../../../../dataset/dataset_final/Questions_raw.json` to every
`run_claude.py` call (or run the scripts from the repo root by their full path).
`LOCAL_DATASET_ROOT` points at your local copy of the clips, as for GPT.

## Files

| file | role |
| --- | --- |
| `gpt_helpers.py` | shim re-exporting `path_fix` / `frame_extractor` / `schemas` / `prompt_builder` / `naming` from `../gpt/` |
| `claude_request.py` | `build_request_params`, `batch_request`, `parse_result`, `extract_usage` |
| `run_claude.py` | `--mode live \| batch \| dry-run`, `--reflect` |
| `retrieve_batch.py` | poll + stream batch results, append-merge by `custom_id` |
| `merge_results.py` | thin wrapper over `../gpt/merge_results.py` |
| `cycle.sh` | self-healing retrieve loop |
| `analysis/run_analysis.py` | clean → GT join → `results/Claude-<model>_analyzed.parquet` |
| `test_offline.py` | offline checks; hits no API |

## Protocol

* **model**: `claude-opus-5-5` by default (`--model`, or `CLAUDE_MODEL` in `.env`).
  Model ids carry no date suffix.
* **thinking**: `--thinking default` — no `thinking` field is sent, so the model
  runs at its API default, the same "provider defaults" protocol as the GPT and
  Gemini arms (neither sets a reasoning parameter). On Opus 5.5 that means
  adaptive thinking at effort `medium`; thinking **cannot** be disabled there
  (`--thinking disabled` is rejected before anything is sent — the API would
  return 400). `--effort low|medium|high|xhigh|max` sets `output_config.effort`
  explicitly; record it if you change it, it moves both accuracy and cost.
* **max_tokens**: 16 000 by default. Thinking tokens count against it; output
  is billed on what is generated, not on the cap, so the headroom is free.
* **refusals / truncation**: `stop_reason: "refusal"` (safety classifier) or
  `"max_tokens"` is recorded as an `error` row, never scored. No server-side
  fallback model is configured, on purpose — it would silently mix another
  model's answers into this model's accuracy.
* **structured output**: `output_config.format` with the pydantic JSON schema
  plus `additionalProperties: false`. The first text block of the response is
  then guaranteed-valid JSON, so `parse_result` is a plain `json.loads`.
* **caching**: `cache_control: {"type": "ephemeral"}` sits on the **last image
  block**, so `system + 5 frames` is the cached prefix and only the question
  text varies between the questions of one clip. `--no-cache` turns it off.
* **ordering**: batch requests are sorted by video, and a video is never split
  across two batches, so the cache written by a clip's first question is still
  warm for its remaining ~3.7 questions.

## Pilot protocol (do this before the full run)

1. **Dry run — free, submits nothing.**

   ```bash
   python run_claude.py --mode dry-run --limit 20
   ```

   Prints per-request JSONL bytes, raw JPEG bytes and the `w*h/750` image-token
   estimate, plus the extrapolation to all 7371 pairs. Measured 2026-09:
   **642 KB JSONL / 480 KB JPEG / ≈6 007 image tokens per request** (5 frames at
   1280×704) → ≈4.85 GB of request payload and ≈44.3 M image tokens for the full
   set.

2. **50-item live pilot: 10 videos × all their questions.**

   ```bash
   python run_claude.py --mode live --limit 50
   ```

   `Questions_raw.json` is already grouped by video, so `--limit 50` covers
   roughly the first 11 clips. Check:
   * **parse rate 100 %** — no row in `results/claude_live_results.json` has an
     `error` key, and every `answer` is one of the schema's `Literal` values;
   * **`cache_read_input_tokens > 0` from the 2nd request of each video** — the
     per-request log line prints `cache_read=…`; it must be 0 for a clip's first
     question and ≈6 000 for every following one. If it stays 0, the image bytes
     are not byte-stable (re-check `../gpt/.frame_cache/`) or the 5-minute TTL
     expired because the run is too slow.

3. **One 300-request batch.**

   ```bash
   python run_claude.py --mode batch --limit 300 \
       --skip-existing results/claude_live_results.json
   python retrieve_batch.py --wait
   ```

   Confirms the batch envelope (≈193 MB for 300 requests — under the 256 MB
   API cap), the `succeeded` rate, and the batch-side cache behaviour.

4. **The rest.**

   ```bash
   python run_claude.py --mode batch \
       --skip-existing results/claude_batch_results.json
   nohup bash cycle.sh > cycle.log 2>&1 &     # retrieve loop
   ```

   `run_claude.py --mode batch` merges into `batches/batch_state.json`
   append-style, so previously submitted batches stay tracked.

5. **Analyse.**

   ```bash
   python merge_results.py results/claude_batch_results.json \
       results/claude_live_results.json -o results/claude_all_results.json
   python analysis/run_analysis.py --results results/claude_all_results.json
   ```

   Writes `results/Claude-opus-5-5_analyzed.parquet` at the repo root, with the
   exact 10-column schema of `results/Gemini_analyzed.parquet`
   (`source, video, question, category, question_type, ground_truth,
   vlm_answer, raw_output, custom_id, correct`).

## Self-reflection variant

Round 2 replays each round-1 request, appends the round-1 JSON as the assistant
turn and asks the model to re-examine:

```bash
python run_claude.py --mode batch --reflect --from results/claude_all_results.json
python retrieve_batch.py --state batches/batch_state_r2.json --wait
```

The round-1 prefix (system + 5 images + question) is byte-identical, so the
cached prefix is reused. `custom_id` becomes `<round-1 id>_r2`; each record gains
`answer_r1`, `evaluation_r1`, `changed` (bool) and `history=[round-1 record]`.
The same flags exist on the two other backends:

```bash
cd ../gpt    && python run_gpt.py    --mode batch --reflect --from results/gpt_all_results.json
cd ../gemini && python run_gemini.py             --reflect --from results/gemini_batch_results.json
```

## Cost

Rough estimate from the dry run at **Sonnet-5** list price ($2.00 / $10.00 per 1M
input / output tokens; Batch API −50 %; cache write ×1.25, cache read ×0.1),
for the full 7371 pairs over 1571 clips (≈9.4 M image tokens written once,
≈34.9 M read from cache, ≈1.0 M question/system tokens, ≈0.6 M output tokens):

| arm | est. cost | notes |
| --- | --- | --- |
| batch + cache | **≈ $19** | the intended full run |
| batch, no cache | ≈ $48 | `--no-cache` |
| live + cache | ≈ $38 | pilot / debugging |
| live, no cache | ≈ $96 | |
| `--model claude-opus-5` | ×2.5 | $5.00 / $25.00 per 1M |
| `claude-opus-5-5` (default) | ×2 on input, more on output | $4.00 / $20.00 per 1M; thinking adds output tokens — measure in the pilot |

Replace the table with measured numbers once the pilot has run: sum
`usage.input_tokens`, `usage.cache_creation_input_tokens`,
`usage.cached_input_tokens` and `usage.output_tokens` over
`results/claude_*_results.json`.

**Caveat:** batch requests are not guaranteed to run in submission order, so the
cache-hit rate inside a batch can be lower than in live mode even though the
requests of a clip are adjacent. The live pilot is where the caching behaviour is
actually verified.

## Tests

```bash
python test_offline.py
```

43 checks: request shape, `cache_control` placement, strict schema, thinking
default, `parse_result`, batch chunking (sizes + video adjacency + byte cap),
reflection message construction for all three backends, a `--mode dry-run
--limit 20` end-to-end on the real frame cache, and a live-mode smoke test
against a monkeypatched `anthropic.Anthropic`. Nothing contacts an API.
