# Annotation studies (paper revision E9 / E10 / E11)

Both re-annotation studies run on the **existing** FastAPI app (`../final_dataset_web_app`),
which streams the clips from HuggingFace. Nothing new to install.

## For the teammate — E9 reliability

There are two lists: the **core 100 clips** (mandatory) and an **optional extension** of 200
further clips. They are disjoint down to the scenario level (no two clips share an underlying
scene), so core + extension = 300 independent clips.

1. `git pull` this branch (`revision-experiments`).
2. In the app checkout: copy `src/annotation/manifests/reliability_videos_available.json`
   over the app's `videos_available.json` (keep a backup of the original).
3. In `app.py`, set `DATA_DIRS = ["cosmos-drive-dreams", "cosmos-predict1"]` — it currently
   says `cosmos-predict2`, and 31 of the 100 core clips live under `cosmos-predict1/artifacts`
   (75 of the 200 extension clips do). Leave `standard_questions.json` alone: E9 must ask
   exactly the original questions (`manifests/standard_questions_reliability.json` is the
   reduced set the app actually serves — 332 QA pairs over the core, 653 over the extension).
4. `uvicorn app:app --port 8000`, open http://localhost:8000, annotate all 100 core clips.
   Use the *remove* button for a clip that is genuinely unusable.
5. **Optional extension.** Once the core list is done, copy
   `manifests/reliability_ext_videos_available.json` over the app's `videos_available.json`
   and reload — do **not** reset `output/`: the app appends to the same
   `output/questions_<dataset>.json` files, so the core answers are kept and the extension
   answers accumulate next to them. Nothing else changes (same questions, same `DATA_DIRS`).
   Stopping part-way is fine: `agreement.py` scores exactly the clips that were completed and
   reports how far you got.
6. Send back `output/questions_cosmos-drive-dreams.json` and
   `output/questions_cosmos-predict1.json` (or the `/download_output` zip) — one pair of files
   whether you did 100 clips or 300.

## For the teammate — E11 structured state

Same loop, but copy **both** `structured_state_videos_available.json` →
`videos_available.json` *and* `standard_questions_structured_state.json` →
`standard_questions.json` (back up the originals; `DATA_DIRS` needs no change here — all 60
clips are cosmos-drive-dreams). Reset `output/` first so the two studies do not mix.
Both annotators do this one; send back the same two `output/questions_*.json` files.

## E10 faithfulness

`manifests/faithfulness_rubric.csv` — 100 rows, open in a spreadsheet, watch the clip via
`hf_video_url`, and fill the last five columns (`cites_tool` Y/N,
`citation_matches_output` Y/N/NA, `conclusion_follows` 1-3, `hallucination` Y/N, `note`).
Two raters each fill their own copy. Do not edit the other columns.

## Rebuilding / scoring (this repo, `.venv/bin/python`)

```
python src/annotation/build_manifest.py --task reliability      --n 100 --seed 0
python src/annotation/build_manifest.py --task reliability      --n 200 --seed 1 \
    --exclude src/annotation/manifests/reliability_videos_available.json \
    --name reliability_ext
python src/annotation/build_manifest.py --task structured-state --n 60  --seed 0
python src/annotation/build_manifest.py --task faithfulness     --n 100 --seed 0

python src/annotation/agreement.py --selftest
# --key defaults to reliability_key.json + reliability_ext_key.json (whichever exist)
python src/annotation/agreement.py --task reliability --files <teammate output dir> \
    --annotators ground_truth teammate --export-disagreements
python src/annotation/agreement.py --task structured-state --files <dir A> <dir B> \
    --annotators matthieu teammate
python src/annotation/agreement.py --task faithfulness --files a.csv b.csv
# adjudication of the disagreements: local web UI writing straight into the CSV, then convert + rescore
python src/annotation/serve_adjudication.py results/annotation/adjudication_reliability.csv --port 8765
python src/annotation/adjudication_to_jsonl.py      # -> reliability_adjudicated.jsonl + adjudication_summary.json
python src/annotation/agreement.py --task reliability --files results/annotation/annotator2 results/annotation/annotator2_ext \
    --annotators ground_truth teammate --export-disagreements   # picks up the jsonl -> "vs adjudicated"
```

`--exclude` (repeatable) drops the clips of an already-issued `videos_available.json`, and
every clip sharing a *scenario* with them, from the pool before stratification; `--name` sets
the output stem. That is how the extension stays nested and disjoint. The default reliability
command still reproduces the original three files byte-for-byte.

Results land in `results/annotation/agreement_<task>.{json,tex}` (+ a
`*_disagreements.md` table and a `manifests/*_adjudication_videos_available.json` app list
for the joint adjudication pass). For E9 annotator 1 is the original ground truth
(`manifests/reliability_key.json` + `manifests/reliability_ext_key.json`) — the user does not
re-annotate. The report opens with
`n clips annotated by the second rater: X of 300 (core: Y/100, extension: Z/200)` and, when
both lists are in play, adds a `per_list` breakdown so core-only and core+extension agreement
can be compared.
