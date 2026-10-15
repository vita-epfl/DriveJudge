# 📦 src/data_preparation/

Turns the ground-truth question files (`dataset/dataset_final/questions_cosmos-drive-dreams.json`
and `questions_cosmos-predict1.json`, tracked in this repo) into
JSON files of **chat-formatted messages**, one per `(video, question)` pair, ready for the
evaluation scripts. Run from the repo root; all write into `dataset/dataset_final/`, and the eval
scripts read them from `<STORAGE>/dataset/`.

Edit `storage_path` at the top of each script first: it is prepended to every video path and must
point at your copy of the clips (`<STORAGE>/final_dataset`).

- `prep_data.py` — the canonical builder → `Questions.json` (the main benchmark question set, used
  by every `../evaluation/open_source/eval_*.py`).
- `prep_data_with_hints.py` — `Questions_with_hint.json` (adds the hint pool from
  `resources/hint_questions.json`; used by the hinted agent runs).
- `prep_data_direct_prompting.py` — `Questions_raw.json` (minimal prompt; used by the DriveJudge agent and the closed-source runners).

See the [root README](../../README.md) for how this feeds the pipeline.
