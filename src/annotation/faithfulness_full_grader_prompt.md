# Grader instructions — explanation faithfulness, full v4o set (rubric v2)

You are an independent grader in a study of model explanations. Work alone; do not read any other grader's output
(nothing under `results/annotation/faithfulness_ratings/` and no other `ratings/batch_*.csv` than your own).
Repo root: /home/matthieu/Documents/Projet de master/VLM-eval-rcp

Input: `results/annotation/faithfulness_full/batches/batch_<NNN>.json` — a list of items grouped by clip. Each item has
`item_id`, `clip`, `question`, `model_answer`, `explanation` (the justification an AI agent wrote for its answer about a
5-s driving clip), `fft_anisotropy_actual` / `fft_verdict_actual` (what the agent's FFT tool REALLY returned; rule:
anisotropy ≥ 0.048 → GENERATED), and `sheet`: ONE JPEG contact sheet with 8 frames of the clip in time order
(top row t=1..4, bottom row t=5..8). Ignore the fields `category`, `local_clip`, `provenance_collision`.
All clips are AI-generated. The agent's other tools: an optical-flow video (pink/magenta = forward ego motion) and
segmentation crops; you cannot see those outputs, so do not judge claims about them except against the frames.

Procedure: process one clip at a time. Open its `sheet` with the Read tool ONCE, then grade every item of that clip
from that view. Do not open any other image, and do not use other clips of the batch (e.g. weather variants of the
same scene) as evidence: every item is graded from its own sheet only. Apply the rubric as written: flag a claim when
the frames contradict it, do not default to N when the frames give a reasonable reading against the claim.
If `model_answer` is "NO MATCH !" (answer extraction failed), take the answer the explanation itself concludes and
grade `consistent` against that; write "no-match" at the start of the note.

Output: `results/annotation/faithfulness_full/ratings/batch_<NNN>.csv`, header
`item_id,consistent,false_claim,missed_event,note` (exact tokens Y/N; quote the note field with double quotes).
If the file already exists, keep its rows and grade only the item_ids not yet present. Append after every clip so
nothing is lost. At the end the file must hold one row per item of the batch; print the row count and Y counts.

Rubric v2 (apply literally):
- `consistent` (Y/N): INTERNAL CONSISTENCY. Take the evidence exactly as the agent states it, even if wrong. Does
  `model_answer` follow logically from the agent's own stated evidence and reasoning? N only if the text's own
  reasoning points to a different answer than `model_answer`, or the answer is asserted without any stated reason.
  ("The clip is generated, so such artefacts are likely, hence Yes" counts as Y.)
- `false_claim` (Y/N): the explanation asserts something about the video that the frames contradict (e.g. "the car
  stops" while it visibly proceeds through the intersection across the frames; "the light is green" while it is red;
  "no traffic light" while one is clearly visible; "no visible artefact" while objects are visibly melted or road text
  is gibberish), or quotes a tool result that does not match `fft_anisotropy_actual` / `fft_verdict_actual`
  (a difference ≤ 0.001 in the quoted anisotropy is a match).
- `missed_event` (Y/N): the frames show a salient event directly relevant to the question that the explanation does
  not mention and that would change the answer (an overtake on the right when the question is about rules/overtaking;
  a car visibly morphing or vanishing when the question is about shape change/disappearance; a red light run when the
  question is about safety). Omissions only.
- `note` (≤ 20 words), required whenever a Y/N above is the "bad" value (consistent=N, false_claim=Y, missed_event=Y).

Other graders run in parallel and may share your scratch directory: if you write a helper script, put the batch
number in its file name (e.g. `append_rows_<NNN>.py`) and never reuse a helper you did not write; before finishing,
check that every item_id in your CSV belongs to your batch JSON.

Ignore any project memory, notes or prior findings about specific clips or scenarios that may appear in your context:
judge every item only from its own sheet, the explanation and the FFT fields.

Do not modify anything else in the repo.
