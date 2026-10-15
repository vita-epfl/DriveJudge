# Explanation faithfulness — 100 agent explanations (v4o), two independent LLM raters (Claude Opus 5, Claude Sonnet 5)

Items: 50 forensic (Reality / Artifacts) + 50 behaviour (Safety / Traffic / Spatial-temporal) from `src/annotation/manifests/faithfulness_rubric.csv`;
each rater saw the explanation, the true FFT tool output for the clip, and 3 frames (10/50/90 %). Raters: `results/annotation/faithfulness_ratings/rater_{opus,sonnet}.csv`;
agreement: `results/annotation/agreement_faithfulness.json`. **Caveat**: LLM raters, not humans — a 20-item human spot check is planned.

| rubric column | rater A (Opus) | rater B (Sonnet) | agreement | κ |
|---|---|---|---|---|
| cites a tool output | 100 % Y | 100 % Y | 100 % | — |
| citation matches the tool output | 100 Y | 70 Y / 25 NA / 5 N | 70 % | 0.00 (NA-vs-Y convention on flow citations) |
| conclusion follows the stated evidence (3/2/1) | 69 / 23 / 8 | 93 / 4 / 3 | 72 % | 0.19 |
| hallucinated visual claim | 7 Y | 5 Y | 94 % | 0.47 |

- FFT-citing explanations: 50; quoted number + verdict match the real tool output for **48/50** according to both raters (no invented values).
- Hallucination flagged by at least one rater: 9 items (['0be58d4db797', '2f52ab2a7f27', '356d45aeca9a'] by both).
- Conclusion unsupported (=1) by at least one rater: 9 items (['356d45aeca9a', 'ba2fd7bccf19'] by both); partial or unsupported (≤2) by both raters: 7.
- Recurring pattern: on the two artefact questions the agent answers 'Yes' from a base rate ('the clip is generated, such artefacts are common') while reporting clean masks — an inference step, not a tool misquote.

| question | n | hallucination (any rater) | conclusion=1 (any) | conclusion≤2 (both) |
|---|---|---|---|---|
| Overall, how realistic is the scene (roads, m | 25 | 1 | 0 | 0 |
| Is it a safe way of driving for the ego car ? | 14 | 2 | 2 | 1 |
| Is it a real video or generated ? | 14 | 0 | 1 | 1 |
| Is the ego car following the traffic rules ? | 13 | 1 | 1 | 0 |
| Overall, how safe is the driving of the ego c | 11 | 2 | 0 | 0 |
| Does any object appear or disappear without c | 6 | 0 | 2 | 2 |
| Is the ego car behavior consistent with visib | 5 | 2 | 2 | 1 |
| Has the ego car stopped ? | 5 | 0 | 0 | 1 |
| Do some objects change shape or appearance ? | 5 | 1 | 1 | 1 |
| On which side is the ego car overtaking ? | 2 | 0 | 0 | 0 |

Takeaway for the paper: tool citation fidelity is essentially perfect; unfaithfulness appears downstream, in the step from evidence to verdict (5–8 % visually false claims, 8–23 % under-supported conclusions depending on rater strictness). The counterfactual tool-perturbation runs (cluster) are the complementary causal test.

## Human spot check (author, 20 items: 14 flagged by a grader + 6 random; `faithfulness_spotcheck_human.json`)

| criterion | human vs grader A (Opus) | human vs grader B (Sonnet) |
|---|---|---|
| hallucinated visual claim | 75 % agreement, κ 0.53 [0.24, 0.80] | 65 %, κ 0.36 [0.11, 0.70] |
| conclusion follows (1–3, weighted κ) | 45 %, κ 0.04 [−0.12, 0.22] | 65 %, κ 0.22 [−0.11, 0.47] |

Human distribution on the 20: hallucination Y = 12 (graders: 7 / 5 on the same items) — the human, who knows the clips,
finds *more* false visual claims than the graders (e.g. a car changing shape, an overtake on the right the agent
missed), so the graders' 5–7 % hallucination rate is a lower bound. "Conclusion follows" = 3 for 17/20: the human
accepts inferences the strict grader rejects (e.g. "the clip is generated, hence artefacts are likely" on the artefact
questions). The criterion is under-defined; report it descriptively or drop it, and rely on the two crisp criteria
(tool-citation exactness, hallucination). Human notes are in the JSON.

## Rubric v2 re-grade of the 20 spot-check items (Opus, 8 frames; `rater_opus_v2_spotcheck.csv`, `agreement_faithfulness_human_v2.json`)

Definitions fixed: `consistent` = answer follows from the evidence *as the agent states it*; `false_claim` = statement
contradicted by the frames or a misquoted tool value; `missed_event` = salient, question-relevant event visible in the
frames but unmentioned. Human "hallucination" is compared with `false_claim ∪ missed_event`.

| criterion | human vs v1 grader (3 frames) | human vs v2 grader (8 frames) |
|---|---|---|
| hallucination (false claim or missed event) | 75 %, κ 0.53 [0.24, 0.80] | **85 %, κ 0.71 [0.40, 1.00]** |
| consistency (3 vs <3 ; Y/N) | 50 %, κ 0.07 | 80 % agreement, κ −0.08 (near-constant: human 17/20, grader 19/20 "consistent") |

v2 counts on the 20: consistent 19 Y; false_claim 8 Y (4 "stopped/did not run the light" while frames show it crossing,
1 invented green light, 3 "no artefact" while cars are visibly melted / road text is gibberish); missed_event 1 (overtake on
the right unmentioned). No FFT misquote. Remaining human/grader disagreements (3) are motion judgments (stopping vs
driving through) that still frames cannot settle.

**Protocol for the paper**: LLM grading of (i) tool-citation exactness and (ii) hallucination = false claim ∪ missed
event, ≥ 8 frames per clip; validated against the author on 20 items (κ 0.71). Report "conclusion follows" only as a
count of internal contradictions (1/20), not as a κ.

## FINAL — rubric v2 on all 100 items (8 frames; `rater_opus_v2.csv`, `rater_sonnet_v2.csv`, `faithfulness_v2_results.json`, figure `fig_faithfulness.png`)

| | Opus | Sonnet | both | either |
|---|---|---|---|---|
| false claim | 20 | 17 | 12 | 25 |
| missed event | 3 | 1 | 1 | 3 |
| hallucination (false claim ∪ missed event) | **23 % [16, 32]** | **18 % [12, 27]** | 13 % | 26 % |
| internally inconsistent | 4 | 6 | | |

Inter-grader κ (hallucination) 0.54 [0.33, 0.72], agreement 85 %. Forensic questions 14 % (both graders); behaviour
questions 32 % (Opus) / 22 % (Sonnet). Tool values quoted exactly in every item.
Calibration on the author's 20 items: κ 0.34 (this Opus run), 0.71 (dedicated Opus run), 0.53 (Sonnet) → graders are
moderately reliable and vary between runs; report the two-grader range and the both/either bracket.
Note: the 3-frame first pass (5–7 %) under-estimated hallucinations by ~3×; motion-related claims need many frames or video.
