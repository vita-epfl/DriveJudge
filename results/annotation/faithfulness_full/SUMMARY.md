# Explanation faithfulness: all v4o answers (summary)

**What was done.** Sonnet 5 checked every justification the v4o agent wrote, 7371 answers on 1597 clips. Each check
used the rubric v2: false claim, missed event, internal consistency. The grader saw an 8-frame contact sheet per clip
and the true FFT tool value. The work ran as 80 batches of 20 clips, with 8 graders in parallel.
CIs are 95 % cluster bootstrap over base scenarios.

## Main result

| subset | hallucination (false claim or missed event) |
|---|---|
| all answers | **18.4 % [16.7, 20.1]** |
| clips opened correctly | **14.0 % [12.4, 15.8]** |
| clips hit by the filename bug | 57.2 % [51.0, 63.9] |
| answer correct / wrong | 13.5 % / 25.7 % |

The agent contradicts its own reasoning in only 1.8 % of answers.

| category | rate |
|---|---|
| Reality (real or generated) | 11.9 % |
| Visual | 13.5 % |
| Traffic laws | 20.6 % |
| Safety | 23.0 % |
| Spatial-temporal | 25.7 % |
| Artifacts | 29.3 % |

## What goes wrong

| claim type (approximate, from grader notes) | share of answers |
|---|---|
| motion: stopped, proceeded, overtook | 9.6 % |
| wrong FFT value quoted | 4.8 % |
| invented or denied scene content: missing traffic light, gibberish signs, melted cars | 3.9 % |

- On clips opened correctly, FFT values are quoted exactly: 0.06 % are wrong.
- Almost all wrong FFT values come from the filename bug.

## Grader reliability

- **Calibration:** on the 100 items of the earlier study the rate is 15 %, against 18 % (Sonnet) and 23 % (Opus) before.
- **Fresh graders:** three batches were regraded. The rate was 16.8 % against 16.1 % (n=298), with item-level κ 0.39.
- **Conclusion:** the overall rate is stable, but single-item verdicts are noisy, mostly for motion judged from stills.
  Report rates, not individual labels.

## Two findings to act on

1. **The filename bug is not small.** Colliding predict1 clips were resolved to predict2 files, which the FFT tool cannot
   detect, so the agent answered "Real".

   | v4o accuracy | clean clips | collision clips |
   |---|---|---|
   | real-or-generated question, predict1 | 0.82 | 0.43 |
   | all answers | 0.62 | 0.43 |

   The fix could add up to ≈ 1.9 points to v4o's overall accuracy. Gemini currently leads by 4.1 points. The teammate's
   736-item rerun is therefore a priority.
2. **Duplicate clips.** 42 groups of byte-identical clips are filed under different scenario names. The list is in
   `duplicate_clips.json`. Decide whether to deduplicate them or disclose them.

## Files

- Full tables: `faithfulness_full_summary.md` and `faithfulness_full_results.json`.
- Per-item grades: `graded_items.parquet`.
- Grader prompt: `src/annotation/faithfulness_full_grader_prompt.md`.
- Master record: `report/reports/REVISION_FINDINGS.md` §0 and §6.
