# Full-scale faithfulness grading — status / how to resume

- Items: `items.jsonl` (7371 v4o answers, 1597 clips); batches `batches/batch_000..079.json` (+ `batch_cal.json` = the 100 items
  of the earlier study, for grader calibration); sheets `sheets/`.
- Grader: Sonnet 5 subagents, prompt `src/annotation/faithfulness_full_grader_prompt.md`, one batch per subagent, 8 in parallel.
  Output `ratings/batch_NNN.csv`; a grader resumes a partial file (grades only missing ids).
- Pilot (`ratings/pilot_v0_batch_000.csv`, old prompt) excluded: grader used sibling weather variants and defaulted to N.
- Resume: a batch is done when its CSV row count == len(batch json). Relaunch missing/partial batches with the same prompt.
- Analysis: `.venv/bin/python src/annotation/analyze_faithfulness_full.py` → `faithfulness_full_summary.md`, `faithfulness_full_results.json`.
- Started 2026-09-29: wave 1 = cal + 000–006.
- Calibration (batch_cal, new prompt): hallucination 15/100 (old Sonnet v2 18, Opus v2 23) → aggregate rate reproduces;
  item-level κ vs old Sonnet 0.31, vs old Opus 0.29, vs author (20) 0.26 → report rates, not per-item verdicts.
  All 4 FFT misquotes found are on basename-collision clips (the agent quoted the other pool's clip) → bug is visible in text.
- Launched: 000–007 (as each finishes, launch the next index).
- Side finding (grader batch 003): the benchmark holds byte-identical clips under different scenario names
  (`duplicate_clips.json`: 45 md5 groups / 90 clips; 42 groups with different filenames, mostly agg_tk_0NN vs agg_tk_0NN+4/5).
  They count as separate scenarios in the cluster bootstrap and as separate items → report + dedupe decision for the paper.
- Grader strictness varies by batch (e.g. 004: 32 false claims / 0 missed; 005: 13 / 23; 003: 2 / 3). Plan at the end:
  re-grade 2–3 random batches with fresh graders (outputs `ratings_rep/`) → between-grader κ and rate spread; report the
  hallucination rate with a grader-variance component, and false_claim ∪ missed_event (labels are substitutable).
- Motion claims (stopped vs proceeded) drive most of the batch-to-batch spread: batch 023 flags 34 stop/no-stop claims,
  batch 025 flags none on the same scene family ("0.7 s frame spacing cannot rule out a brief stop"). At analysis time:
  classify each flagged note as motion / static-content (traffic light, sign text, objects) / FFT misquote (keywords),
  and report the static + FFT part as the robust hallucination rate, the motion part separately with the grader spread.
- KEY side finding (2026-09-29): the basename collisions are mostly cosmos-predict1 ↔ cosmos-predict2 (same names, e.g.
  `cross_red_003.mp4`); predict2 is FFT-undetectable (real-like anisotropy), so the agent read a "REAL" FFT value for
  those p1 clips. v4o reality-binary accuracy: p1 clean 0.822 (n=337) vs p1 collision 0.433 (n=157); all items: clean 0.62
  (n=6635) vs collision 0.428 (n=736). Upper bound of the fix ≈ +1.9 pp overall v4o accuracy → the 736-item rerun matters
  (Gemini − agent gap was +4.1 pp). Graders independently flag these as FFT misquotes.
- PAUSED 2026-09-29 (usage limit). Done: cal, 000–065 except those still running. Running at pause: 066–072 (may end
  partial; relaunch any whose CSV rows < batch items). Not started: 073–079. Then: replicate 2–3 batches (ratings_rep/),
  note-type classification (motion / static / FFT), run analyze_faithfulness_full.py, update REVISION_FINDINGS §6, commit.
- DONE 2026-09-29: all 80 batches + cal + 3 replicates (005/049/053). Final: 18.4 % [16.7, 20.1] overall, 14.0 % clean
  clips, 57.2 % collision clips; replicate rates 16.1 vs 16.8 %, κ 0.39. Findings in REVISION_FINDINGS §6.
