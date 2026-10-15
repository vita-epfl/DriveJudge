# DriveJudgeBench revision — master record of findings (2026-09-09 → 09-11; local work complete)

Single entry point. Branch `revision-experiments`. Each section: what was done, the numbers, where the evidence is, the decision
taken, what the paper must say. Detailed notes: `response_to_reviewers.md` (point-by-point), `fft_revision_findings.md`,
`results/annotation/faithfulness_summary.md`, `results/structured_state/summary.md`.

## 0. Corrections to the published numbers (apply before anything else)
- **Model name**: the "Qwen2.5-Omni" of the paper is `Qwen3-Omni-30B-A3B-Instruct` (checkpoint in `eval_qwen_omni.py`). Renamed everywhere on the branch.
- **Headline accuracies**: README/JSON used a first-match parser that captured an echoed prompt; corrected: Qwen3-Omni 10.4 → **27.4 %**, LLaVA 23.1 → 23.9, Cosmos-Reason 25.7 → 26.5, InternVL-30B 30.8 → 30.9. `report.tex` table was already right.
- **Per-folder baseline**: `aggressive_takeover` follow-rules baseline is 5.0 %, not 5.6 % (paper repeated v4o's figure).
- **Majority-class baseline = 76.3 %** (per question; always-"Generated" = 100 % on the reality binary) — above every model incl. Gemini (64.0). Headline metrics must move to macro accuracy / balanced accuracy / per-class F1 (all computed). Files: `results/stats/`.
- **Provenance bug in the agent results**: tools resolved clips by bare filename; 190 filenames exist in 2–3 pools; 188 benchmark clips / 736 answers affected; for 88 clips the anisotropy the agent quoted is the other pool's clip. Fixed (`src/agentic/video_resolve.py`); rerun list `src/agentic/counterfactual/rerun_collisions.json` (cluster, priority 1). **Effect is NOT small** (measured 2026-09-29): most collisions pair a cosmos-predict1 clip with the same-named cosmos-predict2 clip, which the FFT tool cannot detect, so the agent read a real-looking anisotropy and answered "Real". v4o reality-binary accuracy on predict1: 0.82 clean (n=337) vs 0.43 collision (n=157); all items 0.62 clean vs 0.43 collision (n=736) → the fix is worth up to ≈ +1.9 pp overall v4o accuracy (Gemini − agent gap = 4.1 pp). The explanations show it: 47 % of collision-clip answers quote the wrong FFT value (0.06 % on clean clips).
- **Duplicate clips**: 42 groups of byte-identical benchmark clips filed under different scenario names (mostly `aggressive_takeover`, `crossing_red_lights`; list `results/annotation/faithfulness_full/duplicate_clips.json`) — they count as independent scenarios/items. Decide: dedupe or disclose.

## 1. Statistics (E12) — done
Cluster bootstrap by base scenario (`src/analysis/stats.py`, `run_stats.py`). Design effect ≈ 5 → CI widths ×2.2 vs i.i.d.
All headline gaps stay significant: Gemini − agent = +4.1 pp [1.8, 6.4]; agent − GPT-5.4-mini = +20.8 pp [18.6, 22.9]; only Cosmos-Reason vs
InternVL-8B / Qwen3-Omni are indistinguishable. Macro accuracy, per-class recall/F1/MCC, Likert MAE + weighted κ available.
Files: `results/stats/*.parquet`, `stats_summary.json`, `tab_overall_ci.tex`; figures `fig_overall_ci`, `fig_per_category_ci`, `fig_pairwise_vs_v4o`.

## 2. Transportation breakdown (E11) — done
Per folder × question with CIs (`results/stats/per_folder_question.parquet`, `fig_folder_heatmap.pdf`). Overtake-side answers are a
left/right **convention artefact**: swapping predictions helps 8/10 models, +28.8 pp mean (v4p 6.8 → 88.7).
**Structured state exists**: drive-dreams violations were authored as ego trajectories in the Cosmos-Drive-Dreams editor
(`rds_hq/novel_pose`). 48/58 mapped scenarios hand-edited (median shift 110 m); 39/46 cross a controlled stop bar without stopping.
Per-clip labels stay visual (generator variance: green light, event cut); author review of 18 flagged scenarios: 16 show the
violation, 2 do not (`cross_red_025`, `cross_red_029` → **dropped 2026-09-11**, list in `results/structured_state/dropped_clips.txt`; local GT JSON patched, PVC copy + final tables to follow). Trajectory→clip chunk mapping is
not reliable; use scenario-level statements only. Files: `results/structured_state/`, `fig_structured_state_example.pdf`.
Paper must describe the editor pipeline (was never stated).

## 3. FFT tool (E7 + E8) — done
- Threshold 0.048 holds on native files: calibrated 0.0472 [0.0457, 0.0512] on 1589 non-benchmark clips (scenario-disjoint).
- Balanced 1000/1000 test: AUC **0.881** (original 36 clips: 0.975 — overstated). At 0.048: recall 0.81, FPR 0.20.
- **Container confound**: same clips re-encoded to one container (n10) → AUC **0.784**; FPR 0.20 → 0.35. Resizing adds its own
  fingerprint (real clips down/up-scaled → 96 % called generated), so "normalising" is not a fix; state the container.
- LOGO dd→p1 0.89 / p1→dd 0.87 (raw). Weather: Rainy +0.007 within scene (appendix line). Degradation grid (600 clips):
  stable to CRF/fps/codec/frame count, degrades only with downscaling (0.71–0.73); 5 frames suffice (`fig_robustness_summary`).
- **Lattice score** (new, `src/fft/lattice.py`): the VAE grid peaks at (±W/4, ±H/4) that real clips never show. n10 AUC 0.918;
  **anisotropy + lattice: 0.977 raw / 0.971 n10**, container-invariant; drive-dreams recall ≈ 1.0, predict1 ≈ 0.8.
  Rule: `1.87·(a−0.0484)/0.0107 + 4.09·(L−1.585)/1.324 + 1.58 ≥ 0`. Agent flag `--fft-lattice` (default off = published).
- predict2 dropped from the study (never in the paper). `detect_periodic_artifacts` in the tool is a dead feature (always 0).
Figures: `fft_spectra_panel`, `fft_roc_raw_vs_n10_aniso_mean`, `fft_roc_n10_aniso_vs_lattice`, `fft_robustness_summary`.

## 4. Balanced real/generated set (E6) — built, VLM runs pending (cluster)
1000 real (CarCrash Normal) / 1000 generated (566 dd / 434 p1), pilot 500/500, ≤3 clips per scenario, baselines + balanced metrics in
`analyze_realgen.py --mode v2`. Agent ROC via threshold sweep (`--fft-threshold` 0.036…0.060) and lattice arm. `src/real_vs_generated/v2/`, `HANDOFF_v2.md`.

## 5. Annotation reliability (E9) — done, paper-ready
**Protocol.** Original labels: one author (built the clips, wrote the questions). Reliability pass: second rater = teammate,
non-expert, blind to the original labels and to every model output, told that all clips are synthetic; same web app as the
original annotation (FR/EN), subjective questions only (safety yes/no + 1–3 score, realism 1–3, rule compliance, light
consistency, overtaking side, two artefact yes/no questions). 100 clips stratified by folder × question × model-disagreement
stratum, ≤1 clip per scenario, then 75 extra artefact clips → 175 clips, 494 answers, 432 matched to the ground truth.
Adjudication: the author reviewed every disagreement (119) with the clip, set a final label and a reason
(label_error / annotator_slip / ambiguous_clip / convention).

**Agreement** (`results/annotation/agreement_reliability.json`; CIs = item bootstrap, B = 2000):

| question | n | % agree | κ | note |
|---|---|---|---|---|
| all | 432 | 72 | **0.64 [0.58, 0.69]** | Krippendorff α 0.64 |
| safe driving yes/no | 68 | 96 | 0.91 | |
| safety score 1–3 | 68 | 85 | 0.74 (weighted 0.84) | |
| following traffic rules | 28 | 96 | 0.65 | prevalence-limited |
| overtaking side | 6 | 83 | 0.67 | |
| light consistency | 10 | 90 | 0.0 | 9/10 same class, κ undefined in practice |
| shape change (artefact) | 87 | 83 | 0.23 | GT ≈ 88 % Yes; PABAK 0.66 |
| appear/disappear (artefact) | 68 | 74 | 0.36 | PABAK 0.47; rater sees fewer artefacts |
| realism 1–3 | 97 | 28 | 0.03 | ρ = 0.61; anchor offset, κ 0.59 once aligned |

**Adjudication** (`adjudication_summary.json`): realism 70/70 disagreements = scale-anchor convention (rater used "3" as
"no visible defect", author never used 3 → report realism as ordinal, re-anchor the wording). Other 49: rater slip 40,
**ground-truth error 3**, ambiguous 4, convention 2. → **GT label-error rate 3/432 = 0.7 % [0.2, 2.0] (Wilson)**, all three on
artefact yes/no questions (GT said No, event is there). GT vs adjudicated κ 0.99; rater vs adjudicated κ 0.65 (acc 0.73).
Model accuracy on the 432 items moves < 0.7 pp under adjudicated labels (≤ 3 labels differ) — not re-run.
Borderline examples for the paper (`report/borderline_examples.md`, `report/figures/borderline/`): overtake on both sides
(`agg_tk_012`; side convention = "the forbidden side"), stop-line cut just before the line (`cross_stop_018_Foggy`),
safety score on a clip with a strong artefact (`negative/07b2e448…_Snowy`), `pedestrians_003` appear/disappear.

**What the paper says.** Annotation protocol + rater description above; κ table (overall substantial, safety near-perfect,
artefact questions prevalence-limited, realism ordinal); adjudication: label noise ≈ 1 %, disagreements dominated by rater slips
and one scale-anchor convention; 4 borderline clips shown. App bug (duplicate radio ids) fixed before the second pass; GT unaffected.
Files: `results/annotation/` (`agreement_reliability.*`, `adjudication_reliability.csv`, `reliability_adjudicated.jsonl`,
`adjudication_summary.json`, raw `annotator2{,_ext}/`); tools `src/annotation/{agreement,serve_adjudication,adjudication_to_jsonl}.py`.

## 6. Explanation faithfulness (E10, human-eval half) — done, full scale
**Full set (2026-09-29):** all 7371 v4o justifications graded by Sonnet 5 (rubric v2, 8-frame contact sheet per clip,
true FFT value given), 80 batches of 20 clips, one grader per batch. Files: `results/annotation/faithfulness_full/`
(`faithfulness_full_summary.md`, `faithfulness_full_results.json`, `graded_items.parquet`, `STATUS.md`); prompt
`src/annotation/faithfulness_full_grader_prompt.md`; code `build_faithfulness_full.py`, `analyze_faithfulness_full.py`.
CIs = cluster bootstrap over base scenarios.

| | hallucination (false claim ∪ missed event) |
|---|---|
| all 7371 answers | **18.4 % [16.7, 20.1]** |
| clean clips (6635) | **14.0 % [12.4, 15.8]** |
| filename-collision clips (736) | 57.2 % [51.0, 63.9] |
| answer correct / wrong | 13.5 % / 25.7 % |
| Reality / Visual / Traffic / Safety / Spatial-temporal / Artifacts | 11.9 / 13.5 / 20.6 / 23.0 / 25.7 / 29.3 % |
| drive-dreams / predict1 | 15.8 / 26.7 % |

Internally inconsistent answers 1.8 %. Claim types (keyword classification of grader notes, approximate): motion
claims (stopped / proceeded / overtook) 9.6 %, FFT misquote 4.8 % (47 % on collision clips, **0.06 % on clean clips** →
tool values are quoted exactly unless the resolver loaded the wrong file), static content (non-existent traffic light,
gibberish sign text "read as clean", melted cars denied) 3.9 %.
**Grader reliability:** calibration on the 100-item study reproduces the rate (15 % vs 18 % Sonnet / 23 % Opus earlier);
three batches regraded by fresh graders: rates 16.1 vs 16.8 % (n=298), item-level κ 0.39; item-level κ vs the earlier
graders 0.29–0.31, vs the author's 20 items 0.26. → the aggregate rate is stable, single-item verdicts are noisy
(motion-from-stills is the main disagreement); report rates, not per-item labels. Protocol deviations: some graders
compared weather variants of the same scene despite the instruction (minor, both directions).
**Paper wording:** "An LLM verifier (Claude Sonnet 5, validated against a second grader, a fresh-grader replicate and
20 author ratings) flags a visually false or omitted-but-decisive claim in 18 % of the agent's 7,371 justifications
(14 % on correctly resolved clips); tool outputs are quoted verbatim; errors concentrate in motion claims and in
behaviour questions (23–26 %) rather than the forensic reality question (12 %)."
Earlier 100-item study (Opus 5 + Sonnet 5): 18–23 %, κ 0.54 between graders — `results/annotation/faithfulness_summary.md`.
Counterfactual tool runs = cluster half (`src/agentic/counterfactual/`).

## 7. Frontier models (E4) — code only, deferred
`closed_source/claude/` (Batch API, structured output, caching), self-reflection round for all providers, model-id-parametrised
outputs; no API call made. Cost: Sonnet 5 ≈ $25 batch+cache / $105 worst case for 7371 items. Decision pending (budget, final dataset).

## 8. Open items
| item | owner |
|---|---|
| apply `results/structured_state/dropped_clips.txt` to the PVC GT JSON; exclude these clips in final tables (`run_stats.py`) | teammate / paper time |
| cluster: 736-item rerun → E6 v2 runs + agent sweep → E10 counterfactuals | teammate |
| E4 budget | you / supervisor |
| paper text: reproducibility appendix, presentation fixes, licences, offline-tool statement, prompt-injection note, "no VLM fine-tuning" wording | writing |
