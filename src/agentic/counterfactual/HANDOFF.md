# Hand-off — E10 counterfactual harness + provenance re-run

Everything here is written and tested locally; nothing has been run on the
cluster.  You need a RunAI pod with 5 GPUs (4 for vLLM, 1 for the tools), the
`vita-scratch` PVC mounted, and the usual agent venv.  Exact commands:
**`commands.txt`** — this file is the why and the order.

---

## 1. The collision bug (read this first)

`main_multi_tools_v4.py` built a `basename -> Path` cache by `rglob`-ing
`final_dataset`, and the VLM hands the tools a **bare filename** — on the v4o run
all 3158 tool calls passed a basename, never a path.  190 mp4 basenames exist in
more than one pool of `final_dataset` (`car_brake_001.mp4` is in
`cosmos-drive-dreams`, `cosmos-predict1` *and* `cosmos-predict2`), and the cache
kept whichever copy `rglob` visited last, so for those clips RAFT / SAM / FFT
could be computed on a **different video than the question was about** — silently,
with the artefacts cached under the wrong clip's path.  188 benchmark clips
(736 QA items) are affected.  The fix (`src/agentic/video_resolve.py`) pins each
worker thread to the item's own video path before the agent runs: a bare
filename that matches the current item resolves to *that* file, an ambiguous
basename with no match logs a WARNING listing every candidate instead of picking
one silently, and absolute/relative identifiers behave exactly as before.
`src/agentic/counterfactual/test_resolver.py` covers all of it offline.

Consequence for you: the 736 items in `rerun_collisions.json` must be re-run with
the fixed agent (v4o **and** v4p configs) and the new answers merged into
`v4o_analyzed.parquet` / `v4p_analyzed.parquet` before those numbers are quoted
again.  That is section **A** of `commands.txt`.

---

## 2. What to copy where

| from (repo)                          | to (PVC)                                     |
| ------------------------------------ | -------------------------------------------- |
| `src/agentic/` (whole dir, incl. `video_resolve.py`, `counterfactual/`) | `$STORAGE/code/src/agentic/` |
| `src/agentic/counterfactual/subsets/` | comes along with the dir                      |
| `src/agentic/counterfactual/rerun_collisions.json` | comes along with the dir       |

`$STORAGE = /mnt/vita/vita-students/users/scharffe/VLM-eval`.

Nothing else changed.  `main_counterfactual.py` is a surgical-patch copy of the
fixed canonical agent (same convention as `src/real_vs_generated/scripts/`), so
it must be run **with `src/agentic/counterfactual/` as the working directory**
(`import cf_tools`, and `sys.path` reaches `../` for `utils.*`, `fft.*`,
`video_resolve`).

---

## 3. Run order

0. `make_cf_videos.py` — CPU only, ~5 min.  Pre-renders `raft_reverse/**`
   (hue-rotated flow), `raft_corrupt/static_gray_flow.mp4` and
   `sam_corrupt/object_{1..4}_clean.mp4` under `$ROOT/counterfactual`.
   Idempotent: re-running it skips what exists.
1. `vllm serve …` on GPUs 0-3 (identical to the v4o command), wait for `:8000`.
2. **Section A** — the 736-item re-run, v4o then v4p, with the *canonical*
   `main_multi_tools_v4.py`.  Do this first: it is the one that unblocks the
   paper numbers.
3. **Section B** — the counterfactual runs, with `main_counterfactual.py`:
   baseline ×2, then FFT `{none, corrupt, reverse}` and SAM `{none, corrupt}` on
   `cf_forensic.json`, then RAFT `{none, corrupt, reverse}` on `cf_motion.json`.
   Order inside B does not matter; each run resumes from its own output file if
   it is interrupted (same `FileLock` + resume logic as v4o).

Budget (measured v4o rate ≈ 20 s/item, 4 workers ⇒ ≈ 5 s wall/item):

| block                                    | items | wall |
| ---------------------------------------- | ----: | ---: |
| baseline ×2 (600 forensic + 300 motion)  | 1800  | 2.5 h |
| FFT {none, corrupt, reverse} × 600       | 1800  | 2.5 h |
| SAM {none, corrupt} × 600                | 1200  | 1.7 h |
| RAFT {none, corrupt, reverse} × 300      |  900  | 1.25 h |
| rerun_collisions v4o + v4p × 736         | 1472  | 2.0 h |
| **total**                                | **7172** | **≈ 10 h wall ≈ 50 GPU-h** |

Add up to **~9 h of serialised GPU-4 tool time** for section A on a cold cache
(188 clips × SAM ≈ 90 s + RAFT ≈ 60 s + FFT ≈ 20 s; they share `_gpu_lock`).  On
a warm cache it is ~30 min.  Section B runs on the warm v4o caches — the FFT
counterfactuals are arithmetic on the cached summaries and never touch the GPU,
and the RAFT ones were pre-rendered in step 0.

---

## 4. Expected output files

Under `$STORAGE/results/counterfactual/` (JSON, `indent=2`, agent result shape
plus `cf_modes` and `cf_tool_text`):

| file                          | records | approx size |
| ----------------------------- | ------: | ----------: |
| `baseline_run1.json`          |     900 | 1.5–3 MB |
| `baseline_run2.json`          |     900 | 1.5–3 MB |
| `fft-none.json`               |     600 | 0.5–1 MB |
| `fft-corrupt.json`            |     600 | 1–2 MB |
| `fft-reverse.json`            |     600 | 1–2 MB |
| `sam-none.json`               |     600 | 1–2 MB |
| `sam-corrupt.json`            |     600 | 1–2 MB |
| `raft-none.json`              |     300 | 0.4–1 MB |
| `raft-corrupt.json`           |     300 | 0.4–1 MB |
| `raft-reverse.json`           |     300 | 0.4–1 MB |
| `v4o_rerun_collisions.json`   |     736 | 1–3 MB |
| `v4p_rerun_collisions.json`   |     736 | 1–3 MB |

(The real files will be somewhat larger than the ranges above if the
`history` traces are long; `--no-save-history` shrinks them but
`src/analysis/answer_analysis_agent.py` needs `history`, so keep it on.)

Under `$ROOT/counterfactual/` (artefacts, regenerable — **do not** copy back):
`raft_reverse/**` (one mp4 per cached flow video, same size as the source),
`raft_corrupt/static_gray_flow.mp4`, `sam_corrupt/*.mp4`, `fft_reverse/**`,
`fft_corrupt/**`, `fft_real/**` (small JSONs), `sam_mismatch/**` (only if you
run the optional `--sam-mode mismatch` arm).

Sanity checks before sending anything back:

```bash
python - <<'EOF'
import json, collections, glob
for f in sorted(glob.glob("$OUT/*.json")):
    d = json.load(open(f))
    a = collections.Counter(r["answer"] for r in d)
    print(f"{f.split('/')[-1]:28s} n={len(d):4d}  ERROR={a['ERROR']:3d} "
          f"MISSING={a['MISSING']:3d}  modes={d[0].get('cf_modes')}")
EOF
```

A run with more than a handful of `ERROR` answers means vLLM died or the
`--cf-root` was not writable — check the pod log before re-running (it resumes).

---

## 5. Sending results back

Only the JSONs are needed — they are a few MB in total:

```bash
tar czf cf_results.tgz -C "$STORAGE/results" counterfactual
# then scp / upload cf_results.tgz
```

Unpack into `results/counterfactual/` of the repo and run, on a laptop:

```bash
.venv/bin/python src/agentic/counterfactual/analyze_cf.py \
    --runs-dir results/counterfactual --n-boot 2000 --seed 0
```

which writes `results/counterfactual/{cf_metrics.parquet,cf_summary.json,cf_table.tex}`
and `report/figures/fig_counterfactual_flips.{pdf,png}`.

For the provenance re-run, feed the two `*_rerun_collisions.json` files through
the usual agent pipeline (`postprocess_missing.py`, then
`src/analysis/answer_analysis_agent.py`) and merge the 736 rows into the
existing v4o / v4p parquets, replacing the old rows on `(video, question)`.

---

## 6. What the harness actually does (so you can sanity-check the numbers)

* `--fft-mode reverse` reads the **cached real** FFT summary, mirrors
  `anisotropy.mean` about the decision threshold (`2·0.048 − x`, clamped to
  `[0.025, 0.09]`), keeps every other number, and **regenerates** the whole
  interpretation text from the mirrored summary.  Verdict and numbers therefore
  stay mutually consistent — the model cannot tell it from a genuine
  opposite-verdict clip.  The cached real `interpretation` string is never
  returned in a non-real mode, and nothing is ever written into the canonical
  `final_dataset/fft/` cache.
* `--fft-mode corrupt` draws the anisotropy uniformly from `U(0.030, 0.082)`,
  seeded by `f"{cf_seed}:{stem}"`.  That band straddles the threshold but is not
  centred on it, so ≈ 65 % of corrupt draws read "GENERATED" — expect the
  corrupt arm's accuracy to move toward the generated-heavy prior, not toward
  chance.
* `--raft-mode reverse` shows the flow video hue-rotated by 180°, i.e. pink
  (ego moving forward) ↔ green (ego moving backward); magnitude, texture, frame
  count and compression are untouched.  Verified locally: median hue shift
  exactly 180.0°.
* `--sam-mode mismatch` shows the SAM crops of a *different* clip from the same
  pool, chosen deterministically from `random.Random(f"{cf_seed}:sam:{stem}")`.
* `--fft-threshold` is a real knob: it re-renders the tool description, the
  system prompt and the interpretation text, so a recalibrated threshold can be
  tested without touching the canonical tool.  At the default 0.048 the
  generated text is byte-identical to `fft/compute_fft.generate_interpretation`
  (checked over 500 random summaries in `test_cf_fft.py`).

Both subsets exclude the 736 basename-colliding items (their v4o baseline is
untrustworthy) and cap each `(question, scenario)` at 2 items, where a scenario
is the clip stem with the weather suffix and the `_7b`/`_14b` suffix stripped —
the same clustering `src/analysis/stats.py` uses, and the unit the bootstrap in
`analyze_cf.py` resamples.

---

## 7. Files

```
src/agentic/video_resolve.py             the provenance fix (stdlib only)
src/agentic/main_multi_tools_v4.py       patched: uses it + saves `history`
src/agentic/counterfactual/
  cf_tools.py                counterfactual text/path helpers (offline-testable)
  main_counterfactual.py     the harness (patched copy of the fixed v4 agent)
  make_cf_videos.py          pre-render the reverse/corrupt video artefacts
  build_rerun_collisions.py  -> rerun_collisions.json   (736 items, 188 clips)
  build_cf_subset.py         -> subsets/cf_{forensic,motion,subsets_gt}.json
  analyze_cf.py              -> cf_metrics.parquet / cf_summary.json /
                                cf_table.tex / fig_counterfactual_flips.pdf
  test_resolver.py           offline tests for the resolver
  test_cf_fft.py             offline tests for the FFT counterfactual text
  commands.txt               the pod commands
  HANDOFF.md                 this file
```
