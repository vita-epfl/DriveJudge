# FFT tool: validation, container confound, and the lattice score (E7 / E8)

Data: 3000 real clips (CarCrash "Normal", BDD100K-sourced; mpeg4, 1280×720, 10 fps) and 2510 generated
(cosmos-drive-dreams 1930, cosmos-predict1 580). Balanced test set E6 = 1000 real / 1000 generated
(566 dd / 434 p1), 1416 scenario clusters. All CIs = 95 % bootstrap over scenarios (2000 resamples).
Scripts: `src/fft/{manifest,features,extract_features,perturb,splits,calibrate,lattice,roc_two_conditions}.py`.

## 1. The published threshold is defensible on the files as stored
Calibrated on 1589 clips never in the benchmark (197 generated from unlabeled folders + 1392 real), scenario-disjoint
from every test clip: Youden / balanced-accuracy threshold **0.0472 [0.0457, 0.0512]**, EER 0.0487 [0.0476, 0.0497].
The published 0.048 lies inside both intervals.

## 2. Separation is real but the original 36-clip set overstated it
| set | n | AUC (anisotropy, raw files) |
|---|---|---|
| original 18 + 18 calibration clips | 36 | 0.975 [0.891, 1.000] |
| balanced E6 test | 2000 | **0.881 [0.864, 0.897]** |
At t = 0.048 on E6: recall 0.81, **false-positive rate 0.20**, MCC 0.61.
Figure: `report/figures/fft_roc_raw_vs_n10_aniso_mean.png`.

## 3. Container confound, quantified (reviewer: codec / source fingerprint)
Re-encoding every clip to one container (n10 = libx264 crf 23, 1280×704, 10 fps, 50 frames) on the same 2000 clips:
AUC 0.881 → **0.784 [0.762, 0.806]**; FPR at 0.048 0.20 → 0.35 (real clips move up from median 0.0415 to 0.0452,
generated stay at 0.054). About a third of the raw separation was the mpeg4-vs-h264 container. The threshold is
container-specific.

## 4. Resampling fingerprint: normalization by resizing is not a fix
Resizing alone creates the signal the tool reads. 200 real clips downscaled to 768×432 and upscaled back reach
median anisotropy **0.059** (0.042 in their native form) with **96.5 % called "generated"**. Anisotropy responds to
resampling history, so any pipeline that rescales inputs adds a fingerprint of its own; a "normalised" condition is
not a neutral one, and the threshold has to be recalibrated for whatever resampling the inputs went through.

## 5. What the spectra actually show, and the lattice score
`dataset/fft_inspect/fft_panel_raw.png` (+ 4 spectrum videos): generated clips carry a regular grid of bright peaks at
(±W/4, ±H/4) and (i·W/8, j·H/8) — the VAE patch periodicity — real clips never do. Anisotropy measures the radial
"rays", which real footage with strong edges also produces (the false positives) and which some generated clips lack
(the false negatives), while the grid is present in both. `src/fft/lattice.py` scores the grid peaks
(background-subtracted log amplitude at the 4 primary sites; 4 frames per clip, ~1 s/clip).

| detector (E6, 2000 clips) | raw AUC | n10 AUC | recall at 10 % FPR (dd / p1), n10 |
|---|---|---|---|
| anisotropy (published) | 0.880 | 0.783 | 0.31 / 0.50 |
| lattice, 4 primary sites | 0.878 [0.854, 0.900] | **0.918 [0.901, 0.934]** | 0.99 / 0.56 |
| anisotropy + lattice (logistic, out-of-fold) | **0.977** | **0.971** | 0.99 / 0.82 |
| + spectral slope + HF ratio | 0.986 | 0.983 | 1.00 / 0.91 |
Lattice catches every drive-dreams clip; anisotropy carries predict1; the pair is nearly container-invariant.
Per-generator lattice medians (n10): dd 2.47, p1 0.91, real 0.57 — the W/8 grid is much weaker in predict1, which is
why the two metrics have to be reported together.
Figures: `report/figures/fft_spectra_panel.png` (paper panel: real vs dd vs p1 mean spectra, lattice sites circled, both numbers per clip), `fft_roc_n10_aniso_vs_lattice.png` (n10: anisotropy 0.784 / lattice 0.918 / combined 0.971 [0.963, 0.979]), `fft_roc_raw_vs_n10_lattice_primary.png` (lattice, raw vs n10).

**Does the threshold transfer to another generator?** Leave-one-generator-out (fit on {one generator + real}, test on
the other generator + held-out real clips): on raw files dd→p1 AUC **0.892 [0.873, 0.910]** and p1→dd **0.867
[0.852, 0.882]**, with balanced accuracy at 0.048 of 0.83 and 0.80; after normalization dd→p1 **0.825 [0.794, 0.856]**
and p1→dd **0.748 [0.714, 0.779]** (bal. acc. 0.76 / 0.69). The operating point is therefore not fitted to one
generator's fingerprint — it degrades on an unseen generator by about as much as normalization costs overall, not more.

## 6. Weather (within-scenario, raw generated clips)
Rainy raises anisotropy by +0.0066 within the same scene (AUC vs real 0.957, recall 0.985 at 0.048); Original 0.835 /
0.688; Snowy 0.829 / 0.735. Weather alone moves the metric by ~15 % of the threshold.

## 7. Other findings
- `detect_periodic_artifacts` in the tool returns 0 on every clip (dead feature); the lattice score replaces it.
- Provenance: for the 188 benchmark videos whose basename exists in another pool, the anisotropy value quoted by the
  v4o agent matches the *other pool's* clip for 88 videos (354 answers) and the correct clip for 68 → tools ran on the
  wrong video for ~half of the collisions (fix: `src/agentic/video_resolve.py`; rerun list of 736 items).
- Degradation grid (E8): see §8.

## Recommendation for the tool
Report anisotropy and lattice score together (two numbers to the VLM, or a single logistic score), calibrate on
clips in the benchmark's native container, state the container explicitly, and do not resize inputs.

## 8. Robustness to degradations (E8; 300 real + 300 generated, one clip per scenario)
Each condition is applied on top of the n10 encode, anisotropy re-extracted on 20 frames (`src/fft/run_grid.sh`,
`robustness.py`; figure `report/figures/fft_robustness_summary.png`, per-axis figures `fft_robustness_{crf,resolution,fps,codec,frames}.pdf`).

| condition | AUC [95 % CI] | note |
|---|---|---|
| original files (mixed containers) | 0.886 [0.857, 0.913] | reference |
| n10 common container | 0.795 [0.759, 0.831] | baseline for the rows below |
| CRF 28 / 45 | 0.759 / 0.781 | compression barely matters at 20 frames |
| 640×352 / 480×264 | **0.714 / 0.726** | downscaling hurts most (removes the high-frequency ring) |
| 5 fps / 24 fps | 0.789 / 0.773 | frame rate irrelevant |
| mpeg4 / HEVC | 0.755 / 0.803 | codec shifts real-clip medians (0.046 / 0.047) but not separability much |
| 5 / 10 / 60 frames analysed, offset 1 | 0.799 / 0.795 / 0.794 / 0.794 | frame sampling irrelevant: 5 frames are enough |

Reading: once the container is equalized, the metric is stable to compression, frame rate, codec and frame sampling
(all within the n10 CI) and degrades only with spatial downscaling. Not run: VP9, CRF 18/35, 960×528, swap controls
(scripts support them). The lattice score was not put through the grid.
