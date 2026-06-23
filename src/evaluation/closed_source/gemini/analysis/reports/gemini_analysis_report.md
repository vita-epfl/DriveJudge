# Why does bare Gemini 3 Flash beat the harness?

A synthesis of six per-category investigations into Gemini 3 Flash Preview's
behaviour on the DriveJudgeBench benchmark (n = 7 282 (video, question) pairs,
5 JPEG frames @ 1 fps per clip, no audio, no full video). Per-category
deep-dives are in `./reports/<category>.md`.

---

## Headline

| | Accuracy |
| --- | ---: |
| Bare Gemini 3 Flash *(this work)* | **64.0 %** |
| Qwen3-VL agent (full harness, ~1.3 s / question) | 60.1 % |
| Qwen3-VL agent + hints | 50.9 % |
| GPT-5.4-mini *(prior best non-agent)* | 39.1 % |
| InternVL3.5-30B *(best open-source baseline)* | 30.8 % |
| Cosmos-Reason / InternVL3.5-8B / LLaVA-OneVision | 23–26 % |
| Qwen3-VL / Qwen2.5-Omni | 10–22 % |

The surprising finding is not just that Gemini wins, but that it wins
**without** the FFT / SAM / RAFT / SSIM agentic toolchain that closed the gap
for the Qwen3-VL agent. The remaining 36 % of error sits almost entirely on
the subsets where 5-frame, 1-fps sampling discards motion information — the
exact territory where the agent's video tools earn their keep.

## Per-category breakdown

| Category | n | Acc | Notes |
| --- | ---: | ---: | --- |
| Visual understanding | 304 | **95.7 %** | Highway / red-light / lighting cues; near-saturated. |
| Artifacts understanding | 727 | **83.9 %** | Garbled text + morphing geometry. Pairwise frame consistency. |
| Spatial-temporal | 339 | **77.6 %** | 91 % on "stopped"; only 59 % on "overtaking side" — almost entirely a *labelling-convention* failure. |
| Safety understanding | 2 122 | **71.0 %** | 87 % on negatives, 9–19 % on aggressive-takeover. |
| Traffic laws | 665 | **59.4 %** | 82 % stops, 71 % red lights, **9 %** aggressive takeover. |
| Reality understanding | 3 125 | **51.1 %** | Easiest to break: false positives on cosmos-predict1 high-fidelity clips. |

## Why does Gemini get things right?

Five mechanisms recur across the chains-of-thought, in roughly decreasing order
of contribution.

### 1. Strong OCR + language prior on signage

The single most-cited correct-call cue across both Reality and Artifacts is
*almost-English* text on signs, license plates, and pavement markings:

> "morphing of the **'STOPP'** text on the road" — `cross_stop_012_Morning.mp4`
> "garbled, unreadable characters" on a street-lamp pole — `negative/3d656470…_Snowy.mp4`
> "the 'PARKING' sign in the background morphing into **'PANNI'**" — `car_brake_008_Rainy.mp4`

This requires both legible character extraction from low-res sampled frames
*and* a language model that flags the strings as not-real-words. Open-source
7B–8B VLMs struggle on either side of that pipeline. It is by far Gemini's
sharpest tool in the artifact-detection toolbox.

### 2. A learned generative-artifact taxonomy

Gemini consistently invokes a small, named vocabulary in its reasoning —
*morphing, temporal flickering, physically implausible lens flare, mushy
textures, structural integrity, pixel popping, sliding pedestrians*. This
maps directly onto the failure modes cosmos-drive-dreams and cosmos-predict1
actually produce, suggesting Gemini's training included generative-model
evaluation data. Open VLMs in this benchmark *describe* scenes; Gemini
*diagnoses* them.

### 3. Frame-pair consistency reasoning (not optical flow)

In ~35 % of correct artifact calls, Gemini explicitly localises to a frame
index ("between frames 4 and 5"). It treats the input as five labelled
images and asks "does object X have a plausible counterpart in frame i+1?":

> "the silver sedan directly in front of the black SUV in the first frame
> disappears instantly in the second frame" — `cross_stop_010_Snowy.mp4`

This is **object-permanence reasoning**, not motion estimation. It generalises
remarkably well to the "without continuous motion" question even though the
model never sees continuous motion.

### 4. Static cues do most of the safety / law work

Lane lines, stop signs, red lights, weather, and "is the road clear?" are
answerable from a single well-chosen frame. With 5 frames, Gemini almost
always finds at least one that resolves the question. Combined with a strong
language prior for *what makes driving unsafe* — its safety reasoning reads
like a transferred-from-text checklist:

> "Driving through a red light, especially in rainy conditions with reduced
> traction and visibility, constitutes a major safety violation."

Static-cue safety is exactly where Gemini overlaps with — and on per-question
latency, beats — the agentic pipeline, since the agent pays tool-call cost on
trivially-resolvable single-frame questions.

### 5. Per-clip class priors are exploitable

About 52 % of the safety dataset is "safe" and ~88 % of the artifact dataset
is "yes-artifact". Gemini's defaults align with these priors, giving it a
~50 % free baseline before any vision is used. The remaining 30 % comes from
correctly resolving the visible cues.

## Why does it fail when it fails?

Failures cluster into four distinct mechanisms.

### A. Five frames at 1 fps destroy *motion-encoded* violations

Aggressive-takeover (lane changes, tailgating, shoulder driving) disappears
under 1-fps sampling. On `aggressive_takeover/`, traffic-law accuracy is
**9 %** and safety drops to **5.7 %** on the 1–3 score variant. Chains-of-
thought become short and template-y ("maintains its lane … reasonable speed
… no observable violations") — diagnostic of pattern-matching on static
highway aesthetics. This is the exact failure mode an optical-flow / RAFT
tool is designed to fix.

### B. cosmos-predict1 high-fidelity clips defeat the per-frame artifact scan

Binary Real-vs-Generated accuracy on cosmos-predict1 is **36.3 %** vs
**65.8 %** on cosmos-drive-dreams. When 5 stills look photographic and
Gemini cannot find OCR garbage or morph cues, it produces confidently-
wrong "indistinguishable from real dashcam footage" verdicts dozens of
times verbatim. Temporal artifacts (flickers, slow drifts) — exactly what
agent FFT/SSIM tools sniff out — are wiped by sub-sampling.

### C. Score-task mode collapse to {1, 3}

On 1-2-3 realism / safety scoring, Gemini essentially never emits "2".
73 % of safety-score errors are the maximally-wrong "GT=1, VLM=3", and
realism scores show 277 + 100 confusions of GT=1 → VLM=2/3 with very
few middle-class outputs. The score variant trails the binary by 5–6
points across the board.

### D. A labelling-convention failure on overtaking side

This is not a perception failure. In 45 of 76 spatial-temporal errors,
Gemini's reasoning correctly localises the slower vehicle on the ego's
right, then labels the maneuver "overtaking on the left" — using a
*passenger-side reference frame* instead of the benchmark's *which-lane-
did-the-ego-use-to-pass* convention. One eval even names the conflict:

> "In standard driving terminology, the ego car is overtaking the vehicle
> in the right lane by being on its left side." — `agg_tk_002_Foggy.mp4`

A one-line prompt clarification would push spatial-temporal above 90 %
and widen Gemini's overall lead.

## What this implies for the agent design

The 4-point gap between bare Gemini (64.0 %) and the user's Qwen3-VL agent
harness (60.1 %) is small, and it inverts the ordering everyone
expected — but the per-category breakdown is more interesting than the
headline. The agent harness was designed to add *temporal* signal
(FFT for flicker, RAFT for motion, SAM/SSIM for object continuity) on top
of a weak base VLM. With Gemini as the base:

- **Reality + Artifacts (~3 850 questions, ~62 % of weight):** the agent's
  FFT/SSIM tools mostly target temporal artifacts that 1-fps sampling
  destroys. Adding them on top of Gemini would likely lift the predict1
  reality slice from 36 % → 60+ %, closing roughly half the remaining
  binary error.
- **Aggressive-takeover (~440 questions, traffic + safety):** RAFT optical
  flow + lane-line detection would attack exactly the failure mode here.
  This is *the* sub-task where a tool-using agent on Gemini should
  catastrophically beat bare Gemini.
- **Visual + Spatial-temporal-stopping (~500 questions, 90 %+ accuracy):**
  agentic tools add latency without changing answers. Gemini already wins
  these zero-shot.

The minimal agent that adds the most value to a Gemini base is therefore
**dense-frame sampling + optical-flow + per-frame realism FFT** — not the
full SAM / DINO / SSIM stack.

## Methodological caveats worth noting

1. **No "Real" clips** in the binary reality slice — every Real-vs-Generated
   item has GT = Generated, so Gemini's 701 binary errors are *all* false-
   positive "Real" calls. Adding genuine real dashcam footage would test
   whether Gemini's prior is calibrated or just biased.
2. **Class imbalance on traffic-law GT** (655 No vs 10 Yes) means
   precision-on-violation is well-measured but recall on compliance is
   essentially untested.
3. **Ground-truth quality** is occasionally suspect: a handful of
   `crossing_red_lights` clips where Gemini correctly describes a stationary
   ego at a red light are still labelled "violation" (GT = No) in the
   dataset. These cases are not large enough to move the headline number
   but inflate the safety-error count by ~5–10 cases.
4. **Score-task labelling**: 31 GT = 2 cases out of 1 553 makes the middle
   class nearly untestable. The model's "no 2 ever" failure may partly be
   a dataset-design artifact.

## TL;DR for the report

> Bare Gemini 3 Flash reaches 64.0 % on this benchmark by combining a
> strong OCR + language prior, a learned generative-artifact taxonomy, and
> frame-pair object-permanence reasoning. Its remaining 36 % of error is
> dominated by motion-encoded subsets (aggressive takeover, cosmos-predict1
> high-fidelity reality, score-task mode collapse) that are exactly the
> domain a temporal-tooling agent harness is built for. The natural
> follow-up — bolt the FFT/RAFT tools onto a Gemini base — is the highest-
> leverage next step the data points to.
