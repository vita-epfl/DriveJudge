# Explanation faithfulness — full v4o set, Sonnet 5 grader (rubric v2)

Graded 7371 / 7371 items (81 batches). Invalid rows 0. CIs: 95 % cluster bootstrap over base scenarios.

### Overall

| group | n | hallucination % | false claim % | missed event % | inconsistent % |
|---|---|---|---|---|---|
| all | 7371 | 18.4 [16.7, 20.1] | 16.2 [14.6, 18.0] | 2.4 [1.8, 3.0] | 1.8 [1.4, 2.1] |

### By category

| group | n | hallucination % | false claim % | missed event % | inconsistent % |
|---|---|---|---|---|---|
| Reality understanding | 3172 | 11.9 [10.3, 13.8] | 10.2 [8.7, 11.7] | 1.9 [1.3, 2.6] | 1.3 [0.8, 1.7] |
| Safety understanding | 2164 | 23.0 [19.1, 26.9] | 18.8 [15.5, 22.3] | 4.5 [2.9, 6.5] | 1.1 [0.7, 1.6] |
| Artifacts understanding | 727 | 29.3 [25.8, 32.8] | 28.6 [25.0, 32.0] | 1.4 [0.5, 2.3] | 6.6 [4.7, 8.6] |
| Traffic laws understanding | 665 | 20.6 [15.6, 26.3] | 19.5 [14.3, 25.2] | 1.1 [0.3, 2.0] | 1.5 [0.6, 2.5] |
| Spatial-temporal understanding | 339 | 25.7 [18.8, 33.4] | 25.7 [18.5, 33.6] | 0.0 [0.0, 0.0] | 1.8 [0.6, 3.2] |
| Visual understanding | 304 | 13.5 [9.2, 18.1] | 13.5 [9.3, 18.2] | 0.0 [0.0, 0.0] | 1.0 [0.0, 2.2] |

### By generator

| group | n | hallucination % | false claim % | missed event % | inconsistent % |
|---|---|---|---|---|---|
| cosmos-drive-dreams | 5657 | 15.8 [13.7, 18.0] | 13.6 [11.7, 15.7] | 2.3 [1.6, 3.1] | 1.2 [0.9, 1.6] |
| cosmos-predict1 | 1714 | 26.7 [23.7, 29.7] | 24.8 [21.9, 27.6] | 2.6 [1.6, 3.9] | 3.6 [2.6, 4.5] |

### By answer correctness

| group | n | hallucination % | false claim % | missed event % | inconsistent % |
|---|---|---|---|---|---|
| answer correct | 4428 | 13.5 [11.5, 15.9] | 12.2 [10.3, 14.3] | 1.3 [0.6, 2.2] | 1.5 [1.1, 2.0] |
| answer wrong | 2943 | 25.7 [22.9, 28.8] | 22.3 [19.5, 25.0] | 4.0 [2.9, 5.1] | 2.1 [1.5, 2.8] |

### By provenance (basename-collision clips)

| group | n | hallucination % | false claim % | missed event % | inconsistent % |
|---|---|---|---|---|---|
| clean clip | 6635 | 14.0 [12.4, 15.8] | 12.1 [10.5, 13.8] | 2.2 [1.5, 2.9] | 1.8 [1.4, 2.2] |
| collision clip | 736 | 57.2 [51.0, 63.9] | 53.5 [47.4, 60.7] | 4.5 [2.4, 7.1] | 1.8 [1.0, 2.6] |

### By question

| group | n | hallucination % | false claim % | missed event % | inconsistent % |
|---|---|---|---|---|---|
| Is it a real video or generated ? | 1597 | 10.1 [8.5, 11.7] | 9.3 [7.9, 10.9] | 0.9 [0.4, 1.6] | 0.1 [0.0, 0.3] |
| Overall, how realistic is the scene (roads, markings, lighti | 1574 | 13.7 [11.6, 15.8] | 11.0 [9.4, 12.7] | 2.9 [2.0, 4.0] | 2.4 [1.5, 3.3] |
| Overall, how safe is the driving of the ego car ? Give a sco | 1082 | 23.4 [19.2, 28.0] | 19.5 [15.6, 23.5] | 4.3 [2.5, 6.4] | 1.1 [0.6, 1.8] |
| Is it a safe way of driving for the ego car ? | 1082 | 22.6 [18.6, 26.6] | 18.1 [14.7, 21.9] | 4.8 [2.9, 7.2] | 1.1 [0.5, 1.8] |
| Is the ego car following the traffic rules ? | 512 | 20.9 [15.4, 26.6] | 19.9 [14.7, 25.5] | 1.0 [0.2, 2.1] | 0.8 [0.2, 1.6] |
| Do some objects change shape or appearance ? | 402 | 29.6 [25.5, 34.0] | 28.9 [24.7, 33.0] | 2.0 [0.8, 3.4] | 6.0 [3.7, 8.5] |
| Does any object appear or disappear without continuous motio | 325 | 28.9 [24.3, 33.4] | 28.3 [23.8, 32.9] | 0.6 [0.0, 1.6] | 7.4 [4.7, 10.2] |
| Has the ego car stopped ? | 201 | 25.4 [15.3, 36.0] | 25.4 [15.3, 36.7] | 0.0 [0.0, 0.0] | 2.0 [0.4, 4.2] |
| Is the ego car on a highway ? | 159 | 3.1 [0.7, 5.7] | 3.1 [0.7, 6.0] | 0.0 [0.0, 0.0] | 1.9 [0.0, 4.2] |
| Is the ego car behavior consistent with visible cues of the  | 153 | 19.6 [11.8, 29.1] | 18.3 [10.7, 27.0] | 1.3 [0.0, 3.3] | 3.9 [0.7, 7.5] |
| Are the traffic lights red ? | 144 | 25.0 [17.9, 32.3] | 25.0 [17.9, 31.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| On which side is the ego car overtaking ? | 133 | 27.1 [17.6, 38.3] | 27.1 [17.5, 38.3] | 0.0 [0.0, 0.0] | 1.5 [0.0, 3.8] |
| Has the ego car stopped after the sign ? | 3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Is the lighting changing ? | 1 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Has the ego car stopped after the stop sign ? | 1 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Are the traffic signs consistent with ego movement? For exam | 1 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| Has the ego car stopped after the stop ? | 1 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

### Claim types (keyword classification of grader notes; approximate)

```
{
 "claim_type_counts": {
  "motion": 710,
  "fft_misquote": 352,
  "static_content": 291,
  "none": 6018
 },
 "claim_type_rates": {
  "fft_misquote": [
   0.047754714421381086,
   0.039242290715690124,
   0.05752993612725984
  ],
  "motion": [
   0.096323429656763,
   0.0806939547975936,
   0.11299866000523352
  ],
  "static_content": [
   0.03947903947903948,
   0.032361872228731166,
   0.04707066663791722
  ]
 },
 "claim_type_by_provenance": {
  "collision clip": {
   "fft_misquote": 47.28,
   "motion": 4.35,
   "static_content": 5.57
  },
  "clean clip": {
   "fft_misquote": 0.06,
   "motion": 10.22,
   "static_content": 3.77
  }
 }
}
```

### Replicate graders (same batch, fresh grader)

```
{
 "batch_005": {
  "n": 138,
  "agreement": 0.775,
  "kappa": 0.3783783783783787,
  "rate_main": 0.261,
  "rate_rep": 0.21
 },
 "batch_049": {
  "n": 80,
  "agreement": 0.863,
  "kappa": 0.2334494773519169,
  "rate_main": 0.025,
  "rate_rep": 0.163
 },
 "batch_053": {
  "n": 80,
  "agreement": 0.9,
  "kappa": 0.5000000000000002,
  "rate_main": 0.125,
  "rate_rep": 0.1
 }
}
```

### Grader stability vs the 100-item study

```
{
 "rater_sonnet_v2": {
  "n": 100,
  "agreement": 0.81,
  "kappa_hallucination": 0.311594202898551,
  "rate_new": 0.15,
  "rate_old": 0.18
 },
 "rater_opus_v2": {
  "n": 100,
  "agreement": 0.78,
  "kappa_hallucination": 0.2926045016077169,
  "rate_new": 0.15,
  "rate_old": 0.23
 }
}
```
