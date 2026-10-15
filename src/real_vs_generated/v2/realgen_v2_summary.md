# real/generated v2 — balanced benchmark extension

Built by `src/real_vs_generated/build_realgen_v2.py` (seed 42, max 3 clips per `scenario_id`).

* **1000 real** clips (CarCrash `videos/Normal`), 50 of them carried over from the v1 study
* **1000 generated** clips (cosmos-drive-dreams 566, cosmos-predict1 434) — benchmark folders only, cosmos-predict2 excluded
* **1000** clips flagged `in_pilot` (first 1000 messages of every question file)
* 1416 distinct scenario clusters
* 120 clips whose basename is ambiguous under `final_dataset` (`basename_collides`); dd: 31 hard-excluded, p1: 0 hard-excluded

## label x source

| ground_truth | cosmos-drive-dreams | cosmos-predict1 | real | total |
|---|---|---|---|---|
| Generated | 566 | 434 | 0 | 1000 |
| Real | 0 | 0 | 1000 | 1000 |
| **total** | 566 | 434 | 1000 | 2000 |

## cosmos-drive-dreams: category x weather

| category | - | Foggy | Golden hour | Morning | Night | Original | Rainy | Snowy | Sunny | total |
|---|---|---|---|---|---|---|---|---|---|---|
| aggressive_takeover | 0 | 11 | 13 | 13 | 13 | 16 | 12 | 12 | 13 | 103 |
| aggressive_takeover|negative | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| crossing_red_lights | 0 | 12 | 11 | 8 | 10 | 14 | 11 | 8 | 14 | 88 |
| crossing_stop | 0 | 18 | 16 | 12 | 16 | 15 | 15 | 12 | 15 | 119 |
| negative | 1 | 31 | 31 | 31 | 31 | 34 | 30 | 35 | 30 | 254 |
| **total** | 1 | 73 | 72 | 64 | 70 | 79 | 68 | 67 | 72 | 566 |

## cosmos-predict1: gen_size x weather

| gen_size | - | Foggy | Golden hour | Morning | Night | Rainy | Snowy | Sunny | total |
|---|---|---|---|---|---|---|---|---|---|
| - | 67 | 29 | 29 | 29 | 27 | 30 | 31 | 29 | 271 |
| 14b | 83 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 83 |
| 7b | 80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 80 |
| **total** | 230 | 29 | 29 | 29 | 27 | 30 | 31 | 29 | 434 |

## pilot membership

| source | False | True | total |
|---|---|---|---|
| cosmos-drive-dreams | 283 | 283 | 566 |
| cosmos-predict1 | 217 | 217 | 434 |
| real | 500 | 500 | 1000 |
| **total** | 1000 | 1000 | 2000 |

## sampling diagnostics

```json
{
  "dd": {
    "generator": "dd",
    "target": 566,
    "picked": 566,
    "pool": 1100,
    "pool_after_collision_filter": 1069,
    "collisions_hard_excluded": 31,
    "collisions_in_sample": 0,
    "capacity_under_cap": 582,
    "n_scenarios": 214,
    "cap": 3
  },
  "p1": {
    "generator": "p1",
    "target": 500,
    "picked": 434,
    "pool": 494,
    "pool_after_collision_filter": 494,
    "collisions_hard_excluded": 0,
    "collisions_in_sample": 120,
    "capacity_under_cap": 434,
    "n_scenarios": 202,
    "cap": 3
  }
}
```

## files

* `questions_realgen_v2.json` — 2000 entries
* `questions_realgen_v2_pilot.json` — 1000 entries
* `questions_realgen_v2_hints.json` — 2000 entries
* `questions_realgen_v2_pilot_hints.json` — 1000 entries
* `questions_realgen_v2_agent.json` — 2000 entries
* `questions_realgen_v2_pilot_agent.json` — 1000 entries
* `questions_realgen_v2_agent_hints.json` — 2000 entries
* `questions_realgen_v2_pilot_agent_hints.json` — 1000 entries
* `questions_realgen_v2_raw.json` — 2000 entries
* `questions_realgen_v2_pilot_raw.json` — 1000 entries
* `groundtruth_realgen_v2.json` — 2000 entries
* `realgen_v2_real_clips.txt` — 1000 entries
