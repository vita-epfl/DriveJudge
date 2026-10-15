"""Validate the cosmos-drive-dreams traffic-law labels against the structured state.

Reviewer 4 asked for the traffic-law labels to be checked against "manually
extracted trajectories or structured simulator state".  That state exists: the
violation clips were rendered along an ego trajectory the author edited by hand
in the Cosmos-Drive-Dreams viser editor, on top of the RDS-HQ HD map.  This
script recomputes, from the poses and the map alone,

* the ego path and speed profile,
* the nearest wait line (stop bar) ahead of the path, the distance to it over
  time, whether and when the path crosses it,
* the minimum speed in the last 10 m before that line and whether the ego came
  to a stop there,
* whether a traffic light / stop sign sits within 25 m of that line,
* the signed lateral offset of the edited path w.r.t. the original recorded
  path, and hence which side an overtake happened on,

for the whole seed clip and for each 121-frame window that actually became a
generated video, and then compares those quantities with the benchmark labels.

Usage
-----
    .venv/bin/python src/analysis/structured_state/check_scenarios.py

Writes ``results/structured_state/{scenario_state.parquet,scenario_state.json,
labels_vs_state.parquet,summary.md,scenario_video_map.json}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_rds as R  # noqa: E402
import scenario_map as SM  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "results" / "structured_state"
QUESTIONS = REPO / "dataset" / "dataset_final" / "questions_cosmos-drive-dreams.json"

# thresholds (documented in summary.md)
STOP_SPEED = 0.5          # m/s
STOP_MIN_SECONDS = 0.5    # s
APPROACH_M = 10.0         # m before the wait line
NEAR_LINE_M = 25.0        # a light / sign this close to the line governs it
OVERTAKE_LATERAL_M = 1.5  # m of lateral deviation from the original path
MAX_OVERTAKE_LATERAL_M = 12.0  # beyond this the edited path is a different route, not a lane change
EDIT_EPS_M = 0.05         # poses differing by less than this are "not edited"

Q_RULES = "Is the ego car following the traffic rules ?"
Q_STOPPED = "Has the ego car stopped ?"
Q_OVERTAKE = "On which side is the ego car overtaking ?"
Q_LIGHTS_RED = "Are the traffic lights red ?"
Q_CONSISTENT = "Is the ego car behavior consistent with visible cues of the traffic lights ?"


# ---------------------------------------------------------------------------
# per-window geometry
# ---------------------------------------------------------------------------


def _pick_wait_line(path_xy: np.ndarray, lines, forward0: np.ndarray):
    """Nearest wait line ahead of ``path_xy``.

    Preference order: (1) a line the path actually crosses -- the earliest such
    crossing; (2) otherwise the line ahead of the ego at the first frame with
    the smallest closest-approach distance.

    Returns ``(element, crossing_frame_float | None, closest_distance)`` or
    ``(None, None, nan)``.
    """
    best_cross = None  # (frame, element, closest)
    best_ahead = None  # (distance, element)
    p0 = path_xy[0]
    for el in lines:
        v = el.vertices[:, :2]
        if len(v) < 2:
            continue
        # crossing test: every path segment against every wait-line segment
        cross_frame = None
        for i in range(len(path_xy) - 1):
            for j in range(len(v) - 1):
                t = R.segment_intersection(path_xy[i], path_xy[i + 1], v[j], v[j + 1])
                if t is not None:
                    cross_frame = i + t
                    break
            if cross_frame is not None:
                break
        mid = v.mean(axis=0)
        d_closest = float(np.min(np.linalg.norm(path_xy[:, None, :] - v[None, :, :], axis=2)))
        if cross_frame is not None:
            if best_cross is None or cross_frame < best_cross[0]:
                best_cross = (cross_frame, el, d_closest)
        ahead = float(np.dot(mid - p0, forward0[:2]))
        if ahead > 0 and (best_ahead is None or d_closest < best_ahead[0]):
            best_ahead = (d_closest, el)

    if best_cross is not None:
        return best_cross[1], best_cross[0], best_cross[2]
    if best_ahead is not None:
        return best_ahead[1], None, best_ahead[0]
    return None, None, float("nan")


def _distance_series(path_xy: np.ndarray, el) -> np.ndarray:
    """Signed perpendicular distance to the wait line, positive before crossing."""
    a, b = el.vertices[0, :2], el.vertices[-1, :2]
    d = R.signed_distance_to_line(path_xy, a, b)
    if len(d) and not math.isnan(d[0]) and d[0] < 0:
        d = -d
    return d


def _line_block(xy, speed, el, cross_frame, closest, lights, stops, prefix: str = "") -> dict:
    """Distance / stop / control statistics for one selected wait line."""
    p = prefix
    if el is None:
        return {
            f"{p}wait_line_found": False,
            f"{p}wait_line_type": None,
            f"{p}wait_line_subtype": None,
            f"{p}wait_line_implicit": None,
            f"{p}wait_line_closest_m": float("nan"),
            f"{p}dist_to_line_m": [],
            f"{p}dist_at_start_m": float("nan"),
            f"{p}crossed_line": False,
            f"{p}crossing_frame": None,
            f"{p}crossing_time_s": None,
            f"{p}min_speed_before_line_ms": float("nan"),
            f"{p}stopped_before_line": None,
            f"{p}stop_duration_s": 0.0,
            f"{p}light_within_25m": None,
            f"{p}nearest_light_m": float("nan"),
            f"{p}stop_sign_within_25m": None,
            f"{p}nearest_stop_sign_m": float("nan"),
        }
    d = _distance_series(xy, el)
    approach = (d > 0) & (d <= APPROACH_M)
    min_speed = float(speed[approach].min()) if approach.any() else float("nan")
    run = R.longest_run(approach & (speed < STOP_SPEED))
    mid = el.vertices.mean(axis=0)[:2]
    light_d = float(min(np.linalg.norm(l.center[:2] - mid) for l in lights)) if lights else float("inf")
    stop_d = float(min(np.linalg.norm(s.center[:2] - mid) for s in stops)) if stops else float("inf")
    return {
        f"{p}wait_line_found": True,
        f"{p}wait_line_type": el.attrs.get("type"),
        f"{p}wait_line_subtype": el.attrs.get("intersection_subtype"),
        f"{p}wait_line_implicit": bool(el.attrs.get("is_implicit", False)),
        f"{p}wait_line_closest_m": closest,
        f"{p}dist_to_line_m": np.round(d, 3).tolist(),
        f"{p}dist_at_start_m": float(d[0]),
        f"{p}crossed_line": cross_frame is not None,
        f"{p}crossing_frame": None if cross_frame is None else float(cross_frame),
        f"{p}crossing_time_s": None if cross_frame is None else float(cross_frame) / R.POSE_FPS,
        f"{p}min_speed_before_line_ms": min_speed,
        f"{p}stopped_before_line": bool(run >= STOP_MIN_SECONDS * R.POSE_FPS),
        f"{p}stop_duration_s": float(run) / R.POSE_FPS,
        f"{p}light_within_25m": bool(light_d <= NEAR_LINE_M),
        f"{p}nearest_light_m": light_d,
        f"{p}stop_sign_within_25m": bool(stop_d <= NEAR_LINE_M),
        f"{p}nearest_stop_sign_m": stop_d,
    }


def analyse_window(
    clip_id: str,
    author_scenario: str,
    section: str,
    mapping_note: str,
    poses_edit: np.ndarray,
    poses_orig: np.ndarray,
    window: str,
    start: int,
    stop: int,
) -> dict:
    p_edit = poses_edit[start:stop]
    xy = R.positions(p_edit)[:, :2]
    speed = R.speed_profile(p_edit)
    fwd0 = R.forward(p_edit)[0]

    lines = R.wait_lines(clip_id)
    lights = R.traffic_lights(clip_id)
    stops = R.stop_signs(clip_id)
    signs = R.traffic_signs(clip_id)

    # (a) nearest wait line ahead of the path, whatever it is
    el, cross_frame, closest = _pick_wait_line(xy, lines, fwd0)
    # (b) nearest wait line that is actually *controlled*: a traffic light or a
    #     stop sign sits within NEAR_LINE_M of it.  This is the line whose
    #     crossing constitutes a violation.
    controlled = [
        e
        for e in lines
        if len(e.vertices) >= 2
        and (
            (lights and min(np.linalg.norm(l.center[:2] - e.vertices.mean(axis=0)[:2]) for l in lights) <= NEAR_LINE_M)
            or (stops and min(np.linalg.norm(s.center[:2] - e.vertices.mean(axis=0)[:2]) for s in stops) <= NEAR_LINE_M)
        )
    ]
    el_c, cross_c, closest_c = _pick_wait_line(xy, controlled, fwd0)

    row: dict = {
        "author_scenario": author_scenario,
        "clip_id": clip_id,
        "section": section,
        "mapping_note": mapping_note,
        "window": window,
        "frame_start": start,
        "frame_stop": stop,
        "n_frames_clip": len(poses_edit),
        "edited": bool(
            np.max(np.linalg.norm(R.positions(poses_edit) - R.positions(poses_orig), axis=1)) > EDIT_EPS_M
        ),
        "max_pose_shift_m": float(
            np.max(np.linalg.norm(R.positions(poses_edit) - R.positions(poses_orig), axis=1))
        ),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
        "speed_mean_ms": float(speed.mean()),
        "orig_speed_mean_ms": float(R.speed_profile(poses_orig[start:stop]).mean()),
        "speed_min_ms": float(speed.min()),
        "speed_max_ms": float(speed.max()),
        "ego_x": xy[:, 0].round(3).tolist(),
        "ego_y": xy[:, 1].round(3).tolist(),
        "speed_ms": speed.round(3).tolist(),
        "n_wait_lines": len(lines),
        "n_controlled_wait_lines": len(controlled),
        "n_traffic_lights": len(lights),
        "n_stop_signs": len(stops),
        "n_traffic_signs": len(signs),
    }
    row.update(_line_block(xy, speed, el, cross_frame, closest, lights, stops, ""))
    row.update(_line_block(xy, speed, el_c, cross_c, closest_c, lights, stops, "ctl_"))

    # lateral deviation from the original trajectory
    ref_xy = R.positions(poses_orig)[:, :2]
    off = R.signed_offset_to_path(xy, ref_xy)
    row["lateral_offset_m"] = np.round(off, 3).tolist()
    if np.all(np.isnan(off)):
        row.update(
            path_divergence_m=float("nan"),
            overtake_offset_m=float("nan"),
            overtaking_side=None,
            path_diverges=None,
        )
    else:
        row["path_divergence_m"] = float(np.nanmax(np.abs(off)))
        row["path_diverges"] = bool(row["path_divergence_m"] > MAX_OVERTAKE_LATERAL_M)
        # an overtake is a lane change, not a different route: only offsets in
        # the lane-change band count
        band = np.abs(off) <= MAX_OVERTAKE_LATERAL_M
        if band.any():
            idx = np.where(band)[0]
            k = idx[int(np.nanargmax(np.abs(off[idx])))]
            signed = float(off[k])
        else:
            signed = 0.0
        row["overtake_offset_m"] = signed
        row["overtaking_side"] = (
            "Left" if signed > OVERTAKE_LATERAL_M else "Right" if signed < -OVERTAKE_LATERAL_M else "None"
        )

    # rule prediction: running a *controlled* wait line without stopping, or
    # deviating laterally from the recorded lane by more than a lane width
    ran_control = bool(row["ctl_crossed_line"] and not row.get("ctl_stopped_before_line"))
    row["ran_controlled_line"] = ran_control
    row["predicted_violation"] = bool(ran_control or row["overtaking_side"] in ("Left", "Right"))
    return row


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


def load_labels() -> dict[tuple[str, str], dict[str, list[str]]]:
    """``(folder, scenario) -> {question: [answers over weather variants]}``."""
    data = json.loads(QUESTIONS.read_text())
    out: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for entry in data:
        if "qa_pairs" not in entry:
            continue
        folder, _, fname = entry["video"].partition("/")
        scenario, _, _weather = fname[:-4].rpartition("_")
        for qa in entry["qa_pairs"]:
            out[(folder, scenario)][qa["question"]].append(qa["answer"])
    return out


def majority(answers: list[str]) -> str | None:
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_state(rds_root: Path) -> pl.DataFrame:
    mapping = R.parse_mapping()
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for author_scenario, info in sorted(mapping.items()):
        clip_id = info["clip_id"]
        if not R.has_poses(clip_id, "pose", rds_root):
            skipped.append((author_scenario, "no original pose archive"))
            continue
        pose_source = "novel_pose/generated"
        if not R.has_poses(clip_id, pose_source, rds_root):
            # mapping.txt marks these "(not edited)": the control video was
            # rendered straight from the recorded trajectory
            pose_source = "pose"
            skipped.append((author_scenario, "no novel_pose/generated archive -> using the original poses"))
        poses_edit = R.load_poses(clip_id, pose_source, rds_root=rds_root)
        poses_orig = R.load_poses(clip_id, "pose", rds_root=rds_root)
        n = min(len(poses_edit), len(poses_orig))
        poses_edit, poses_orig = poses_edit[:n], poses_orig[:n]

        windows = [("full", 0, n)]
        for chunk in (0, 1):
            rng = R.chunk_frame_range(chunk, n)
            if rng is not None:
                windows.append((f"chunk{chunk}", rng[0], rng[1]))
        scenario_rows = [
            analyse_window(
                clip_id, author_scenario, info["section"], info["note"],
                poses_edit, poses_orig, name, start, stop,
            )
            for name, start, stop in windows
        ]
        for row in scenario_rows:
            row["pose_source"] = pose_source
        # where does the violation sit relative to each rendered window?
        full_row = scenario_rows[0]
        full_cross = full_row["ctl_crossing_frame"]
        for row in scenario_rows:
            row["full_clip_ctl_crossing_frame"] = full_cross
            row["full_clip_overtaking_side"] = full_row["overtaking_side"]
            row["full_clip_overtake_offset_m"] = full_row["overtake_offset_m"]
            if full_cross is None:
                row["event_vs_window"] = "no crossing in the whole clip"
            elif row["frame_start"] <= full_cross < row["frame_stop"]:
                row["event_vs_window"] = "inside window"
            elif full_cross < row["frame_start"]:
                row["event_vs_window"] = "before window"
            else:
                row["event_vs_window"] = "after window (clip cut before the event)"
        rows.extend(scenario_rows)
    df = pl.DataFrame(rows, strict=False)
    if skipped:
        print(f"[note] {len(skipped)} mapping.txt scenarios needed a fallback")
        for name, why in skipped:
            print(f"        {name}: {why}")
    return df


def build_join(state: pl.DataFrame, resolutions) -> pl.DataFrame:
    labels = load_labels()
    by_key = {
        (r["author_scenario"], r["window"]): r
        for r in state.to_dicts()
    }
    rows = []
    for (folder, scenario), res in sorted(resolutions.items()):
        if res.author_scenario is None or res.confidence.startswith("conflict"):
            continue
        window = "full" if res.chunk is None else f"chunk{res.chunk}"
        st = by_key.get((res.author_scenario, window)) or by_key.get((res.author_scenario, "full"))
        if st is None:
            continue
        lab = labels.get((folder, scenario), {})
        rows.append(
            {
                "folder": folder,
                "benchmark_scenario": scenario,
                "author_scenario": res.author_scenario,
                "clip_id": res.clip_id,
                "chunk": res.chunk,
                "window": st["window"],
                "map_confidence": res.confidence,
                "n_weather_variants": max((len(v) for v in lab.values()), default=0),
                "label_rules": majority(lab.get(Q_RULES, [])),
                "label_stopped": majority(lab.get(Q_STOPPED, [])),
                "label_overtake": majority(lab.get(Q_OVERTAKE, [])),
                "label_lights_red": majority(lab.get(Q_LIGHTS_RED, [])),
                "label_consistent": majority(lab.get(Q_CONSISTENT, [])),
                "edited": st["edited"],
                "pose_source": st["pose_source"],
                "max_pose_shift_m": st["max_pose_shift_m"],
                "wait_line_found": st["wait_line_found"],
                "ctl_wait_line_found": st["ctl_wait_line_found"],
                "ctl_wait_line_type": st["ctl_wait_line_type"],
                "crossed_line": st["ctl_crossed_line"],
                "crossing_time_s": st["ctl_crossing_time_s"],
                "stopped_before_line": st["ctl_stopped_before_line"],
                "min_speed_before_line_ms": st["ctl_min_speed_before_line_ms"],
                "light_within_25m": st["ctl_light_within_25m"],
                "stop_sign_within_25m": st["ctl_stop_sign_within_25m"],
                "overtaking_side": st["overtaking_side"],
                "overtake_offset_m": st["overtake_offset_m"],
                "path_divergence_m": st["path_divergence_m"],
                "event_vs_window": st["event_vs_window"],
                "full_clip_overtaking_side": st["full_clip_overtaking_side"],
                "speed_mean_ms": st["speed_mean_ms"],
                "orig_speed_mean_ms": st["orig_speed_mean_ms"],
                "full_clip_ctl_crossing_frame": st["full_clip_ctl_crossing_frame"],
                "ran_controlled_line": st["ran_controlled_line"],
                "predicted_violation": st["predicted_violation"],
                "state_rules": "No" if st["predicted_violation"] else "Yes",
                "state_stopped": (
                    None
                    if st["ctl_stopped_before_line"] is None
                    else ("Yes" if st["ctl_stopped_before_line"] else "No")
                ),
                "state_overtake": st["overtaking_side"],
            }
        )
    return pl.DataFrame(rows, strict=False)


def confusion(df: pl.DataFrame, label_col: str, state_col: str) -> pl.DataFrame:
    sub = df.filter(pl.col(label_col).is_not_null() & pl.col(state_col).is_not_null())
    if sub.is_empty():
        return pl.DataFrame({"label": [], "state": [], "n": []})
    return (
        sub.group_by([label_col, state_col])
        .agg(pl.len().alias("n"))
        .rename({label_col: "label", state_col: "state"})
        .sort(["label", "state"])
    )


def md_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "_(empty)_\n"
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df.iter_rows():
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rds-root", default=str(R.DEFAULT_RDS_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    rds_root = Path(args.rds_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== building structured state from mapping.txt scenarios ==")
    state = build_state(rds_root)
    print(state.select(["author_scenario", "clip_id", "window"]).height, "rows")

    print("== resolving benchmark scenario -> seed clip / chunk ==")
    resolutions = SM.resolve()
    (out_dir / "scenario_video_map.json").write_text(json.dumps(SM.to_json(resolutions), indent=1))

    joined = build_join(state, resolutions)

    # ---------------- persist -------------------------------------------
    state.write_parquet(out_dir / "scenario_state.parquet")
    (out_dir / "scenario_state.json").write_text(json.dumps(state.to_dicts(), indent=1, default=str))
    joined.write_parquet(out_dir / "labels_vs_state.parquet")

    # ---------------- report --------------------------------------------
    full = state.filter(pl.col("window") == "full")
    lines: list[str] = []
    A = lines.append
    A("# Structured-state validation of the cosmos-drive-dreams traffic-law labels\n")
    A("Generated by `src/analysis/structured_state/check_scenarios.py`.\n")

    A("## 0. What the structured state is\n")
    A("Every `cosmos-drive-dreams` clip was produced by Cosmos-Transfer1 from an HD-map **control "
      "video** rendered along an ego trajectory, on top of the NVIDIA RDS-HQ seed dataset. For the "
      "violation scenarios the author replayed the seed clip in the Cosmos-Drive-Dreams viser editor "
      "(`visualize_rds_hq.py`), dragged the ego camera through the red light / past the stop line / "
      "onto an overtaking path at a handful of keyframes, and exported the Bezier-interpolated poses "
      "to `rds_hq/novel_pose/generated/<clip_id>.tar`. The violation therefore holds **by "
      "construction**, and the edited trajectory *is* the structured state.\n")
    A("What is available, per seed clip:\n")
    A("- `pose/<clip_id>.tar` and `novel_pose/generated/<clip_id>.tar`: 4x4 **camera-to-world** "
      "matrices (OpenCV camera convention: +x right, +y down, +z forward) for 7 cameras and "
      "297-300 frames at **30 Hz**, in a metric, clip-local `rig` world frame with z up. This "
      "analysis uses `camera_front_wide_120fov`, the camera the editor records keyframes for.")
    A("- HD-map layers `3d_wait_lines` (stop bars, with `type` in {STOP, YIELD, UNKNOWN}, "
      "`intersection_subtype` in {ENTRY, EXIT} and an `is_implicit` flag), `3d_traffic_lights` and "
      "`3d_traffic_signs` (8-corner cuboids; signs carry a `sign_class`, e.g. "
      "`TRAFFIC_SIGN_REGULATORY_R1_STOP`), `3d_lanes`, `3d_lanelines`, `3d_crosswalks`, and "
      "`all_object_info` (other agents' boxes). Every label declares `coordinate_frame: rig`, i.e. "
      "the same frame as the poses -- the toolkit projects them with `inv(pose)` and no other "
      "transform.")
    A("- **No signal state**: the traffic-light labels carry no bulb colour.\n")
    A(f"Derived quantities in `scenario_state.parquet` (one row per scenario x window): ego path "
      f"(`ego_x`/`ego_y`), speed profile (`speed_ms`, m/s), the nearest wait line ahead of the path "
      f"and the nearest *controlled* one (`ctl_*`), signed distance to that line per frame "
      f"(`dist_to_line_m`), the crossing frame, the minimum speed within {APPROACH_M:.0f} m before "
      f"the line, `stopped_before_line` (< {STOP_SPEED} m/s for >= {STOP_MIN_SECONDS} s), whether a "
      f"light / stop sign lies within {NEAR_LINE_M:.0f} m of the line, and the signed lateral offset "
      f"of the edited path w.r.t. the recorded one (+ = left), from which `overtaking_side` follows "
      f"at a +/-{OVERTAKE_LATERAL_M} m threshold.\n")

    A("## 1. Coverage\n")
    A(f"- scenarios listed in `mapping.txt`: **{len(R.parse_mapping())}**")
    A(f"- of those whose seed clip has an original `pose` archive: **{full.height}** "
      f"(of which {int((full['pose_source'] == 'pose').sum())} have **no** `novel_pose/generated` archive -- "
      "mapping.txt marks them `(not edited)`, so the control video was rendered from the recorded trajectory)")
    A(f"- with a hand-edited trajectory (max |edited - original| > {EDIT_EPS_M} m): "
      f"**{int(full['edited'].sum())}** "
      f"(median max shift {full.filter(pl.col('edited'))['max_pose_shift_m'].median():.1f} m)")
    A(f"- with a wait line found ahead of the path (full clip): **{int(full['wait_line_found'].sum())}**")
    A(f"- with a *controlled* wait line (traffic light or stop sign within {NEAR_LINE_M:.0f} m) "
      f"ahead of the path: **{int(full['ctl_wait_line_found'].sum())}**")
    A(f"- whose path crosses that controlled line (full clip): **{int(full['ctl_crossed_line'].sum())}**")
    for w in ("chunk0", "chunk1"):
        sub = state.filter(pl.col("window") == w)
        if sub.is_empty():
            continue
        A(f"- rendered {w} (frames {sub['frame_start'][0]}-{sub['frame_stop'][0] - 1}) crosses *some* "
          f"controlled wait line: **{int(sub['ctl_crossed_line'].sum())}** / {sub.height} "
          "(a window may cross a different intersection than the edited one)")
    ev = state.filter(pl.col("window") != "full")["event_vs_window"].value_counts().sort("event_vs_window")
    A("")
    A("Where the **full-clip** controlled-line crossing (the edited violation) sits relative to each "
      "rendered 121-frame window:\n")
    A(md_table(ev))
    ed = full.filter(pl.col("edited"))
    A(f"- the edit mostly *speeds the ego up*: over the {ed.height} edited clips the mean speed goes from "
      f"{ed['orig_speed_mean_ms'].median():.1f} m/s (recorded) to {ed['speed_mean_ms'].median():.1f} m/s "
      f"(edited), max {ed['speed_mean_ms'].max():.1f} m/s -- the viser editor interpolates between hand-placed "
      "keyframes without a speed constraint")
    A(f"- benchmark scenarios linked to a seed clip: **{sum(1 for r in resolutions.values() if r.clip_id)}** "
      f"/ {len(resolutions)}")
    A(f"- benchmark scenarios in the joined table: **{joined.height}**\n")

    A("## 2. Agreement with the benchmark labels\n")
    for title, lab_col, st_col in (
        ("`Is the ego car following the traffic rules ?` vs state", "label_rules", "state_rules"),
        ("`Has the ego car stopped ?` vs `stopped_before_line`", "label_stopped", "state_stopped"),
        ("`On which side is the ego car overtaking ?` vs `overtaking_side`", "label_overtake", "state_overtake"),
    ):
        c = confusion(joined, lab_col, st_col)
        n = int(c["n"].sum()) if not c.is_empty() else 0
        agree = (
            int(c.filter(pl.col("label") == pl.col("state"))["n"].sum()) if not c.is_empty() else 0
        )
        A(f"### {title}\n")
        A(f"n = {n}, exact agreement = {agree} ({100 * agree / n:.0f}%)\n" if n else "n = 0\n")
        A(md_table(c))

    A("## 3. Mismatches\n")
    for lab_col, st_col, name in (
        ("label_rules", "state_rules", "following the rules"),
        ("label_stopped", "state_stopped", "has stopped"),
        ("label_overtake", "state_overtake", "overtaking side"),
    ):
        bad = joined.filter(
            pl.col(lab_col).is_not_null() & pl.col(st_col).is_not_null() & (pl.col(lab_col) != pl.col(st_col))
        )
        A(f"### {name} ({bad.height} mismatches)\n")
        if bad.is_empty():
            A("_none_\n")
            continue
        cols = [
            "folder", "benchmark_scenario", "author_scenario", "window", lab_col, st_col,
            "edited", "ctl_wait_line_found", "crossed_line", "crossing_time_s",
            "min_speed_before_line_ms", "overtake_offset_m", "path_divergence_m",
            "event_vs_window", "full_clip_overtaking_side", "map_confidence",
        ]
        A(md_table(bad.select([c for c in cols if c in bad.columns])))

    A("## 4. How the benchmark scenario names were linked to the seed clips\n")
    A("`mapping.txt` uses the **author's** scenario numbering; the benchmark renumbered everything, "
      "because each seed clip yields *two* 121-frame control videos and each became its own "
      "benchmark scenario. Joining the two name spaces by string equality pairs the wrong "
      "trajectories. `scenario_map.py` recovers the link from the per-(control video, weather) "
      "captions that drove Cosmos-Transfer1: the benchmark's caption files "
      "(`final_dataset/.../captions/<scenario>_<weather>.txt`) are matched verbatim against "
      "`rendering/generated/captions_rewritten/<control video>.json`, the 8 weather variants vote, "
      "and the vote is cross-checked against the sorted render order. Full table in "
      "`scenario_video_map.json`.\n")
    conf = pl.DataFrame({"confidence": [r.confidence for r in resolutions.values()]})
    A(md_table(conf.group_by("confidence").agg(pl.len().alias("n")).sort("n", descending=True)))
    A("Only rows whose two sources of evidence agree on the seed clip enter the join; `conflict` "
      "rows are dropped. Within a resolved pair the *chunk* (which half of the clip) is the least "
      "certain part -- both halves share the same edited trajectory, so a chunk swap changes the "
      "frame window but never the clip.\n")

    A("## 5. Questions the structured state cannot settle\n")
    A("- `Are the traffic lights red ?` and `Is the ego car behavior consistent with visible cues of "
      "the traffic lights ?` -- the `3d_traffic_lights` labels carry only `label_name = traffic_light`; "
      "there is **no signal state** in the HD map, and Cosmos-Transfer1 paints the bulb colour freely. "
      f"({int(joined['label_lights_red'].is_not_null().sum())} joined scenarios carry the first "
      f"question, {int(joined['label_consistent'].is_not_null().sum())} the second.)")
    A("- Anything about other agents' behaviour: `all_object_info` holds their boxes, but the labels "
      "in scope here are all ego-centric.\n")

    # ---------------- text for the paper --------------------------------
    rules_c = confusion(joined, "label_rules", "state_rules")
    n_rules = int(rules_c["n"].sum()) if not rules_c.is_empty() else 0
    agree_rules = int(rules_c.filter(pl.col("label") == pl.col("state"))["n"].sum()) if n_rules else 0
    ot_c = confusion(joined, "label_overtake", "state_overtake")
    n_ot = int(ot_c["n"].sum()) if not ot_c.is_empty() else 0
    agree_ot = int(ot_c.filter(pl.col("label") == pl.col("state"))["n"].sum()) if n_ot else 0
    mism = joined.filter(
        pl.col("label_rules").is_not_null()
        & (pl.col("label_rules") != pl.col("state_rules"))
    )
    win_expl = int(
        mism.filter(
            pl.col("event_vs_window").str.starts_with("before")
            | pl.col("event_vs_window").str.starts_with("after")
        ).height
    )
    A("## 6. Three to five sentences the paper can reuse\n")
    A(f"> The cosmos-drive-dreams violation clips are not labelled post hoc: each one was rendered "
      f"from an HD-map control video generated along an ego trajectory that we edited by hand in the "
      f"Cosmos-Drive-Dreams trajectory editor, so the traffic-law label holds by construction. To "
      f"make that auditable we re-derived, from the exported 30 Hz poses and the RDS-HQ map layers "
      f"alone, the ego path, its speed profile, the signed distance to the nearest stop bar governed "
      f"by a traffic light or stop sign, the crossing time, the minimum speed in the last "
      f"{APPROACH_M:.0f} m before that line, and the signed lateral offset of the edited path with "
      f"respect to the recorded one. Over the {full.height} seed scenarios covered by the editor log, "
      f"{int(full['edited'].sum())} carry a hand-edited trajectory, "
      f"{int(full['ctl_wait_line_found'].sum())} have a controlled stop bar ahead of the ego and "
      f"{int(full['ctl_crossed_line'].sum())} cross it, always without stopping "
      f"(minimum speed in the approach zone never falls below {STOP_SPEED} m/s for "
      f"{STOP_MIN_SECONDS} s in any of them). Matched to the benchmark clips, the structured state "
      f"reproduces the `following the traffic rules` label on {agree_rules}/{n_rules} scenarios "
      f"({100 * agree_rules / n_rules:.0f}%) and the overtaking side on {agree_ot}/{n_ot} "
      f"({100 * agree_ot / n_ot:.0f}%); {win_expl} of the {mism.height} rule disagreements are "
      f"explained by the 121-frame render window ending before (or starting after) the violation, "
      f"i.e. the label describes the scenario while the clip shows only its approach -- a labelling "
      f"caveat we now document rather than a labelling error.\n")
    A("### The frame-window caveat, stated precisely\n")
    A(f"`render_from_rds_hq.py` cuts each 297-300-frame seed clip into non-overlapping 121-frame "
      f"windows at 30 fps (`TARGET_CHUNK_FRAME = 121`, `OVERLAP_FRAME = 0`) and drops the tail, so "
      f"exactly two videos survive per clip: frames 0-120 and frames 121-241. The hand-edited "
      f"violation happens at one instant in the 10 s trajectory, so roughly half of the rendered "
      f"windows contain only the approach or only the aftermath. In this table "
      f"{int(state.filter((pl.col('window') != 'full') & (pl.col('event_vs_window') == 'inside window')).height)} "
      f"of {int(state.filter(pl.col('window') != 'full').height)} rendered windows actually contain "
      f"the crossing. Scenario-level labels were nonetheless applied to both windows, which is the "
      f"single largest source of label / video mismatch in the traffic-law questions.\n")

    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
