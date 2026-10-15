"""Readers for the RDS-HQ structured state behind the cosmos-drive-dreams clips.

Why this exists
---------------
Every ``cosmos-drive-dreams`` clip in the benchmark was produced by
Cosmos-Transfer1 from an HD-map *control video* that was rendered along an ego
trajectory.  For the violation scenarios that trajectory was **edited by hand**
in the Cosmos-Drive-Dreams viser editor (``visualize_rds_hq.py``): the author
dragged the ego through a red light / past a stop line / onto an overtaking
path and exported the interpolated poses to ``novel_pose/``.  The HD-map
layers, the original poses and the edited poses together *are* the structured
simulator state of those clips, so the traffic-law labels can be validated
against them.

This module only reads that state.  All the analysis lives in
``check_scenarios.py``.

Coordinate frames (derived from the toolkit source, not guessed)
----------------------------------------------------------------
``rds_hq/pose/<clip_id>.tar`` and ``rds_hq/novel_pose/<folder>/<clip_id>.tar``
are webdataset tars whose members are named

    <clip_id>.<frame:06d>.pose.<camera_name>.npy      # 4x4 float64

*  The matrix is **camera-to-world**.  ``render_from_rds_hq.py`` builds the
   world-to-camera matrix as ``torch.inverse(pose_all_frames[frame_idx])``
   before projecting world points (``render_sample_lidar``), and
   ``visualize_rds_hq.py`` line 607 annotates it as
   ``# [4,4] camera to world. opencv convention``.
*  The camera axes follow the **OpenCV convention**: column 0 = right,
   column 1 = down, column 2 = forward.  ``visualize_rds_hq.py`` reads the ego
   heading as ``pose[:3, 2]`` and ``grounding_pose(..., convention='opencv')``
   treats ``pose[:3, 2]`` as forward, ``-pose[:3, 0]`` as left and
   ``-pose[:3, 1]`` as up.  Verified numerically: ``pose[:3, 2]`` has a dot
   product of 0.9995 with the direction of travel on the clips we checked.
*  The **world frame is clip-local** and metric (metres).  It is the "rig"
   frame of the clip: every HD-map label in ``3d_*`` carries the attribute
   ``coordinate_frame: "rig"`` and ``create_minimap_projection`` feeds those
   vertices straight through ``inv(pose)`` with no extra transform, so map
   vertices and pose translations live in **one and the same frame**.  ``z`` is
   the world up axis (``grounding_pose`` uses ``up = [0, 0, 1]`` and keeps
   ``pose[2, 3]`` as the camera height).
*  The translation ``pose[:3, 3]`` is the *optical centre of the camera*, not
   the rear axle: for ``camera_front_wide_120fov`` it sits ~1.3 m above the
   ground and ~1.8 m ahead of the rig origin.  Distances reported downstream
   are therefore camera-referenced; the offset is constant so speeds, lateral
   offsets and crossing detections are unaffected, only absolute
   distance-to-line is biased by ~1.8 m (the camera crosses the line before the
   bumper does... which is the conservative direction for "did it cross").

Sampling rate
-------------
``config/dataset_rds_hq.json`` sets ``INPUT_POSE_FPS = 30``: the poses are a
**30 Hz** sequence.  Clips hold 297-300 pose frames (~10 s).

Which frames became a generated clip
------------------------------------
``dataset_rds_hq.json`` -> ``NOT_POST_TRAINING`` sets
``TARGET_RENDER_FPS = 30``, ``TARGET_CHUNK_FRAME = 121`` and
``OVERLAP_FRAME = 0``.  ``prepare_input`` therefore keeps every pose frame
(``range(0, frame_num, 30 // 30)``) and ``prepare_output`` cuts

    for i in range(0, len(render_frame_ids), 121 - 0):
        if i + 121 > len(render_frame_ids): continue
        ...writes  f"{clip_id}_{cur_idx}.mp4"

so with 297-300 frames exactly two windows survive:

    chunk 0 -> pose frames   0 .. 120   (inclusive)
    chunk 1 -> pose frames 121 .. 241   (inclusive)

and frames 242+ are **dropped**.  Confirmed against the rendered control
videos, which are all 121 frames at 30 fps, 1280x704.

Layers
------
``rds_hq/3d_<layer>/<clip_id>.tar`` holds a single JSON with a ``labels`` list.
Each label carries ``labelData.shape3d`` with ``polyline3d.vertices`` (lanes,
lanelines, wait lines, road boundaries, poles), ``polygon3d`` /
``surface3d`` style vertices (crosswalks, road markings) or
``cuboid3d.vertices`` = 8 corners (traffic lights, traffic signs).  A few
labels are ``labelData.emptyLabel`` and carry no geometry.  Useful attributes:

* wait lines: ``type`` in {STOP, YIELD, UNKNOWN}, ``intersection_subtype`` in
  {ENTRY, EXIT, NOT_APPLICABLE}, ``is_implicit`` flag (an implicit wait line is
  inferred by the autolabeller, not a painted stop bar).
* traffic signs: ``sign_class`` text, e.g.
  ``TRAFFIC_SIGN_REGULATORY_R1_STOP``.
* traffic lights: only ``label_name = traffic_light`` -- **the labels carry no
  signal state**, so the bulb colour cannot be read out of the structured
  state.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants derived from the toolkit configuration
# ---------------------------------------------------------------------------

POSE_FPS = 30
"""INPUT_POSE_FPS in cosmos-drive-dreams-toolkits/config/dataset_rds_hq.json."""

CHUNK_FRAMES = 121
"""TARGET_CHUNK_FRAME (NOT_POST_TRAINING)."""

CHUNK_OVERLAP = 0
"""OVERLAP_FRAME (NOT_POST_TRAINING)."""

FRONT_CAMERA = "camera_front_wide_120fov"
"""The only camera in the single-view rds_hq config, and the one the editor
records keyframes for."""

DEFAULT_RDS_ROOT = Path(
    os.environ.get(
        "RDS_HQ_ROOT",
        "/home/matthieu/Documents/Projet de master/Cosmos-Drive-Dreams/rds_hq",
    )
)

DEFAULT_TOOLKIT_ROOT = Path(
    os.environ.get(
        "COSMOS_TOOLKIT_ROOT",
        "/home/matthieu/Documents/Projet de master/Cosmos-Drive-Dreams/"
        "cosmos-drive-dreams-toolkits",
    )
)

STOP_SIGN_CLASS = "TRAFFIC_SIGN_REGULATORY_R1_STOP"


def chunk_frame_range(chunk: int, n_frames: int) -> tuple[int, int] | None:
    """Pose-frame window ``[start, stop)`` that became rendered chunk ``chunk``.

    Returns ``None`` when the clip is too short for that chunk (mirrors the
    ``if i + TARGET_CHUNK_FRAME > len(render_frame_ids): continue`` guard in
    ``prepare_output``).
    """
    step = CHUNK_FRAMES - CHUNK_OVERLAP
    start = chunk * step
    stop = start + CHUNK_FRAMES
    if stop > n_frames:
        return None
    return start, stop


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------


def _tar_members(path: Path, suffix: str):
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            if member.name.endswith(suffix):
                yield member.name, tf.extractfile(member).read()


@lru_cache(maxsize=512)
def load_poses(
    clip_id: str,
    source: str = "pose",
    camera: str = FRONT_CAMERA,
    rds_root: Path | str = DEFAULT_RDS_ROOT,
) -> np.ndarray:
    """Read one pose tar into an ``(n_frames, 4, 4)`` camera-to-world array.

    Parameters
    ----------
    source
        ``"pose"`` for the original recorded trajectory, or a novel-pose folder
        such as ``"novel_pose/generated"`` / ``"novel_pose/to_be_generated"``
        for the hand-edited one.
    """
    path = Path(rds_root) / source / f"{clip_id}.tar"
    if not path.exists():
        raise FileNotFoundError(path)
    frames: dict[int, np.ndarray] = {}
    for name, blob in _tar_members(path, f".pose.{camera}.npy"):
        # <clip_id>.<frame:06d>.pose.<camera>.npy  ->  frame index
        frame_idx = int(name.split(".")[-4])
        frames[frame_idx] = np.load(io.BytesIO(blob))
    if not frames:
        raise ValueError(f"no poses for camera {camera!r} in {path}")
    return np.stack([frames[k] for k in sorted(frames)]).astype(np.float64)


def has_poses(clip_id: str, source: str, rds_root: Path | str = DEFAULT_RDS_ROOT) -> bool:
    return (Path(rds_root) / source / f"{clip_id}.tar").exists()


def positions(poses: np.ndarray) -> np.ndarray:
    """``(T, 3)`` world position of the camera optical centre."""
    return poses[:, :3, 3]


def forward(poses: np.ndarray) -> np.ndarray:
    """``(T, 3)`` world-frame forward axis (OpenCV camera +z)."""
    return poses[:, :3, 2]


def left(poses: np.ndarray) -> np.ndarray:
    """``(T, 3)`` world-frame left axis (OpenCV camera -x)."""
    return -poses[:, :3, 0]


def up(poses: np.ndarray) -> np.ndarray:
    """``(T, 3)`` world-frame up axis (OpenCV camera -y)."""
    return -poses[:, :3, 1]


def speed_profile(poses: np.ndarray, fps: int = POSE_FPS) -> np.ndarray:
    """Instantaneous speed in m/s, ``(T,)``, central differences at the edges.

    ``speed[t]`` is attached to frame ``t`` so it can be indexed alongside
    positions and distances.
    """
    p = positions(poses)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
    if len(d) == 0:
        return np.zeros(len(p))
    return np.concatenate([[d[0]], 0.5 * (d[:-1] + d[1:]), [d[-1]]])


def heading(poses: np.ndarray) -> np.ndarray:
    """Unwrapped yaw of the forward axis in the world xy-plane, radians."""
    f = forward(poses)
    return np.unwrap(np.arctan2(f[:, 1], f[:, 0]))


# ---------------------------------------------------------------------------
# Map layers
# ---------------------------------------------------------------------------


@dataclass
class MapElement:
    """One HD-map label expressed in the clip-local ``rig`` world frame."""

    layer: str
    vertices: np.ndarray  # (K, 3), polyline vertices or the 8 cuboid corners
    attrs: dict = field(default_factory=dict)
    kind: str = "polyline"  # 'polyline' | 'polygon' | 'cuboid3d'

    @property
    def center(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    @property
    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices[0], self.vertices[-1]


def _attrs_to_dict(attributes) -> dict:
    out = {}
    for a in attributes or []:
        name = a.get("name")
        if "enum" in a:
            out[name] = a["enum"]
        elif "text" in a:
            out[name] = a["text"]
        elif "flag" in a:
            out[name] = a["flag"]
        elif "numeric" in a:
            out[name] = a["numeric"]
    return out


@lru_cache(maxsize=512)
def load_layer(
    clip_id: str,
    layer: str,
    rds_root: Path | str = DEFAULT_RDS_ROOT,
) -> tuple[MapElement, ...]:
    """Read one ``3d_<layer>`` tar into ``MapElement``s (rig/world frame).

    ``layer`` is the bare name, e.g. ``"wait_lines"``, ``"traffic_lights"``,
    ``"traffic_signs"``, ``"lanes"``, ``"lanelines"``, ``"crosswalks"``.
    Returns an empty tuple when the clip has no tar for that layer (several
    thousand clips are missing individual layers).
    """
    path = Path(rds_root) / f"3d_{layer}" / f"{clip_id}.tar"
    if not path.exists():
        return ()
    payload = None
    for _name, blob in _tar_members(path, ".json"):
        payload = json.loads(blob)
        break
    if payload is None:
        return ()

    out: list[MapElement] = []
    for label in payload.get("labels", []):
        shape = label.get("labelData", {}).get("shape3d")
        if shape is None:  # emptyLabel
            continue
        attrs = _attrs_to_dict(shape.get("attributes"))
        for key, kind in (
            ("polyline3d", "polyline"),
            ("polygon3d", "polygon"),
            ("cuboid3d", "cuboid3d"),
            ("surface3d", "polygon"),
        ):
            geom = shape.get(key)
            if geom is None:
                continue
            verts = geom.get("vertices")
            if not verts:
                continue
            arr = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
            out.append(MapElement(layer=layer, vertices=arr, attrs=attrs, kind=kind))
            break
    return tuple(out)


def wait_lines(clip_id: str, rds_root: Path | str = DEFAULT_RDS_ROOT) -> tuple[MapElement, ...]:
    return load_layer(clip_id, "wait_lines", rds_root)


def traffic_lights(clip_id: str, rds_root: Path | str = DEFAULT_RDS_ROOT) -> tuple[MapElement, ...]:
    return load_layer(clip_id, "traffic_lights", rds_root)


def traffic_signs(clip_id: str, rds_root: Path | str = DEFAULT_RDS_ROOT) -> tuple[MapElement, ...]:
    return load_layer(clip_id, "traffic_signs", rds_root)


def stop_signs(clip_id: str, rds_root: Path | str = DEFAULT_RDS_ROOT) -> list[MapElement]:
    return [s for s in traffic_signs(clip_id, rds_root) if s.attrs.get("sign_class") == STOP_SIGN_CLASS]


# ---------------------------------------------------------------------------
# Geometry helpers shared by the analysis
# ---------------------------------------------------------------------------


def signed_distance_to_line(points_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance of ``points_xy`` to the infinite line a-b.

    Positive on the left-hand side of ``a -> b``.
    """
    d = b_xy - a_xy
    n = np.array([-d[1], d[0]])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.full(len(points_xy), np.nan)
    n = n / norm
    return (points_xy - a_xy) @ n


def segment_intersection(p1, p2, q1, q2):
    """Parameter ``t`` in [0, 1] along ``p1->p2`` where it crosses ``q1->q2``.

    Returns ``None`` when the two 2-D segments do not properly intersect.
    """
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return float(np.clip(t, 0.0, 1.0))
    return None


def signed_offset_to_path(query_xy: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
    """Signed lateral offset of each query point w.r.t. a reference polyline.

    For every query point we find the closest point on ``path_xy`` (segment
    wise) and return the offset along the reference path's **left** normal at
    that point: positive = the query point is to the left of the reference
    trajectory.  This is invariant to along-track shifts (a clip that merely
    drives faster along the same lane gets ~0 offset), which is what
    "overtaking" needs.
    """
    a = path_xy[:-1]
    b = path_xy[1:]
    ab = b - a
    denom = (ab * ab).sum(axis=1)
    denom[denom < 1e-12] = 1e-12

    out = np.empty(len(query_xy))
    for i, q in enumerate(query_xy):
        t = ((q - a) * ab).sum(axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d2 = ((q - proj) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        tangent = ab[j]
        nrm = np.linalg.norm(tangent)
        if nrm < 1e-9:
            out[i] = np.nan
            continue
        tangent = tangent / nrm
        normal_left = np.array([-tangent[1], tangent[0]])
        out[i] = float((q - proj[j]) @ normal_left)
    return out


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a boolean array."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# mapping.txt
# ---------------------------------------------------------------------------


def parse_mapping(
    toolkit_root: Path | str = DEFAULT_TOOLKIT_ROOT,
) -> dict[str, dict]:
    """Parse ``cosmos-drive-dreams-toolkits/mapping.txt``.

    Returns ``{author_scenario_name: {clip_id, section, note}}``.  ``section``
    is ``"generated"`` for the ``VIDEOS GENEREES`` block and ``"to_generate"``
    for the ``A GENERER`` block.  Lines such as
    ``agg_tk_019 + driving_on_wrong_lane_005 : <clip>`` register **both**
    scenario names against the same clip; the trailing ``(not edited)`` note is
    kept verbatim.
    """
    path = Path(toolkit_root) / "mapping.txt"
    out: dict[str, dict] = {}
    section = "generated"
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "GENERER" in line and "VIDEOS" not in line:
                section = "to_generate"
            elif "GENEREES" in line:
                section = "generated"
            continue
        if ":" not in line or line.startswith("http"):
            continue
        names, _, rhs = line.partition(":")
        rhs = rhs.strip()
        note = ""
        if "(" in rhs:
            clip_id, _, tail = rhs.partition("(")
            clip_id = clip_id.strip()
            note = "(" + tail.strip()
        else:
            clip_id = rhs.split()[0] if rhs else ""
        if not clip_id or "-" not in clip_id:
            continue
        for name in names.split("+"):
            name = name.strip()
            if name:
                out[name] = {"clip_id": clip_id, "section": section, "note": note}
    return out
