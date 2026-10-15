"""Figure: the structured state behind one red-light and one stop-sign clip.

Two rows (a red-light scenario and a stop-sign scenario), two panels each:

* left  -- top-down HD map (lanes, the controlled wait line, the traffic light /
  stop sign) with the recorded ego trajectory in grey and the hand-edited one
  coloured by speed; the crossing point is marked, and the extent of the
  121-frame window that became the generated video is marked on the path;
* right -- speed against distance-to-the-wait-line for both trajectories, with
  the 10 m approach band and the 0.5 m/s stop threshold.

Colours: Okabe-Ito categorical hues (vermillion / blue / reddish purple) plus a
neutral grey, each also carried by a distinct marker shape, and the perceptually
uniform, CVD-optimised ``cividis`` ramp for speed.

    .venv/bin/python src/analysis/structured_state/make_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_scenarios as C  # noqa: E402
import load_rds as R  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "report" / "figures"

GREY = "#7f7f7f"
LANE = "#c9c9c9"
VERMILLION = "#d55e00"   # wait line
BLUE = "#0072b2"         # traffic light
PURPLE = "#cc79a7"       # stop sign
INK = "#222222"

EXAMPLES = [
    ("cross_red_017", "(a)  Red light: the recorded ego stops at the stop bar, the edited one drives through"),
    ("cross_stop_013", "(b)  Stop sign: the recorded ego stops at the stop bar, the edited one rolls through"),
]


def _controlled_lines(clip_id):
    lines = R.wait_lines(clip_id)
    lights = R.traffic_lights(clip_id)
    stops = R.stop_signs(clip_id)
    out = []
    for e in lines:
        if len(e.vertices) < 2:
            continue
        mid = e.vertices.mean(axis=0)[:2]
        near_light = lights and min(np.linalg.norm(l.center[:2] - mid) for l in lights) <= C.NEAR_LINE_M
        near_stop = stops and min(np.linalg.norm(s.center[:2] - mid) for s in stops) <= C.NEAR_LINE_M
        if near_light or near_stop:
            out.append(e)
    return out, lights, stops


def _frame(origin, heading):
    """2-D rotation taking ``heading`` to +y, so the ego drives up the page."""
    t = heading / np.linalg.norm(heading)
    n = np.array([t[1], -t[0]])  # +x of the plot = to the ego's right

    def tf(pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))[:, :2] - origin
        return np.stack([pts @ n, pts @ t], axis=1)

    return tf


def draw(fig, ax_map, ax_speed, author_scenario: str, title: str):
    mapping = R.parse_mapping()
    clip_id = mapping[author_scenario]["clip_id"]
    source = "novel_pose/generated" if R.has_poses(clip_id, "novel_pose/generated") else "pose"
    edit = R.load_poses(clip_id, source)
    orig = R.load_poses(clip_id, "pose")
    n = min(len(edit), len(orig))
    edit, orig = edit[:n], orig[:n]

    xy_e = R.positions(edit)[:, :2]
    xy_o = R.positions(orig)[:, :2]
    v_e = R.speed_profile(edit)
    v_o = R.speed_profile(orig)

    controlled, lights, stops = _controlled_lines(clip_id)
    el, cross_frame, _ = C._pick_wait_line(xy_e, controlled, R.forward(edit)[0])
    assert el is not None and cross_frame is not None, author_scenario

    k = int(np.floor(cross_frame))
    frac = cross_frame - k
    cross_pt = xy_e[k] * (1 - frac) + xy_e[k + 1] * frac
    tf = _frame(cross_pt, xy_e[min(k + 1, n - 1)] - xy_e[k])

    # ---------------- map panel -----------------------------------------
    for layer, lw in (("lanes", 0.6), ("lanelines", 0.6), ("crosswalks", 0.6)):
        for e in R.load_layer(clip_id, layer):
            p = tf(e.vertices)
            ax_map.plot(p[:, 0], p[:, 1], color=LANE, lw=lw, zorder=1)

    for e in controlled:
        p = tf(e.vertices)
        ax_map.plot(p[:, 0], p[:, 1], color=VERMILLION, lw=2.6, solid_capstyle="butt", zorder=4)

    mid = el.vertices.mean(axis=0)[:2]
    for l in lights:
        if np.linalg.norm(l.center[:2] - mid) <= C.NEAR_LINE_M:
            p = tf(l.center[None, :])
            ax_map.plot(p[0, 0], p[0, 1], marker="s", ms=7, mfc=BLUE, mec="white", mew=0.8, zorder=6)
    for s in stops:
        if np.linalg.norm(s.center[:2] - mid) <= C.NEAR_LINE_M:
            p = tf(s.center[None, :])
            ax_map.plot(p[0, 0], p[0, 1], marker="^", ms=8, mfc=PURPLE, mec="white", mew=0.8, zorder=6)

    po = tf(xy_o)
    ax_map.plot(po[:, 0], po[:, 1], color=GREY, lw=1.8, ls=(0, (4, 2)), zorder=3)

    pe = tf(xy_e)
    seg = np.stack([pe[:-1], pe[1:]], axis=1)
    norm = plt.Normalize(0, max(v_e.max(), v_o.max()))
    lc = LineCollection(seg, cmap="cividis", norm=norm, linewidth=3.0, zorder=5, capstyle="round")
    lc.set_array(v_e[:-1])
    ax_map.add_collection(lc)

    # the rendered 121-frame window that contains the crossing
    chunk = 0 if cross_frame < R.CHUNK_FRAMES else 1
    rng = R.chunk_frame_range(chunk, n)
    if rng is not None:
        a, b = rng
        for idx, lbl in ((a, "window start"), (b - 1, "window end")):
            ax_map.plot(pe[idx, 0], pe[idx, 1], marker="o", ms=6.5, mfc="white", mec=INK, mew=1.5, zorder=7)
            ax_map.annotate(
                lbl, xy=(pe[idx, 0], pe[idx, 1]), xytext=(6, 0), textcoords="offset points",
                fontsize=7.2, color=INK, va="center",
            )

    ax_map.plot(0, 0, marker="X", ms=10, mfc="white", mec=INK, mew=1.6, zorder=8)
    ax_map.set_aspect("equal")
    span = 55
    ax_map.set_xlim(-span, span)
    ax_map.set_ylim(-span, span)
    ax_map.set_xlabel("lateral offset from the crossing point (m)")
    ax_map.set_ylabel("along-track (m)")
    ax_map.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax_map.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax_map.spines[sp].set_visible(False)

    # ---------------- speed panel ---------------------------------------
    d_e = C._distance_series(xy_e, el)
    d_o = C._distance_series(xy_o, el)
    keep_e = (d_e > -25) & (d_e < 90)
    keep_o = (d_o > -25) & (d_o < 90)
    ax_speed.axvspan(0, C.APPROACH_M, color=VERMILLION, alpha=0.10, lw=0, zorder=0)
    ax_speed.axvline(0, color=VERMILLION, lw=1.6, zorder=2)
    ax_speed.axhline(C.STOP_SPEED, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax_speed.plot(d_o[keep_o], v_o[keep_o], color=GREY, lw=1.8, ls=(0, (4, 2)), zorder=3)
    ax_speed.scatter(
        d_e[keep_e], v_e[keep_e], c=v_e[keep_e], cmap="cividis", norm=norm, s=11, lw=0, zorder=4
    )
    ax_speed.invert_xaxis()
    ax_speed.set_xlabel("distance to the wait line (m), ego travels right to left")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax_speed.spines[sp].set_visible(False)
    cb = fig.colorbar(lc, ax=ax_map, fraction=0.045, pad=0.02)
    cb.set_label("speed (m/s)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax_speed.annotate(
        f"crossed at t = {cross_frame / R.POSE_FPS:.1f} s, "
        f"min speed in the last {C.APPROACH_M:.0f} m = "
        f"{np.nanmin(v_e[(d_e > 0) & (d_e <= C.APPROACH_M)]):.1f} m/s",
        xy=(0.02, 0.96), xycoords="axes fraction", fontsize=8, color=INK, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=1.0),
    )
    return lc


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 8.5,
            "axes.edgecolor": "#999999",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), gridspec_kw={"width_ratios": [1.0, 1.05]})
    for row, (scenario, title) in enumerate(EXAMPLES):
        draw(fig, axes[row, 0], axes[row, 1], scenario, title)

    handles = [
        Line2D([], [], color=GREY, lw=1.8, ls=(0, (4, 2)), label="recorded ego trajectory"),
        Line2D([], [], color="#3b4d6b", lw=3.0, label="hand-edited trajectory (colour = speed)"),
        Line2D([], [], color=VERMILLION, lw=2.6, label="controlled wait line (stop bar)"),
        Line2D([], [], color="none", marker="s", ms=7, mfc=BLUE, mec="white", label="traffic light"),
        Line2D([], [], color="none", marker="^", ms=8, mfc=PURPLE, mec="white", label="stop sign"),
        Line2D([], [], color="none", marker="X", ms=9, mfc="white", mec=INK, label="crossing point"),
        Line2D([], [], color="none", marker="o", ms=6.5, mfc="white", mec=INK, mew=1.5,
               label="ends of the rendered 121-frame window"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.4,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_structured_state_example.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
