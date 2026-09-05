"""Engineering debug view (read-only) over telemetry frames.

Draws, top-down: road boundaries and the reference direction (explicitly NOT a
lane), ego footprint + safety margin + heading, obstacle footprints with
id/type, predicted trajectories with uncertainty, every candidate trajectory
(rejected in red, feasible in grey, selected in green), the controller target
point, and a text panel with speed, behaviour state, TTC/risk, planner latency
and the selected candidate's cost breakdown.

It consumes the same dictionaries the JSONL telemetry sink writes, so it can
replay a log file or render frames straight from an in-memory run. It never
touches the simulation.

    python -m visualization.debug_view logs/sudden_cattle_crossing.jsonl --time 6.5 --save out.png
    python -m visualization.debug_view logs/sudden_cattle_crossing.jsonl --animate --save run.gif
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

COLORS = {
    "CRUISE": "#2e7d32", "FOLLOW": "#1565c0", "CAUTION": "#f9a825",
    "AVOID": "#ef6c00", "EMERGENCY_BRAKE": "#c62828", "STOPPED": "#6a1b9a",
}


def _rect(cx: float, cy: float, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2, width / 2
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw], [hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def load_frames(path: Path | str) -> list[dict[str, Any]]:
    frames = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def planning_frames(frames: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in frames if f.get("planning_cycle") and f.get("plan") and "candidates" in f["plan"]]


def render_frame(ax, f: dict[str, Any], vehicle: dict[str, float], safety_margin: float,
                 window: tuple[float, float] = (-15.0, 55.0)) -> None:
    ax.clear()
    ego = f["ego"]
    road = f["road"]
    L, W, off = vehicle["length"], vehicle["width"], vehicle["footprint_center_offset"]

    # road
    for key, style in (("left_boundary", "k-"), ("right_boundary", "k-")):
        pts = np.asarray(road[key])
        ax.plot(pts[:, 0], pts[:, 1], style, lw=2)
    ref = np.asarray(road["reference"])
    ax.plot(ref[:, 0], ref[:, 1], color="0.6", ls=(0, (6, 6)), lw=0.8,
            label=f"reference direction (lane markings: {road.get('lane_markings', 'NONE')})")

    # candidates
    plan = f.get("plan") or {}
    sel_id = plan.get("selected_id")
    for c in plan.get("candidates", []):
        tr = c["trajectory"]
        if c["id"] == sel_id:
            continue
        if c["feasible"]:
            ax.plot(tr["x"], tr["y"], color="0.55", lw=0.7, alpha=0.8)
        else:
            ax.plot(tr["x"], tr["y"], color="#e53935", lw=0.6, alpha=0.45)
    sel = plan.get("selected")
    if sel:
        tr = sel["trajectory"]
        ax.plot(tr["x"], tr["y"], color="#2e7d32", lw=2.5, label=f"selected: {sel['id']}")
        # ego footprints along the selected trajectory (every ~1 s)
        for k in range(0, len(tr["x"]), max(1, len(tr["x"]) // 4)):
            fx = tr["x"][k] + off * math.cos(tr["yaw"][k]); fy = tr["y"][k] + off * math.sin(tr["yaw"][k])
            r = _rect(fx, fy, tr["yaw"][k], L, W)
            ax.plot(r[:, 0], r[:, 1], color="#2e7d32", lw=0.6, alpha=0.5)

    # predictions
    for p in f.get("predictions", []):
        ax.plot(p["x"], p["y"], ".", color="#8e24aa", ms=3, alpha=0.7)
        for k in range(0, len(p["x"]), max(1, len(p["x"]) // 4)):
            circ = _rect(p["x"][k], p["y"][k], 0.0, 2 * p["sigma"][k], 2 * p["sigma"][k])
            th = np.linspace(0, 2 * math.pi, 40)
            ax.plot(p["x"][k] + p["sigma"][k] * np.cos(th), p["y"][k] + p["sigma"][k] * np.sin(th),
                    color="#8e24aa", lw=0.5, alpha=0.35)

    # objects
    for o in f.get("objects", []):
        r = _rect(o["x"], o["y"], o["heading"], o["length"], o["width"])
        ax.fill(r[:, 0], r[:, 1], color="#fb8c00", alpha=0.85)
        ax.plot(r[:, 0], r[:, 1], color="#e65100", lw=1)
        ax.annotate(f"{o['id']} ({o['type']})", (o["x"], o["y"] + o["length"] / 2 + 0.3), ha="center", fontsize=8)
        ax.arrow(o["x"], o["y"], o["vx"], o["vy"], head_width=0.3, color="#e65100", length_includes_head=True)

    # ego
    fx = ego["x"] + off * math.cos(ego["yaw"]); fy = ego["y"] + off * math.sin(ego["yaw"])
    r = _rect(fx, fy, ego["yaw"], L, W)
    ax.fill(r[:, 0], r[:, 1], color="#1e88e5", alpha=0.9, label="ego footprint")
    m = _rect(fx, fy, ego["yaw"], L + 2 * safety_margin, W + 2 * safety_margin)
    ax.plot(m[:, 0], m[:, 1], color="#1e88e5", ls="--", lw=0.8, label=f"safety margin {safety_margin:.2f} m")
    ax.arrow(ego["x"], ego["y"], 3 * math.cos(ego["yaw"]), 3 * math.sin(ego["yaw"]),
             head_width=0.4, color="#0d47a1", length_includes_head=True)

    # controller target
    trk = f.get("tracker") or {}
    lat = trk.get("lateral") if trk else None
    if lat:
        ax.plot(lat["target_x"], lat["target_y"], "x", color="#d81b60", ms=9, mew=2, label="controller target")

    # text panel
    beh = f.get("behavior") or {}
    risk = f.get("risk") or {}
    saf = f.get("safety") or {}
    state = beh.get("state", "-")
    ttc = risk.get("min_ttc")
    ttc_s = f"{ttc:.2f} s" if ttc is not None else "inf"
    lines = [
        f"t = {f['timestamp']:.2f} s    state: {state}",
        f"speed {ego['longitudinal_velocity']:.2f} m/s   steer {math.degrees(ego['steering_angle']):+.1f} deg   "
        f"a {f['control']['acceleration'] - f['control']['brake']:+.2f} m/s^2",
        f"risk {risk.get('max_level', '-')} ({risk.get('max_score', 0.0):.2f})   route TTC {ttc_s}   "
        f"min pred dist {risk.get('min_predicted_distance') if risk.get('min_predicted_distance') is not None else float('nan'):.2f} m",
        f"planner {plan.get('latency_ms', 0.0):.1f} ms   candidates {len(plan.get('candidates', []))}  "
        f"feasible {plan.get('feasible_count', 0)}  rejected {plan.get('rejected_count', 0)}",
    ]
    if sel and sel.get("weighted_costs"):
        parts = ", ".join(f"{k} {v:.2f}" for k, v in sel["weighted_costs"].items())
        lines.append(f"selected cost J = {sel['total_cost']:.3f}  [{parts}]")
    if saf.get("override_active"):
        lines.append(f"SAFETY OVERRIDE: {saf.get('reason', '')}")
    if beh.get("reason"):
        lines.append("why: " + beh["reason"][:110])
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=8,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec=COLORS.get(state, "k"), alpha=0.9))

    ax.set_xlim(ego["x"] + window[0], ego["x"] + window[1])
    ax.set_ylim(-8, 13)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"SIH26037 Stage 1 engineering view - {state}", color=COLORS.get(state, "k"))
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.2)


def vehicle_dict() -> dict[str, float]:
    from autonomy.core.config import load_vehicle_parameters
    p = load_vehicle_parameters()
    return {"length": p.length, "width": p.width, "footprint_center_offset": p.footprint_center_offset}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="telemetry JSONL file")
    ap.add_argument("--time", type=float, default=None, help="render the planning frame nearest this time")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--save", default=None, help="output file (.png for a frame, .gif/.mp4 for animation)")
    ap.add_argument("--stride", type=int, default=2, help="planning frames per animation step")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from autonomy.core.config import AutonomyConfig
    margin = AutonomyConfig.load().planning.safety_margin_m
    veh = vehicle_dict()
    frames = planning_frames(load_frames(args.log))
    if not frames:
        raise SystemExit("no planning frames in log")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    if args.animate:
        seq = frames[::args.stride]
        anim = FuncAnimation(fig, lambda i: render_frame(ax, seq[i], veh, margin), frames=len(seq), interval=100)
        if args.save:
            anim.save(args.save, writer="pillow" if args.save.endswith(".gif") else None, fps=10)
            print("saved", args.save)
        else:
            plt.show()
        return 0
    t = args.time if args.time is not None else frames[len(frames) // 2]["timestamp"]
    f = min(frames, key=lambda fr: abs(fr["timestamp"] - t))
    render_frame(ax, f, veh, margin)
    if args.save:
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print("saved", args.save)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
