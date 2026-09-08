"""Record real telemetry from the frozen stack into replay files the UI can scrub.

    scenario -> Simulation (unmodified) -> TelemetryPublisher -> CaptureSink -> ui/replay/*.json

Every frame written here was produced by an actual run of the frozen autonomy stack. Nothing is
synthesised, interpolated or smoothed. The UI replays exactly what the simulator published.

Usage:
    python -m ui.capture                      # the 5 demo scenarios, sensors mode
    python -m ui.capture --all                # all 11 scenarios, both perception modes
    python -m ui.capture --scenario DENSE_MARKET_MIXED_TRAFFIC --mode sensors
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autonomy.core.config import load_vehicle_parameters                    # noqa: E402
from autonomy.telemetry.dashboard import compact_frame, _json_safe          # noqa: E402
from autonomy.telemetry.telemetry import TelemetryFrame, TelemetrySink      # noqa: E402
from scripts.run_scenario import run_with_sinks                             # noqa: E402
from ui.server import DEMO_ORDER, scenario_names                            # noqa: E402

REPLAY_DIR = ROOT / "ui" / "replay"

# One line per scenario, describing what it demonstrates. Descriptive text only; every NUMBER in
# the UI comes from the recorded frames or from docs/FINAL_SYSTEM_RESULTS.json.
BLURB = {
    "UNMARKED_VILLAGE_ROAD": "No lane markings at all. The planner works purely from the drivable corridor.",
    "DENSE_MARKET_MIXED_TRAFFIC": "The stress case: many mixed-class agents in a confined space.",
    "SUDDEN_CATTLE_CROSSING": "Cattle steps into the carriageway 30 m ahead, creating a real future collision.",
    "SUDDEN_PEDESTRIAN_DART": "A pedestrian darts in at short range. This is what emergency braking exists for.",
    "NARROW_LANE_REVERSE_RECOVERY": "Boxed in with no forward path. The ego reverses along a curve to escape.",
    "NARROW_LANE_BOXED_IN": "The same boxed-in problem with different geometry.",
    "NARROW_LANE_MUTUAL_YIELD": "Oncoming vehicle in a lane too narrow for both. Neither has priority.",
    "UNSIGNALIZED_INTERSECTION": "Crossing traffic with no signal and no right-of-way rule.",
    "HIGHWAY_MERGE_SLOW_VEHICLES": "Informal merging at 20 m/s with slower vehicles ahead.",
    "UNPROTECTED_TURN": "Turning across traffic with no protection.",
    "MIXED_TRAFFIC_CURVE": "Curved corridor — proves corridor-frame planning where s,d are not x,y.",
}


def _round(o, nd: int = 3):
    """Trim float precision before serialising.

    Raw telemetry floats serialise as `38.499999999999996` — 18 characters to express a value we
    draw at centimetre resolution. Rounding to 3 dp is a millimetre, far below anything the scene
    or the panels can show, and it cuts the candidate-fan payload by roughly 4x. No value the UI
    displays changes: every panel formats to 1-3 dp anyway.
    """
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, list):
        return [_round(x, nd) for x in o]
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    return o


class CaptureSink(TelemetrySink):
    """Stores every compact frame. Planning cycles carry the full plan; between them the frame is
    still recorded so vehicle motion stays smooth at 50 Hz."""

    def __init__(self, stride: int = 2) -> None:
        self.params = load_vehicle_parameters()
        self.frames: list[dict] = []
        self.stride = stride
        self._n = 0

    def write(self, frame: TelemetryFrame) -> None:
        self._n += 1
        # Keep every planning cycle (they carry the candidate fan) plus every `stride`-th
        # in-between frame, so the car moves smoothly without quadrupling the file size.
        if not frame.planning_cycle and (self._n % self.stride):
            return
        self.frames.append(_round(_json_safe(compact_frame(frame, self.params, candidate_stride=2))))

    def close(self) -> None:
        pass


def capture(scenario: str, mode: str, stride: int = 2) -> dict:
    sink = CaptureSink(stride)
    t0 = time.perf_counter()
    result = run_with_sinks(scenario, log_dir=None, console=False, perception=mode,
                            extra_sinks=[sink])
    wall = time.perf_counter() - t0
    m = result.metrics.to_dict()
    doc = {
        "scenario": result.scenario_name,
        "mode": mode,
        "description": BLURB.get(result.scenario_name, ""),
        "captured_wall_s": round(wall, 2),
        "frame_count": len(sink.frames),
        "duration_s": sink.frames[-1]["t"] if sink.frames else 0.0,
        "vehicle": sink.frames[0]["vehicle"] if sink.frames else None,
        "road": sink.frames[0]["road"] if sink.frames else None,
        "summary": {
            "completed": bool(m.get("scenario_completed")),
            "termination": m.get("termination_reason"),
            "collisions": m.get("collision_count"),
            "min_clearance_m": m.get("minimum_obstacle_clearance"),
            "replans": m.get("replanning_count"),
            "plan_changes": m.get("plan_change_count"),
            "emergency_brakes": m.get("emergency_brake_activations"),
            "reverse_manoeuvres": m.get("reverse_manoeuvres"),
            "reverse_distance_m": m.get("reverse_distance_m"),
            "latency_mean_ms": m.get("planning_latency_mean_ms"),
            "latency_max_ms": m.get("planning_latency_max_ms"),
            "path_length_m": m.get("path_length"),
            "average_speed": m.get("average_speed"),
            "behavior_state_durations": m.get("behavior_state_durations"),
            "detections_total": m.get("detections_total"),
            "track_count_mean": m.get("track_count_mean"),
            "perception_recall": m.get("perception_recall"),
            "perception_precision": m.get("perception_precision"),
        },
        "frames": sink.frames,
    }
    return doc


def write_doc(doc: dict) -> Path:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{doc['scenario'].lower()}__{doc['mode']}.json.gz"
    path = REPLAY_DIR / name
    blob = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6) as fh:
        fh.write(blob)
    return path


def write_index() -> Path:
    """Manifest of everything captured, so the UI never has to guess what exists."""
    items = []
    for p in sorted(REPLAY_DIR.glob("*.json.gz")):
        with gzip.open(p, "rb") as fh:
            d = json.loads(fh.read())
        items.append({
            "file": p.name,
            "scenario": d["scenario"],
            "mode": d["mode"],
            "description": d["description"],
            "frame_count": d["frame_count"],
            "duration_s": d["duration_s"],
            "summary": d["summary"],
            "demo_rank": DEMO_ORDER.index(d["scenario"]) if d["scenario"] in DEMO_ORDER else 99,
        })
    items.sort(key=lambda x: (x["demo_rank"], x["scenario"], x["mode"]))
    idx = REPLAY_DIR / "index.json"
    idx.write_text(json.dumps({"replays": items}, indent=1), encoding="utf-8")
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Record real telemetry into UI replay files")
    ap.add_argument("--scenario", help="one scenario id (default: the 5 demo scenarios)")
    ap.add_argument("--mode", default="sensors", choices=["sensors", "ground_truth"])
    ap.add_argument("--all", action="store_true", help="all 11 scenarios in both perception modes")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth non-planning frame")
    args = ap.parse_args()

    if args.all:
        jobs = [(s, m) for s in scenario_names() for m in ("sensors", "ground_truth")]
    elif args.scenario:
        jobs = [(args.scenario, args.mode)]
    else:
        jobs = [(s, args.mode) for s in DEMO_ORDER]

    print(f"capturing {len(jobs)} run(s) -> {REPLAY_DIR}")
    for i, (sc, md) in enumerate(jobs, 1):
        print(f"  [{i}/{len(jobs)}] {sc} · {md} ... ", end="", flush=True)
        try:
            doc = capture(sc, md, args.stride)
        except Exception as exc:
            print(f"FAILED {type(exc).__name__}: {exc}")
            continue
        path = write_doc(doc)
        kb = path.stat().st_size // 1024
        s = doc["summary"]
        print(f"{doc['frame_count']} frames · {doc['duration_s']:.1f}s sim · {kb} KB · "
              f"{s['termination']} · {s['collisions']} collisions")
    idx = write_index()
    n = len(json.loads(idx.read_text())["replays"])
    print(f"index: {idx}  ({n} replays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
