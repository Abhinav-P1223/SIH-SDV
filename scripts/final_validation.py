"""Phase 8: final closed-loop validation of the frozen autonomy stack.

    python scripts/final_validation.py

Runs every scenario in both perception modes, records every metric the SIH brief asks for, and
captures replanning evidence for the interaction scenarios. Nothing is trained, tuned or modified;
this script only measures.

WHERE THE LEARNED DETECTOR IS, AND IS NOT
    Phases 6 and 7 built a real camera object detector and proved it reaches the tracker through the
    `Detection` interface. It does NOT appear in these closed-loop runs, and cannot: the simulator
    has no image renderer, so there are no pixels for a detector to consume. What "sensors mode"
    exercises is the simulated camera, LiDAR and radar models feeding the same `Detection` contract
    into the same tracker and fusion. The detector's evidence is offline and stays offline. Saying
    otherwise would be the fake integration this project has refused at every phase.

NO GROUND-TRUTH LEAKAGE
    In sensors mode the autonomy stack's object provider IS the fusion tracker, and the tracker
    asserts it never reads a detection's `truth_id`. `verify_no_leakage` checks the wiring at
    runtime rather than trusting the comment.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.core.config import (AutonomyConfig, ObjectProfiles, PerceptionConfig,  # noqa: E402
                                  load_vehicle_parameters)
from simulation.runner import Simulation                                 # noqa: E402
from simulation.scenarios.loader import load_scenario                    # noqa: E402
from autonomy.telemetry.telemetry import TelemetryPublisher              # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "phase8_final_validation.json"

SCENARIOS = [
    # the five the SIH brief names explicitly
    ("UNMARKED_VILLAGE_ROAD", "SIH: unmarked village road"),
    ("UNSIGNALIZED_INTERSECTION", "SIH: busy urban intersection, no signals"),
    ("HIGHWAY_MERGE_SLOW_VEHICLES", "SIH: highway merge with slow vehicles"),
    ("DENSE_MARKET_MIXED_TRAFFIC", "SIH: dense market mixed traffic"),
    ("SUDDEN_CATTLE_CROSSING", "SIH: sudden cattle crossing"),
    # the stronger ones the repository already carries
    ("NARROW_LANE_REVERSE_RECOVERY", "reverse recovery"),
    ("NARROW_LANE_BOXED_IN", "boxed in, reverse available"),
    ("UNPROTECTED_TURN", "unprotected interaction"),
    ("NARROW_LANE_MUTUAL_YIELD", "narrow-lane mutual yield"),
    ("SUDDEN_PEDESTRIAN_DART", "emergency pedestrian event"),
    ("MIXED_TRAFFIC_CURVE", "mixed traffic on a curve"),
]


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def verify_no_leakage(sim: Simulation, mode: str) -> dict:
    """Check the WIRING, at runtime, rather than trusting a comment."""
    out = {"mode": mode}
    if mode == "sensors":
        out["object_provider_is_fusion"] = sim.object_provider is sim.fusion
        out["fusion_present"] = sim.fusion is not None
        out["sensors_present"] = sim.sensors is not None
    else:
        out["object_provider_is_world_truth"] = sim.object_provider is sim.world.object_provider
    out["planner_road_is_scenario_road"] = sim.planner.road is sim.world.road
    return out


def run_one(name: str, mode: str, cfg, params, profiles) -> dict:
    sc = load_scenario(name, profiles)
    perc = PerceptionConfig.load().with_mode(mode)
    sim = Simulation(sc, cfg, params, profiles, TelemetryPublisher(), perception=perc)
    wiring = verify_no_leakage(sim, mode)

    lat, steer, speeds, curv, cross = [], [], [], [], []
    steer_rate, ttc_phys = [], []
    fusion_ms = []
    t0 = time.perf_counter()
    prev_steer, prev_t = 0.0, 0.0
    while sim.running:
        before = sim.plan
        sim.step()
        f_t = sim.time
        if sim.plan is not before and sim.plan is not None:
            lat.append(float(getattr(sim.plan, "latency_ms", 0.0) or 0.0))
        st = float(sim.ego.steering_angle)
        if f_t > prev_t:
            steer_rate.append(abs(st - prev_steer) / (f_t - prev_t))
        prev_steer, prev_t = st, f_t
        steer.append(abs(st))
        speeds.append(float(sim.ego.longitudinal_velocity))
        if sim.plan is not None and sim.plan.selected.trajectory.curvature is not None:
            curv.append(float(np.abs(sim.plan.selected.trajectory.curvature).max()))
        if sim.tracker.last_debug is not None:
            cross.append(abs(sim.tracker.last_debug.lateral.cross_track_error))
        if math.isfinite(sim.risk.min_ttc_current_speed):
            ttc_phys.append(float(sim.risk.min_ttc_current_speed))
        if sim.fusion is not None:
            fusion_ms.append(sim.fusion.mean_latency_s() * 1e3)
    wall = time.perf_counter() - t0
    m = sim.metrics.finalize(sim.termination_reason == "GOAL_REACHED", sim.time,
                             sim.termination_reason)

    return {
        "scenario": name, "mode": mode,
        "completed": bool(m.scenario_completed), "termination": sim.termination_reason,
        "collisions": int(m.collision_count),
        "min_clearance_m": None if not math.isfinite(m.minimum_obstacle_clearance)
                           else round(m.minimum_obstacle_clearance, 3),
        "min_ttc_route_s": None if not math.isfinite(m.minimum_ttc) else round(m.minimum_ttc, 2),
        "min_ttc_physical_s": None if not math.isfinite(m.minimum_ttc_experienced)
                              else round(m.minimum_ttc_experienced, 2),
        "emergency_brakes": int(m.emergency_brake_activations),
        "replans": int(m.replanning_count), "plan_changes": int(m.plan_change_count),
        "replan_latency_ms": {
            "mean": round(float(statistics.mean(lat)), 2) if lat else None,
            "p50": round(pct(lat, 50), 2) if lat else None,
            "p95": round(pct(lat, 95), 2) if lat else None,
            "max": round(max(lat), 2) if lat else None},
        "path_length_m": round(float(m.path_length), 2),
        "path_smoothness": None if m.path_smoothness is None else round(float(m.path_smoothness), 5),
        "mean_abs_curvature": round(float(m.mean_abs_curvature), 5),
        "max_curvature": round(float(m.max_curvature), 4),
        "max_steering_rad": round(max(steer), 4) if steer else None,
        "max_steering_rate_rad_s": round(float(m.max_steering_rate), 4),
        "observed_steering_rate_max": round(max(steer_rate), 4) if steer_rate else None,
        "speed_mps": {"mean": round(float(np.mean(speeds)), 2),
                      "max": round(float(np.max(speeds)), 2),
                      "min": round(float(np.min(speeds)), 2)},
        "max_acceleration": round(float(m.max_acceleration), 2),
        "max_deceleration": round(float(m.max_deceleration), 2),
        "max_cross_track_m": round(max(cross), 3) if cross else None,
        "reverse_manoeuvres": int(m.reverse_manoeuvres),
        "reverse_distance_m": round(float(m.reverse_distance_m), 2),
        "behavior_transitions": int(m.behavior_transitions),
        "behavior_states": {k: round(v, 2) for k, v in (m.behavior_state_durations or {}).items()},
        "duration_s": round(float(sim.time), 2), "wall_seconds": round(wall, 1),
        "realtime_factor": round(float(sim.time) / max(wall, 1e-9), 2),
        # perception, sensors mode only
        "perception": {
            "recall": None if m.perception_recall is None else round(m.perception_recall, 3),
            "precision": None if m.perception_precision is None else round(m.perception_precision, 3),
            "missed_objects": int(m.perception_missed_objects),
            "false_tracks": int(m.perception_false_tracks),
            "track_position_error_m": None if m.tracking_position_error_m is None
                                      else round(m.tracking_position_error_m, 3),
            "track_velocity_error_mps": None if m.tracking_velocity_error_mps is None
                                        else round(m.tracking_velocity_error_mps, 3),
            "track_count_mean": None if m.track_count_mean is None else round(m.track_count_mean, 2),
            "prediction_error_m": None if m.prediction_error_m is None
                                  else round(m.prediction_error_m, 3),
            "detections_total": int(m.detections_total),
            "fusion_latency_ms": round(float(np.mean(fusion_ms)), 1) if fusion_ms else None,
        },
        "wiring": wiring,
    }


def replanning_evidence(name: str, cfg, params, profiles, mode: str = "sensors") -> dict | None:
    """Before / decision / after around the sharpest risk moment of a scenario.

    The moment is chosen as the frame with the lowest physical time-to-collision, which is where
    the stack is actually under pressure, rather than a frame picked to look good.
    """
    from simulation.runner import run_scenario
    r = run_scenario(name, log_dir=None, console=False,
                     perception=PerceptionConfig.load().with_mode(mode))
    frames = [f for f in r.frames if f.plan is not None and f.decision is not None]
    if not frames:
        return None
    risky = [f for f in frames if math.isfinite(f.risk.min_ttc_current_speed)]
    if not risky:
        return None
    k = min(range(len(risky)), key=lambda i: risky[i].risk.min_ttc_current_speed)
    f = risky[k]
    idx = frames.index(f)
    # Scan FORWARD for the first frame whose selected trajectory differs, rather than sampling one
    # arbitrary frame: a replan that lands 1.2 s later is still a replan, and fixing the offset
    # would have reported "no change" for a stack that plainly changed course.
    changed_at, after = None, frames[min(idx + 12, len(frames) - 1)]
    for g in frames[idx + 1:]:
        if g.timestamp - f.timestamp > 3.0:
            break
        if g.plan.selected.id != f.plan.selected.id:
            changed_at, after = round(g.timestamp - f.timestamp, 2), g
            break
    obj = min(f.objects, key=lambda o: math.hypot(o.x - f.ego.x, o.y - f.ego.y)) if f.objects else None
    pred = next((p for p in f.predictions if obj and p.object_id == obj.id), None)
    return {
        "scenario": name, "mode": mode, "t": round(f.timestamp, 2),
        "before": {
            "ego": {"x": round(f.ego.x, 2), "y": round(f.ego.y, 2),
                    "speed": round(f.ego.longitudinal_velocity, 2),
                    "yaw": round(f.ego.yaw, 3)},
            "nearest_object": None if obj is None else {
                "id": obj.id, "type": obj.object_type.value,
                "x": round(obj.x, 2), "y": round(obj.y, 2),
                "vx": round(obj.vx, 2), "vy": round(obj.vy, 2),
                "distance_m": round(math.hypot(obj.x - f.ego.x, obj.y - f.ego.y), 2)},
            "predicted_position_2s": None if pred is None else {
                "x": round(float(np.interp(f.timestamp + 2.0, pred.times, pred.x)), 2),
                "y": round(float(np.interp(f.timestamp + 2.0, pred.times, pred.y)), 2)},
            "min_ttc_physical_s": round(f.risk.min_ttc_current_speed, 2),
            "risk_level": f.risk.max_level.name, "risk_score": round(f.risk.max_score, 3),
            "candidates_total": len(f.plan.candidates),
            "candidates_feasible": int(f.plan.feasible_count),
            "candidates_rejected": len(f.plan.candidates) - int(f.plan.feasible_count),
        },
        "decision": {
            "fsm_state": f.decision.state.name, "reason": f.decision.reason,
            "selected_trajectory": f.plan.selected.id,
            "target_speed": round(float(f.plan.selected.target_speed), 2),
            "lateral_offset_end": round(float(f.plan.selected.lateral_offset_end), 2),
            "safety_override": bool(f.safety.override_active),
            "safety_reason": f.safety.reason,
        },
        "after": {
            "t": round(after.timestamp, 2),
            "fsm_state": after.decision.state.name,
            "selected_trajectory": after.plan.selected.id,
            "trajectory_changed": after.plan.selected.id != f.plan.selected.id,
            "trajectory_changed_after_s": changed_at,
            "ego": {"x": round(after.ego.x, 2), "y": round(after.ego.y, 2),
                    "speed": round(after.ego.longitudinal_velocity, 2)},
            "lateral_shift_m": round(after.ego.y - f.ego.y, 2),
            "speed_change_mps": round(after.ego.longitudinal_velocity - f.ego.longitudinal_velocity, 2),
        },
        "outcome": {"collisions": int(r.metrics.collision_count),
                    "min_clearance_m": None if not math.isfinite(r.metrics.minimum_obstacle_clearance)
                                       else round(r.metrics.minimum_obstacle_clearance, 3),
                    "termination": r.metrics.termination_reason},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["ground_truth", "sensors"])
    ap.add_argument("--evidence", nargs="*",
                    default=["SUDDEN_CATTLE_CROSSING", "SUDDEN_PEDESTRIAN_DART",
                             "DENSE_MARKET_MIXED_TRAFFIC", "NARROW_LANE_REVERSE_RECOVERY"])
    a = ap.parse_args()

    cfg, params, profiles = AutonomyConfig.load(), load_vehicle_parameters(), ObjectProfiles.load()
    results = []
    print("=" * 96)
    print("PHASE 8 FINAL VALIDATION - frozen stack, measurement only")
    print("=" * 96, flush=True)

    for mode in a.modes:
        print(f"\n### perception mode: {mode}")
        hdr = (f"{'scenario':<30}{'done':>5}{'coll':>5}{'clr':>7}{'ttc':>7}{'brk':>4}"
               f"{'replan':>7}{'lat p95':>9}{'path':>8}{'maxK':>7}{'rev':>5}  term")
        print(hdr); print("-" * len(hdr))
        for name, _ in SCENARIOS:
            r = run_one(name, mode, cfg, params, profiles)
            results.append(r)
            ttc = r["min_ttc_physical_s"]
            print(f"{name:<30}{int(r['completed']):>5}{r['collisions']:>5}"
                  f"{(r['min_clearance_m'] if r['min_clearance_m'] is not None else 0):>7.2f}"
                  f"{(ttc if ttc is not None else 99):>7.2f}{r['emergency_brakes']:>4}"
                  f"{r['replans']:>7}{(r['replan_latency_ms']['p95'] or 0):>9.1f}"
                  f"{r['path_length_m']:>8.1f}{r['max_curvature']:>7.3f}"
                  f"{r['reverse_manoeuvres']:>5}  {r['termination']}", flush=True)

    print("\n" + "=" * 96)
    print("REPLANNING EVIDENCE (sensors mode, at each scenario's lowest physical TTC)")
    print("=" * 96, flush=True)
    evidence = []
    for name in a.evidence:
        e = replanning_evidence(name, cfg, params, profiles, "sensors")
        if e is None:
            print(f"{name}: no risk frame captured"); continue
        evidence.append(e)
        b, d, af = e["before"], e["decision"], e["after"]
        print(f"\n{name}  at t={e['t']}s")
        print(f"  BEFORE   ego ({b['ego']['x']}, {b['ego']['y']}) at {b['ego']['speed']} m/s; "
              f"TTC {b['min_ttc_physical_s']}s; risk {b['risk_level']}")
        if b["nearest_object"]:
            o = b["nearest_object"]
            print(f"           nearest {o['type']} {o['id']} at {o['distance_m']} m, "
                  f"v=({o['vx']}, {o['vy']})")
        print(f"           candidates {b['candidates_feasible']} feasible / "
              f"{b['candidates_rejected']} rejected of {b['candidates_total']}")
        print(f"  DECISION {d['fsm_state']}: {d['reason'][:70]}")
        print(f"           selected {d['selected_trajectory']}, override={d['safety_override']}")
        print(f"  AFTER    {af['fsm_state']}, trajectory changed={af['trajectory_changed']}, "
              f"lateral {af['lateral_shift_m']} m, speed {af['speed_change_mps']} m/s")
        print(f"  OUTCOME  collisions {e['outcome']['collisions']}, "
              f"min clearance {e['outcome']['min_clearance_m']} m, {e['outcome']['termination']}")

    # ---- final safety check ------------------------------------------------------------------ #
    print("\n" + "=" * 96)
    print("FINAL SAFETY CHECK")
    print("=" * 96)
    checks = {
        "no collisions in any scenario or mode": all(r["collisions"] == 0 for r in results),
        "every scenario reached its goal": all(r["completed"] for r in results),
        "sensors mode uses fusion as the object provider":
            all(r["wiring"].get("object_provider_is_fusion", True)
                for r in results if r["mode"] == "sensors"),
        "ground-truth mode uses the world provider":
            all(r["wiring"].get("object_provider_is_world_truth", True)
                for r in results if r["mode"] == "ground_truth"),
        "planner consumes the scenario road": all(r["wiring"]["planner_road_is_scenario_road"]
                                                  for r in results),
        "replanning actually changes the trajectory somewhere":
            any(e["after"]["trajectory_changed"] for e in evidence) if evidence else False,
        "emergency braking fires where required":
            any(r["emergency_brakes"] > 0 for r in results),
        "reverse recovery fires where required":
            any(r["reverse_manoeuvres"] > 0 for r in results),
    }
    for k, v in checks.items():
        print(f"   [{'PASS' if v else 'FAIL'}] {k}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scenarios": results, "replanning_evidence": evidence, "safety_checks": checks,
        "note": ("The learned camera detector from Phases 6 and 7 is NOT in this closed loop: the "
                 "simulator renders no images. 'sensors' mode is the simulated camera, LiDAR and "
                 "radar models feeding the same Detection contract into the same tracker."),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
