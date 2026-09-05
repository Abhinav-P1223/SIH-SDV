"""Jury audit: stress battery. Tries to break the closed loop and reports
PASS / DEGRADED / FAIL per case, with what the system actually did.

PASS      completed, no collision, clearance >= margin
DEGRADED  no collision but not completed (stopped / timeout), or margin violated
FAIL      collision; annotated "agent-initiated" when the ego was stationary at impact
          (an agent walked into a stopped ego; only reversing could avoid it)

Sensor-noise / dropout / delay cases wrap the ground-truth provider, since the
repository has no sensor models; they show how the stack degrades when the
ObjectState contract is fed imperfect data.

    python scripts/audit_stress.py
"""
from __future__ import annotations

import copy
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autonomy.core.config import AutonomyConfig, ObjectProfiles, PerceptionConfig, load_vehicle_parameters, load_yaml  # noqa: E402
from autonomy.core.interfaces import ObjectStateProvider  # noqa: E402
from autonomy.core.types import ObjectState  # noqa: E402
from autonomy.telemetry.telemetry import InMemorySink, TelemetryPublisher  # noqa: E402
from simulation.runner import Simulation  # noqa: E402
from simulation.scenarios.loader import SCENARIO_DIR, build_scenario  # noqa: E402

ROAD = {"lane_markings": "NONE", "reference": [[-20, 0], [200, 0]],
        "left_boundary": [[-20, 3.5], [200, 3.5]], "right_boundary": [[-20, -3.5], [200, -3.5]]}
EGO = {"x": 0.0, "y": 1.75, "yaw_deg": 0.0, "speed_mps": 10.0, "desired_speed_mps": 10.0, "desired_lateral_offset_m": 1.75}


def scenario(name, agents, ego=None, road=None, goal=150.0, seed=5):
    return {"name": name, "seed": seed, "road": road or ROAD, "ego": ego or EGO, "goal": {"s_m": goal}, "agents": agents}


class NoisyProvider(ObjectStateProvider):
    """Gaussian position/velocity noise, random dropout and measurement delay on top of ground truth."""

    def __init__(self, inner, sigma_pos=0.0, sigma_vel=0.0, dropout=0.0, delay_s=0.0, dt=0.02, seed=0):
        self.inner, self.sp, self.sv, self.p_drop, self.delay = inner, sigma_pos, sigma_vel, dropout, delay_s
        self.rng = np.random.default_rng(seed)
        self.buffer: deque = deque()
        self.dt = dt
        self.dropped = 0
        self.total = 0

    def get_object_states(self, timestamp):
        truth = self.inner.get_object_states(timestamp)
        self.buffer.append((timestamp, truth))
        while self.buffer and timestamp - self.buffer[0][0] > self.delay + 1e-9:
            self.buffer.popleft()
        objs = self.buffer[0][1] if self.delay > 0 else truth
        meas_time = self.buffer[0][0] if self.delay > 0 else timestamp     # honest measurement time stamp
        out = []
        for o in objs:
            self.total += 1
            if self.p_drop > 0 and self.rng.random() < self.p_drop:
                self.dropped += 1
                continue
            cov = np.eye(2) * max(self.sp, 1e-3) ** 2
            out.append(ObjectState(o.id, o.object_type, meas_time,
                                   o.x + self.rng.normal(0, self.sp), o.y + self.rng.normal(0, self.sp),
                                   o.vx + self.rng.normal(0, self.sv), o.vy + self.rng.normal(0, self.sv),
                                   o.heading, o.length, o.width, 1.0, cov))
        return out


PERCEPTION_MODE = "ground_truth"      # overridden by --perception


def run(data, provider_kwargs=None, cfg=None, sensor_override=None):
    cfg = cfg or AutonomyConfig.load()
    sc = build_scenario(data)
    if provider_kwargs:
        sc.world._provider = NoisyProvider(sc.world._provider, **provider_kwargs)
    mem = InMemorySink()
    perception = PerceptionConfig.load().with_mode(PERCEPTION_MODE)
    if sensor_override:
        perception = sensor_override(perception)
    sim = Simulation(sc, cfg, load_vehicle_parameters(), ObjectProfiles.load(), TelemetryPublisher([mem]),
                     perception=perception)
    m = sim.run()
    return m, mem.frames, sc


def classify(m, margin, frames=None, road=None):
    if m.collision_count > 0:
        # a free-moving agent (not road-following traffic) walking into a STATIONARY ego is agent-initiated:
        # only reversing could avoid it. Road traffic hitting a stopped ego is the ego's failure to clear the path.
        info = contact_info(frames, road) if frames is not None else None
        if info is not None and info[0] < 0.1 and not info[1]:
            return "FAIL (agent-initiated: free-moving agent walked into the stationary ego)"
        return "FAIL"
    if not m.scenario_completed or m.minimum_obstacle_clearance < margin:
        return "DEGRADED"
    return "PASS"


def contact_info(frames, road=None):
    """(ego speed, agent_is_road_following) at the first frame where the ego footprint touches a truth agent."""
    from autonomy.core.geometry import OrientedBox, box_distance, wrap_angle
    from autonomy.vehicle.models import footprint_for
    P = load_vehicle_parameters()
    for f in frames:
        fp = footprint_for(P, f.ego.x, f.ego.y, f.ego.yaw)
        for a in f.agents:
            if box_distance(fp, OrientedBox(a["x"], a["y"], a["heading"], 1.0, 1.0)) <= 0.5:
                following = False
                if road is not None:
                    _, _, h_ref = road.project(a["x"], a["y"])
                    rel = abs(float(wrap_angle(a["heading"] - h_ref)))
                    following = (rel < math.radians(30) or abs(rel - math.pi) < math.radians(30)) and a["speed"] > 1.0
                return f.ego.longitudinal_velocity, following
    return None


def row(name, m, frames, verdict, note=""):
    states = []
    for f in frames:
        if f.decision and (not states or states[-1] != f.decision.state.value):
            states.append(f.decision.state.value)
    ttc = "inf" if math.isinf(m.minimum_ttc) else f"{m.minimum_ttc:.2f}"
    clr = "inf" if math.isinf(m.minimum_obstacle_clearance) else f"{m.minimum_obstacle_clearance:.2f}"
    done = "n/a" if m.time_to_completion is None else f"{m.time_to_completion:.1f}"
    vmin = min(f.ego.longitudinal_velocity for f in frames)
    return (f"| {name} | **{verdict}** | {m.termination_reason} | {m.collision_count} | {clr} | {ttc} | "
            f"{m.emergency_brake_activations} | {vmin:.1f} | {' > '.join(states)} | {done} | {note} |")


def main() -> int:
    global PERCEPTION_MODE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", choices=["ground_truth", "sensors"], default="ground_truth")
    PERCEPTION_MODE = ap.parse_args().perception
    cfg = AutonomyConfig.load()
    margin = cfg.planning.safety_margin_m
    cattle = load_yaml(SCENARIO_DIR / "sudden_cattle_crossing.yaml")
    cases = []

    # 1 two simultaneous crossers from both sides at the same station
    cases.append(("two simultaneous crossers (cow from left, pedestrian from right)", scenario("TWO_CROSSERS", [
        {"id": "cattle_1", "type": "CATTLE", "x": 60, "y": 3.1, "heading_deg": -90, "behavior": "CROSSING",
         "params": {"trigger_distance_m": 30, "crossing_speed_mps": 0.8, "crossing_distance_m": 6.0}},
        {"id": "ped_1", "type": "PEDESTRIAN", "x": 62, "y": -3.2, "heading_deg": 90, "behavior": "CROSSING",
         "params": {"trigger_distance_m": 28, "crossing_speed_mps": 1.4, "crossing_distance_m": 6.4}},
    ]), None))
    # 2 wrong-way motorcycle in the ego's half
    cases.append(("wrong-way motorcycle head-on in ego half (8 m/s)", scenario("WRONG_WAY", [
        {"id": "moto_1", "type": "MOTORCYCLE", "x": 120, "y": 1.2, "heading_deg": 180, "speed_mps": 8.0,
         "behavior": "CONSTANT_VELOCITY", "params": {}}]), None))
    # 3 auto-rickshaw cutting in from the right side into the ego offset
    cases.append(("auto-rickshaw cuts in from the right, 22 m ahead", scenario("CUT_IN", [
        {"id": "auto_1", "type": "AUTO_RICKSHAW", "x": 22, "y": -2.4, "heading_deg": 35, "speed_mps": 4.0,
         "behavior": "MERGING", "params": {"trigger_distance_m": 100, "follow_road": True, "target_speed_mps": 4.0,
                                           "yaw_rate_rad_s": 0.6}}]), None))
    # 4 blocked road: truck across the whole corridor
    cases.append(("blocked road: truck across the corridor (no safe trajectory)", scenario("BLOCKED", [
        {"id": "truck_1", "type": "TRUCK", "x": 70, "y": 0.0, "heading_deg": 90, "behavior": "STATIC", "params": {}}]),
        None))
    # 5 narrow road (3.4 m) with a crossing cow -> braking only
    narrow_road = {"lane_markings": "NONE", "reference": [[-20, 0], [200, 0]],
                   "left_boundary": [[-20, 1.7], [200, 1.7]], "right_boundary": [[-20, -1.7], [200, -1.7]]}
    cases.append(("narrow 3.4 m road, cow crossing (no lateral room)", scenario("NARROW", [
        {"id": "cattle_1", "type": "CATTLE", "x": 60, "y": 2.3, "heading_deg": -90, "behavior": "CROSSING",
         "params": {"trigger_distance_m": 30, "crossing_speed_mps": 0.8, "crossing_distance_m": 5.0}}],
        ego={**EGO, "y": 0.0, "desired_lateral_offset_m": 0.0}, road=narrow_road), None))
    # 6 dense mixed traffic
    cases.append(("dense mixed traffic: 6 agents (slow car, oncoming car, pushcart, 2 pedestrians, cow)", scenario("DENSE", [
        {"id": "car_lead", "type": "CAR", "x": 35, "y": 1.75, "heading_deg": 0, "speed_mps": 6.0, "behavior": "CONSTANT_VELOCITY", "params": {}},
        {"id": "car_onc", "type": "CAR", "x": 140, "y": -1.75, "heading_deg": 180, "speed_mps": 7.0, "behavior": "CONSTANT_VELOCITY", "params": {}},
        {"id": "cart_1", "type": "PUSHCART", "x": 70, "y": 2.9, "heading_deg": 0, "behavior": "STATIC", "params": {}},
        {"id": "ped_a", "type": "PEDESTRIAN", "x": 90, "y": -3.2, "heading_deg": 90, "behavior": "CROSSING",
         "params": {"trigger_distance_m": 25, "crossing_speed_mps": 1.2, "crossing_distance_m": 6.4}},
        {"id": "ped_b", "type": "PEDESTRIAN", "x": 95, "y": 3.2, "heading_deg": -90, "behavior": "CROSSING",
         "params": {"trigger_distance_m": 30, "crossing_speed_mps": 1.0, "crossing_distance_m": 6.4}},
        {"id": "cattle_1", "type": "CATTLE", "x": 120, "y": 3.0, "heading_deg": -90, "behavior": "CROSSING",
         "params": {"trigger_distance_m": 30, "crossing_speed_mps": 0.7, "crossing_distance_m": 6.0}},
    ], goal=170.0), None))
    # 7 high-speed approach: 20 m/s
    fast = copy.deepcopy(cattle); fast["ego"]["speed_mps"] = 20.0; fast["ego"]["desired_speed_mps"] = 20.0
    fast["agents"][0]["params"]["trigger_distance_m"] = 45.0
    cases.append(("high-speed approach 20 m/s (72 km/h), cow crossing triggered at 45 m", fast, None))
    if PERCEPTION_MODE == "ground_truth":
        # 8-11 imperfect ObjectState input straight into the planner (no filtering)
        cases.append(("position noise 0.5 m, velocity noise 0.4 m/s (seeded)", copy.deepcopy(cattle),
                      {"sigma_pos": 0.5, "sigma_vel": 0.4, "seed": 1}))
        cases.append(("30 % detection dropout", copy.deepcopy(cattle), {"dropout": 0.3, "seed": 2}))
        cases.append(("0.4 s measurement delay", copy.deepcopy(cattle), {"delay_s": 0.4}))
        cases.append(("noise 0.3 m/0.3 m/s + 20 % dropout + 0.2 s delay", copy.deepcopy(cattle),
                      {"sigma_pos": 0.3, "sigma_vel": 0.3, "dropout": 0.2, "delay_s": 0.2, "seed": 3}))
    else:
        # 8-11 sensor degradation through the real tracker: ablations and a degraded suite
        def degraded(pc):
            pc = pc.with_mode("sensors")
            for name in pc.sensors:
                s = pc.sensors[name]
                for k in ("bearing_std_deg", "range_std_frac", "range_std_min_m", "position_std_m", "range_std_m",
                          "radial_speed_std_mps", "extent_std_m", "heading_std_deg"):
                    if k in s:
                        s[k] = s[k] * 2.0
                s["dropout_prob"] = min(0.6, s.get("dropout_prob", 0.0) + 0.3)
                s["latency_s"] = s.get("latency_s", 0.0) + 0.1
            return pc
        cases.append(("camera disabled (LiDAR + radar only, classes UNKNOWN)", copy.deepcopy(cattle), None,
                      lambda pc: pc.without("camera")))
        cases.append(("LiDAR disabled (camera + radar only)", copy.deepcopy(cattle), None,
                      lambda pc: pc.without("lidar")))
        cases.append(("radar disabled (camera + LiDAR only, no radial speed)", copy.deepcopy(cattle), None,
                      lambda pc: pc.without("radar")))
        cases.append(("degraded suite: 2x noise, +30 % dropout, +0.1 s latency", copy.deepcopy(cattle), None, degraded))
    # 12 sudden direction change: cow stops mid-road then reverses (two-phase via ERRATIC with strong sigma)
    cases.append(("erratic cow wandering in the ego half (seeded random direction changes)", scenario("ERRATIC_COW", [
        {"id": "cattle_1", "type": "CATTLE", "x": 60, "y": 1.5, "heading_deg": -90, "speed_mps": 0.6,
         "behavior": "ERRATIC", "params": {"heading_sigma_rad": 0.8, "speed_sigma_mps": 0.4, "change_interval_s": 0.8,
                                           "max_speed_mps": 1.5}}]), None))

    print("## Stress battery\n")
    print("| case | verdict | termination | collisions | min clearance [m] | min TTC [s] | EB | min speed [m/s] | "
          "behaviour states | t_done [s] | note |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in cases:
        name, data, pk = case[0], case[1], case[2]
        so = case[3] if len(case) > 3 else None
        try:
            m, frames, sc = run(data, pk, cfg, so)
            note = ""
            if pk and "dropout" in pk:
                prov = sc.world._provider
                note = f"dropped {prov.dropped}/{prov.total} detections"
            print(row(name, m, frames, classify(m, margin, frames, sc.world.road), note))
        except Exception as exc:  # report, never hide
            print(f"| {name} | **ERROR** | exception | - | - | - | - | - | - | - | {type(exc).__name__}: {exc} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
