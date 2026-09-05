"""The five SIH26037 required scenarios (village road, unsignalized intersection,
highway merge, dense market, cattle crossing) run in SENSORS mode: camera +
LiDAR + radar -> tracker/fusion -> planner. Every assertion is on executed
states against simulator truth. Cattle crossing has its own detailed test."""
import math

import pytest

from autonomy.core.config import AutonomyConfig
from autonomy.core.types import BehaviorState
from simulation.runner import run_scenario

REQUIRED = ["UNMARKED_VILLAGE_ROAD", "UNSIGNALIZED_INTERSECTION", "HIGHWAY_MERGE_SLOW_VEHICLES",
            "DENSE_MARKET_MIXED_TRAFFIC"]


@pytest.fixture(scope="module", params=REQUIRED)
def result(request):
    return run_scenario(request.param, log_dir=None, console=False, keep_frames=True, perception="sensors")


def test_completed_without_collision(result):
    m = result.metrics
    assert m.collision_count == 0, f"{result.scenario_name}: collision"
    assert m.scenario_completed and m.termination_reason == "GOAL_REACHED", result.scenario_name


def test_perception_was_actually_in_the_loop(result):
    m = result.metrics
    assert result.frames[0].perception_mode == "sensors"
    assert m.detections_total > 100
    assert m.tracking_position_error_m is not None and m.tracking_position_error_m < 0.8
    ids = {o.id for f in result.frames for o in f.objects}
    assert ids and all(i.startswith("trk_") for i in ids)          # planner saw tracks, not scenario ids


def test_actuator_limits_and_no_teleport(result):
    from autonomy.core.config import load_vehicle_parameters
    p = load_vehicle_parameters()
    cfg = AutonomyConfig.load()
    m = result.metrics
    assert m.max_steering_rate <= p.max_steering_rate + 1e-9
    assert m.max_acceleration <= p.max_acceleration + 1e-6
    assert m.max_deceleration <= p.max_deceleration + 1e-6
    fr = result.frames
    assert all(math.hypot(b.ego.x - a.ego.x, b.ego.y - a.ego.y) <= p.max_speed * cfg.simulation.dt_s + 1e-9
               for a, b in zip(fr, fr[1:]))


def test_ego_stayed_inside_the_corridor(result):
    import numpy as np
    from simulation.world.road import DrivableSpace
    road = result.frames[0].road
    r = DrivableSpace(np.asarray(road["reference"]), np.asarray(road["left_boundary"]), np.asarray(road["right_boundary"]))
    dr, dl = r.lateral_bounds(np.array([0.0]))
    for f in result.frames[::5]:
        s, d, _ = r.project(f.ego.x, f.ego.y)
        dr, dl = r.lateral_bounds(np.array([s]))
        assert dr[0] + 0.9 - 0.05 <= d <= dl[0] - 0.9 + 0.05, f"{result.scenario_name}: off corridor at t={f.timestamp:.2f}"


def test_planner_replanned_and_explained(result):
    m = result.metrics
    assert m.replanning_count > 50
    reasons = {f.decision.reason for f in result.frames if f.decision}
    assert len(reasons) >= 2
    plans = [f.plan for f in result.frames if f.planning_cycle]
    assert any(p.rejected_count > 0 for p in plans)
    assert m.planning_latency_mean_ms < AutonomyConfig.load().planning.period_s * 1000.0


def test_scenarios_are_genuinely_different(request):
    """Guard against 'five screenshots of the same scenario': agent sets and roads differ."""
    from autonomy.core.config import load_yaml
    from simulation.scenarios.loader import SCENARIO_DIR
    sigs = []
    for name in REQUIRED + ["SUDDEN_CATTLE_CROSSING"]:
        d = load_yaml(SCENARIO_DIR / f"{name.lower()}.yaml")
        sigs.append((tuple(sorted(a["type"] for a in d["agents"])), str(d["road"]), d["ego"]["desired_speed_mps"]))
    assert len(set(sigs)) == 5
    types = {t for s in sigs for t in s[0]}
    assert {"CATTLE", "PEDESTRIAN", "AUTO_RICKSHAW", "TRUCK", "BUS", "MOTORCYCLE", "PUSHCART", "BICYCLE", "CAR"} <= types
