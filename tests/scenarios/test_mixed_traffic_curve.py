"""Scenario-level assertions for MIXED_TRAFFIC_CURVE: curved reference,
merging auto-rickshaw with road following, erratic pedestrian, several agents."""
import math

import numpy as np
import pytest

from autonomy.core.config import AutonomyConfig, load_vehicle_parameters
from autonomy.core.types import BehaviorState
from simulation.runner import run_scenario


@pytest.fixture(scope="module")
def result():
    return run_scenario("MIXED_TRAFFIC_CURVE", log_dir=None, console=False, keep_frames=True)


def test_no_collision_completed_and_margin(result):
    m = result.metrics
    assert m.collision_count == 0 and m.scenario_completed and m.termination_reason == "GOAL_REACHED"
    assert m.minimum_obstacle_clearance >= AutonomyConfig.load().planning.safety_margin_m


def test_ego_followed_the_curve_and_stayed_on_road(result):
    yaws = [f.ego.yaw for f in result.frames]
    assert max(yaws) > math.radians(30)              # actually turned through most of the 40 deg curve
    # every frame inside the corridor: |d| + half width <= 3.5 (computed from the road model)
    road = result.frames[0].road
    from simulation.world.road import DrivableSpace
    r = DrivableSpace(np.asarray(road["reference"]), np.asarray(road["left_boundary"]), np.asarray(road["right_boundary"]))
    for f in result.frames[::10]:
        _, d, _ = r.project(f.ego.x, f.ego.y)
        assert abs(d) <= 3.5 - 0.9 + 0.05


def test_merging_agent_merged_and_followed_road(result):
    phases = {a["phase"] for f in result.frames for a in f.agents if a["id"] == "auto_1"}
    assert {"APPROACH", "MERGING", "MERGED"} <= phases
    last = [a for a in result.frames[-1].agents if a["id"] == "auto_1"][0]
    assert last["speed"] == pytest.approx(5.0, abs=0.05)


def test_erratic_agent_moved_deterministically(result):
    xs = [a["x"] for f in result.frames for a in f.agents if a["id"] == "ped_1"]
    assert max(xs) - min(xs) > 1.0
    again = run_scenario("MIXED_TRAFFIC_CURVE", log_dir=None, console=False, keep_frames=True)
    xs2 = [a["x"] for f in again.frames for a in f.agents if a["id"] == "ped_1"]
    assert xs == xs2                                  # seeded: bit-identical replay


def test_follow_state_was_used_and_no_chatter(result):
    states = [f.decision.state for f in result.frames if f.planning_cycle and f.decision]
    assert BehaviorState.FOLLOW in states
    transitions = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert transitions <= 12, f"{transitions} transitions in {len(states)} cycles looks like chatter"
    # the slower auto-rickshaw is still ahead at the goal, so ending in FOLLOW is the correct outcome
    assert states[-1] in (BehaviorState.CRUISE, BehaviorState.FOLLOW)
    assert BehaviorState.EMERGENCY_BRAKE not in states


def test_planner_latency_budget_on_curve(result):
    cfg = AutonomyConfig.load()
    lat = np.array([f.plan.latency_ms for f in result.frames if f.planning_cycle])
    assert lat.mean() < cfg.planning.period_s * 1000.0
    assert np.percentile(lat, 95) < 1.5 * cfg.planning.period_s * 1000.0
