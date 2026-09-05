"""Scenario-level assertions for SUDDEN_CATTLE_CROSSING.

These run the real closed loop. Nothing here is stubbed: if the planner,
controller or vehicle model regress, this test fails.
"""
import math

import pytest

from autonomy.core.config import AutonomyConfig
from autonomy.core.types import BehaviorState
from simulation.runner import run_scenario


@pytest.fixture(scope="module")
def result():
    return run_scenario("SUDDEN_CATTLE_CROSSING", log_dir=None, console=False, keep_frames=True)


@pytest.fixture(scope="module")
def cfg():
    return AutonomyConfig.load()


def test_no_collision_and_completed(result):
    m = result.metrics
    assert m.collision_count == 0
    assert m.scenario_completed is True
    assert m.termination_reason == "GOAL_REACHED"


def test_minimum_clearance_respects_safety_margin(result, cfg):
    assert result.metrics.minimum_obstacle_clearance >= cfg.planning.safety_margin_m


def test_cattle_actually_crossed_the_ego_path(result):
    """The threat must be real: the cattle footprint must have entered the ego's route band."""
    ys = [a["y"] for f in result.frames for a in f.agents if a["id"] == "cattle_1"]
    assert min(ys) < 1.75 < max(ys)          # crossed the ego's desired lateral offset
    phases = {a["phase"] for f in result.frames for a in f.agents if a["id"] == "cattle_1"}
    assert {"WAITING", "CROSSING", "DONE"} <= phases


def test_behavior_escalated_and_recovered(result):
    states = [f.decision.state for f in result.frames if f.decision]
    assert states[0] == BehaviorState.CRUISE
    assert BehaviorState.CAUTION in states or BehaviorState.AVOID in states
    assert BehaviorState.AVOID in states
    # left CRUISE and came back to it
    first_non_cruise = next(i for i, s in enumerate(states) if s != BehaviorState.CRUISE)
    assert BehaviorState.CRUISE in states[first_non_cruise:]
    assert states[-1] == BehaviorState.CRUISE


def test_risk_and_ttc_were_computed(result):
    m = result.metrics
    assert math.isfinite(m.minimum_ttc) and 0.0 < m.minimum_ttc < 4.0   # route-view TTC (desired speed), finite
    levels = {f.risk.max_level.name for f in result.frames}
    assert {"NONE", "HIGH"} <= levels


def test_planner_rejected_and_scored_candidates(result):
    plans = [f.plan for f in result.frames if f.planning_cycle]
    assert any(p.rejection_histogram.get("PREDICTED_COLLISION", 0) > 0 for p in plans)
    for p in plans:
        assert p.selected.feasible or p.selected.fallback
        if p.selected.feasible:
            assert set(p.selected.weighted_costs) == set(p.selected.costs)
            feasible = [c for c in p.candidates if c.feasible]
            assert p.selected.total_cost == min(c.total_cost for c in feasible)
    assert result.metrics.replanning_count > 50


def test_vehicle_was_controlled_not_teleported(result, cfg):
    from autonomy.core.config import load_vehicle_parameters
    params = load_vehicle_parameters()
    m = result.metrics
    assert m.max_steering_rate <= params.max_steering_rate + 1e-9
    assert m.max_acceleration <= params.max_acceleration + 1e-6
    assert m.max_deceleration <= params.max_deceleration + 1e-6
    steps = result.frames
    for a, b in zip(steps, steps[1:]):
        jump = math.hypot(b.ego.x - a.ego.x, b.ego.y - a.ego.y)
        assert jump <= params.max_speed * cfg.simulation.dt_s + 1e-9


def test_returned_to_route_and_boundaries_respected(result):
    last = result.frames[-1]
    assert abs(last.ego.y - 1.75) < 0.3
    for f in result.frames:
        assert -3.5 + 0.9 <= f.ego.y <= 3.5 - 0.9


def test_planner_latency_budget(result, cfg):
    """Planning must fit its 10 Hz budget on average, with bounded spikes (Python/numpy, not real-time)."""
    import numpy as np
    lat = np.array([f.plan.latency_ms for f in result.frames if f.planning_cycle])
    assert len(lat) > 50
    assert lat.mean() < cfg.planning.period_s * 1000.0
    assert np.percentile(lat, 95) < 1.5 * cfg.planning.period_s * 1000.0
    assert result.metrics.planning_latency_mean_ms == pytest.approx(lat.mean(), rel=1e-6)


def test_decisions_are_explained_and_logged(result):
    reasons = {f.decision.reason for f in result.frames if f.decision}
    assert any("intersects" in r for r in reasons)
    assert any("Risk cleared" in r or "resuming" in r for r in reasons)
