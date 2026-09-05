"""Scenario-level assertions for SUDDEN_PEDESTRIAN_DART: exercises the
independent safety supervisor and the EMERGENCY_BRAKE path."""
import pytest

from autonomy.core.config import AutonomyConfig
from autonomy.core.types import BehaviorState
from simulation.runner import run_scenario


@pytest.fixture(scope="module")
def result():
    return run_scenario("SUDDEN_PEDESTRIAN_DART", log_dir=None, console=False, keep_frames=True)


def test_no_collision_and_completed(result):
    m = result.metrics
    assert m.collision_count == 0 and m.scenario_completed and m.termination_reason == "GOAL_REACHED"
    assert m.minimum_obstacle_clearance >= AutonomyConfig.load().planning.safety_margin_m


def test_safety_override_actually_fired_and_was_recorded(result):
    m = result.metrics
    assert m.emergency_brake_activations >= 1
    overrides = [f for f in result.frames if f.safety.override_active]
    assert overrides and all(f.control.source == "safety" for f in overrides)
    assert all(f.control.brake > 0 and f.control.acceleration == 0 for f in overrides)
    assert any("TTC" in f.safety.reason or "fallback" in f.safety.reason for f in overrides)


def test_emergency_brake_state_visited_and_recovered(result):
    states = [f.decision.state for f in result.frames if f.decision]
    assert BehaviorState.EMERGENCY_BRAKE in states
    assert states[-1] == BehaviorState.CRUISE
    # deceleration reached the braking limit but never exceeded it
    from autonomy.core.config import load_vehicle_parameters
    p = load_vehicle_parameters()
    assert result.metrics.max_deceleration == pytest.approx(p.max_deceleration, abs=0.05)


def test_pedestrian_really_crossed_the_path(result):
    ys = [a["y"] for f in result.frames for a in f.agents if a["id"] == "ped_1"]
    assert min(ys) < 0.85 < 2.65 < max(ys)      # traversed the whole ego body band
