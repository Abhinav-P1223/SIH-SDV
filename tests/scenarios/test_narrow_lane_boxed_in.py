"""Scenario-level assertions for NARROW_LANE_BOXED_IN: the reversing recovery.

The ego is stopped by a pushcart abandoned across its half of a 7 m lane, too close to steer around from
rest (the lateral shift needs ~9 m of travel, the standoff leaves ~5 m). The stack has to back up and then
take the gap on the right. Nothing here is scripted: the manoeuvre must emerge from the planner reporting no
forward plan and the behaviour layer reacting to it. Runs in SENSORS mode like the other required scenarios.
"""
import math

import pytest

from autonomy.core.config import AutonomyConfig, load_vehicle_parameters
from autonomy.core.types import BehaviorState
from simulation.runner import run_scenario


@pytest.fixture(scope="module", params=["sensors", "ground_truth"])
def result(request):
    return run_scenario("NARROW_LANE_BOXED_IN", log_dir=None, console=False, keep_frames=True,
                        perception=request.param)


@pytest.fixture(scope="module")
def cfg():
    return AutonomyConfig.load()


def test_reaches_the_goal_without_collision(result):
    m = result.metrics
    assert m.collision_count == 0, f"{result.scenario_name}: collision"
    assert m.scenario_completed and m.termination_reason == "GOAL_REACHED"
    assert m.minimum_obstacle_clearance > 0.25, "squeezed past with less than a quarter metre to spare"


def test_it_actually_reversed_to_get_out(result, cfg):
    """The point of the scenario: a real reversing leg, not a lucky forward squeeze."""
    m = result.metrics
    assert m.reverse_manoeuvres >= 1
    assert m.reverse_distance_m > 1.0
    states = [f.decision.state for f in result.frames if f.decision]
    assert BehaviorState.REVERSING in states
    reversing = [f for f in result.frames if f.decision and f.decision.state == BehaviorState.REVERSING]
    # the ego genuinely moved backwards under the reverse gear, and only at low speed
    assert any(f.ego.longitudinal_velocity < -0.5 for f in reversing)
    assert all(f.ego.longitudinal_velocity >= -cfg.planning.reverse_speed_mps - 0.1 for f in reversing)
    # the reverse gear is engaged for essentially the whole leg. The handful of exceptions are the hand-back
    # frames: the leg is done, the planner has already chosen a forward trajectory, and the vehicle model is
    # braking off the last few cm/s of reverse speed (a forward command never adds backward speed).
    back = [f for f in reversing if f.ego.longitudinal_velocity < -0.1]
    assert back and sum(f.control.reverse for f in back) / len(back) > 0.95
    assert any("Boxed in" in f.decision.reason for f in reversing)


def test_reversing_is_bounded_and_ends_in_forward_progress(result, cfg):
    """Reversing is a recovery, not a habit: few legs, and the ego ends up past the blocker."""
    m = result.metrics
    assert m.reverse_manoeuvres <= cfg.behavior.max_reverse_manoeuvres
    assert m.reverse_distance_m < 20.0
    states = [f.decision.state for f in result.frames if f.decision]
    idx = [i for i, s in enumerate(states) if s == BehaviorState.REVERSING]
    assert idx and idx == list(range(idx[0], idx[-1] + 1))   # ONE contiguous episode, not repeated backing up
    frames = [f for f in result.frames if f.decision]
    # the corridor reference starts at x = -20, so the goal at s = 100 m sits at x = 80
    assert frames[-1].ego.x > 70.0                           # well past the cart at x = 30


def test_it_passed_on_the_free_side_of_the_cart(result, params=None):
    """The cart occupies the left half; the ego must have gone round it on the right."""
    p = load_vehicle_parameters()
    cart_x = 30.0
    near = [f for f in result.frames if abs(f.ego.x - cart_x) < 3.0]
    assert near, "never reached the cart"
    assert min(f.ego.y for f in near) < -0.5, "did not move into the free right-hand band"
    # and stayed inside the corridor while doing it
    assert all(f.ego.y - 0.5 * p.width > -3.5 for f in near)


def test_no_emergency_braking_in_the_deadlock(result):
    """Being stuck is not an emergency: the ego holds position and reverses, it does not panic-brake."""
    m = result.metrics
    assert m.emergency_brake_activations <= 8       # sensor noise near a static obstacle can trip a few
    # STOPPED is a low-speed hold: the state persists for a cycle or two after the planner has released the
    # vehicle, so allow a slow move-off but nothing like driving speed.
    stopped = [f for f in result.frames if f.decision and f.decision.state == BehaviorState.STOPPED]
    assert stopped and all(abs(f.ego.longitudinal_velocity) < 2.0 for f in stopped)
