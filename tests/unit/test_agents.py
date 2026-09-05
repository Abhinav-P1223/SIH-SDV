import math

import pytest

from autonomy.core.types import AgentBehaviorType, ObjectType
from simulation.agents.agent import Agent
from simulation.agents.behaviors import behavior_for
from simulation.scenarios.loader import load_scenario


def make(behavior, **params):
    return Agent("a", ObjectType.CATTLE, 50.0, 3.0, -math.pi / 2, 0.0, 2.0, 0.7,
                 behavior, params, seed=7)


def run(agent, ego_xy, seconds, dt=0.02):
    b = behavior_for(agent)
    for _ in range(int(seconds / dt)):
        b.update(agent, ego_xy, dt)


def test_static_never_moves():
    a = make(AgentBehaviorType.STATIC)
    run(a, (0.0, 0.0), 2.0)
    assert (a.x, a.y) == (50.0, 3.0)


def test_constant_velocity():
    a = make(AgentBehaviorType.CONSTANT_VELOCITY)
    a.speed = 2.0
    run(a, (0.0, 0.0), 1.0)
    assert a.y == pytest.approx(1.0, abs=1e-6)


def test_crossing_waits_for_trigger_then_crosses_and_stops():
    a = make(AgentBehaviorType.CROSSING, trigger_distance_m=10.0, crossing_speed_mps=1.0,
             acceleration_mps2=100.0, crossing_distance_m=2.0)
    run(a, (0.0, 0.0), 1.0)              # ego far away -> waiting
    assert a.phase == "WAITING" and a.y == 3.0
    run(a, (45.0, 3.0), 1.0)             # ego within 10 m -> crosses 1 m
    assert a.phase == "CROSSING" and a.y == pytest.approx(2.0, abs=0.05)
    run(a, (45.0, 3.0), 2.0)
    assert a.phase == "DONE" and a.y == pytest.approx(1.0, abs=0.05) and a.speed == 0.0


def test_merging_turns_toward_target_heading():
    a = make(AgentBehaviorType.MERGING, trigger_distance_m=1000.0, target_heading_rad=0.0,
             yaw_rate_rad_s=1.0, target_speed_mps=3.0)
    a.speed = 1.0
    run(a, (0.0, 0.0), 3.0)
    assert a.heading == pytest.approx(0.0, abs=1e-6)
    assert a.speed == pytest.approx(3.0, abs=1e-6)


def test_erratic_is_deterministic_under_seed():
    a1 = make(AgentBehaviorType.ERRATIC)
    a2 = make(AgentBehaviorType.ERRATIC)
    a1.speed = a2.speed = 1.0
    run(a1, (0.0, 0.0), 5.0)
    run(a2, (0.0, 0.0), 5.0)
    assert (a1.x, a1.y, a1.heading) == (a2.x, a2.y, a2.heading)
    a3 = make(AgentBehaviorType.ERRATIC)
    a3.seed = 8
    a3.__post_init__()
    a3.speed = 1.0
    run(a3, (0.0, 0.0), 5.0)
    assert (a3.x, a3.y) != (a1.x, a1.y)


def test_scenario_loads_and_provides_object_states():
    sc = load_scenario("SUDDEN_CATTLE_CROSSING")
    assert sc.world.road.lane_markings == "NONE"
    objs = sc.world.object_provider.get_object_states(0.0)
    assert len(objs) == 1 and objs[0].object_type == ObjectType.CATTLE
    assert objs[0].length == 2.0 and objs[0].width == 0.7
    assert sc.ego.initial_state.longitudinal_velocity == 10.0
