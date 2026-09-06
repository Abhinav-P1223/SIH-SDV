"""Phase 2: ego-aware traffic. IDM following plus APPROACH / YIELD / COMMIT.

Every test is deterministic. Nothing here depends on wall-clock time or on an agent being
scripted to act at a particular instant: the decisions fall out of the ego state the agent is
given, which is the whole point.
"""
import copy
import functools
import math

import numpy as np
import pytest

from autonomy.core.config import AutonomyConfig, ObjectProfiles, load_vehicle_parameters, load_yaml
from autonomy.core.types import AgentBehaviorType, ObjectType
from autonomy.telemetry.telemetry import InMemorySink, TelemetryPublisher
from simulation.agents.agent import Agent
from simulation.agents.behaviors import behavior_for
from simulation.agents.interaction import (APPROACH, COMMIT, YIELD, EgoView, GapAcceptance,
                                           idm_acceleration, time_to_point)
from simulation.runner import PerceptionConfig, Simulation, run_scenario
from simulation.scenarios.loader import SCENARIO_DIR, build_scenario
from simulation.world.road import DrivableSpace


def interactive_agent(x=60.0, y=-3.0, heading_deg=90.0, speed=2.0, **params):
    p = {"desired_speed_mps": 4.0, "conflict_x": 60.0, "conflict_y": 1.75,
         "critical_gap_s": 3.0, "reaction_time_s": 0.0, "follow_ego": False, "stay_on_road": False}
    p.update(params)
    return Agent("a", ObjectType.AUTO_RICKSHAW, x, y, math.radians(heading_deg), speed,
                 2.6, 1.4, AgentBehaviorType.INTERACTIVE, p)


def drive(agent, ego, seconds, dt=0.02, road=None):
    b = behavior_for(agent)
    for _ in range(int(seconds / dt)):
        b.update(agent, ego, dt, road)
    return agent


# ------------------------------------------------------------------ 1. ego state reaches agent --
def test_agent_receives_ego_position_speed_and_heading():
    """EgoView carries what a road user can see, and still indexes as (x, y) so every pre-existing
    behaviour that reads ego_xy[0]/[1] is unaffected."""
    e = EgoView(10.0, 2.0, 8.0, math.pi / 2)
    assert (e[0], e[1]) == (10.0, 2.0)
    assert e.speed == 8.0 and e.heading == pytest.approx(math.pi / 2)
    assert e.vx == pytest.approx(0.0, abs=1e-9) and e.vy == pytest.approx(8.0)

    # the agent's decision genuinely depends on the speed field, not just the position
    slow = interactive_agent()
    fast = interactive_agent()
    drive(slow, EgoView(0.0, 1.75, 6.0, 0.0), 0.5)
    drive(fast, EgoView(0.0, 1.75, 20.0, 0.0), 0.5)
    assert slow.memory["gap_state"] != fast.memory["gap_state"]


# --------------------------------------------------------------- 2/3. yield vs commit on gap ----
def test_small_gap_yields_and_large_gap_commits():
    g = GapAcceptance(critical_gap_s=3.0, hysteresis_s=0.8, reaction_time_s=0.0)
    assert g.decide(t_agent=2.0, t_ego=3.0) == YIELD        # 1.0 s gap, below critical
    g2 = GapAcceptance(critical_gap_s=3.0, hysteresis_s=0.8, reaction_time_s=0.0)
    assert g2.decide(t_agent=2.0, t_ego=9.0) == COMMIT      # 7.0 s gap, comfortably clear


def test_ego_speed_alone_flips_the_decision():
    """Identical geometry, different ego speed: the agent must reach the opposite conclusion."""
    conflict = (60.0, 1.75)
    for v_ego, expect in ((6.0, COMMIT), (25.0, YIELD)):
        a = interactive_agent()
        drive(a, EgoView(0.0, 1.75, v_ego, 0.0), 0.5)
        assert a.memory["gap_state"] == expect, f"ego at {v_ego} m/s should give {expect}"


def test_ego_heading_affects_conflict_assessment():
    """An ego driving away from the conflict point never arrives, so there is nothing to yield to."""
    towards = time_to_point(0.0, 1.75, 10.0, 0.0, 60.0, 1.75)
    away = time_to_point(0.0, 1.75, 10.0, math.pi, 60.0, 1.75)
    assert towards == pytest.approx(6.0)
    assert away < 0.0                                        # behind it, never reached


# ------------------------------------------------------------------------ 4. hysteresis ---------
def test_commitment_does_not_oscillate():
    """Once committed the agent stays committed, even as the gap collapses. That is what a human
    driver does, and it is what stops the state flapping cycle to cycle."""
    g = GapAcceptance(critical_gap_s=3.0, hysteresis_s=0.8, reaction_time_s=0.0)
    assert g.decide(2.0, 9.0) == COMMIT
    for t_ego in (8.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.0):
        assert g.decide(2.0, t_ego) == COMMIT

    # and a gap hovering on the threshold does not flip YIELD -> COMMIT without clearing hysteresis
    h = GapAcceptance(critical_gap_s=3.0, hysteresis_s=0.8, reaction_time_s=0.0)
    assert h.decide(2.0, 4.5) == YIELD                       # 2.5 s gap
    assert h.decide(2.0, 5.2) == YIELD                       # 3.2 s: above critical, below +hysteresis
    assert h.decide(2.0, 5.9) == COMMIT                      # 3.9 s: clears critical + hysteresis


def test_reaction_delay_is_respected():
    """The agent acts on an ego state as old as its reaction time, so it cannot respond instantly."""
    g = GapAcceptance(reaction_time_s=1.0)
    first = EgoView(0.0, 0.0, 5.0, 0.0)
    assert g.perceive(first, 0.0) is first
    assert g.perceive(EgoView(50.0, 0.0, 5.0, 0.0), 0.5) is first     # still the old view
    assert g.perceive(EgoView(90.0, 0.0, 5.0, 0.0), 1.6).x == 50.0    # 0.5 s sample is now current


# ------------------------------------------------------------------------ 5/6. IDM --------------
def test_idm_accelerates_when_the_gap_is_large():
    a = idm_acceleration(v=5.0, v0=10.0, gap=200.0, dv=0.0, a_max=1.5, b_comf=2.0)
    free = idm_acceleration(v=5.0, v0=10.0, gap=math.inf, dv=0.0, a_max=1.5, b_comf=2.0)
    assert a > 0.5 and free > 0.5
    assert a == pytest.approx(free, abs=0.05)


def test_idm_brakes_when_the_gap_is_small():
    a = idm_acceleration(v=10.0, v0=12.0, gap=3.0, dv=10.0, a_max=1.5, b_comf=2.0)
    assert a < -1.0


def test_idm_converges_to_the_desired_headway():
    """Following a lead at constant speed, the gap settles near s0 + v*T rather than drifting."""
    v_lead, T, s0, dt = 8.0, 1.5, 2.0, 0.05
    v, gap = 8.0, 40.0
    for _ in range(4000):
        acc = idm_acceleration(v, 12.0, gap, v - v_lead, a_max=1.5, b_comf=2.0, s0=s0, headway_s=T)
        v = max(0.0, v + acc * dt)
        gap += (v_lead - v) * dt
    assert gap == pytest.approx(s0 + v_lead * T, rel=0.25)
    assert v == pytest.approx(v_lead, abs=0.5)


def test_idm_never_produces_runaway_commands():
    for gap in (0.0, 0.01, 1e9):
        for v in (0.0, 30.0):
            acc = idm_acceleration(v, 10.0, gap, v, a_max=1.5, b_comf=2.0)
            assert math.isfinite(acc) and -6.0 <= acc <= 1.5


# ------------------------------------------------------------------ 7. road containment ---------
def test_interactive_agent_stops_at_the_road_edge():
    """Containment is a motion command, not a position clamp: the agent brakes and stops on the
    road rather than being teleported back onto it."""
    road = DrivableSpace.straight(200.0, 4.0, 4.0, x0=-20.0)
    a = interactive_agent(x=60.0, y=-3.0, heading_deg=90.0, speed=3.0,
                          conflict_x=None, conflict_y=None, stay_on_road=True,
                          desired_speed_mps=4.0)
    drive(a, EgoView(0.0, 1.75, 8.0, 0.0), 20.0, road=road)
    nose = a.y + 0.5 * a.length * math.sin(a.heading)
    assert -4.0 <= nose <= 4.0, f"agent's nose left the corridor at y={nose:.2f}"
    assert a.speed == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------- 8. scenarios and the ON/OFF ablation ---
@functools.lru_cache(maxsize=None)
def _run(name, interactive, ego_speed=None, mode="ground_truth"):
    """Cached so repeated (scenario, flag, speed) combinations cost one simulation, and
    ground truth by default: the interaction lives on the AGENT side, so perception mode is
    irrelevant to what these tests assert and a sensors run would cost several times more.
    The two "completes safely" tests below still use sensors, which is where it matters."""
    data = copy.deepcopy(load_yaml(SCENARIO_DIR / f"{name.lower()}.yaml"))
    if ego_speed is not None:
        data["ego"]["speed_mps"] = data["ego"]["desired_speed_mps"] = ego_speed
    cfg = AutonomyConfig.load()
    cfg.simulation.interactive_traffic = interactive
    mem = InMemorySink()
    profiles = ObjectProfiles.load()
    sim = Simulation(build_scenario(data, profiles), cfg, load_vehicle_parameters(), profiles,
                     TelemetryPublisher([mem]), perception=PerceptionConfig.load().with_mode(mode))
    metrics = sim.run()
    phases = []
    for f in mem.frames:
        ph = f.agents[0]["phase"]
        if not phases or phases[-1] != ph:
            phases.append(ph)
    return metrics, phases


@pytest.mark.parametrize("name", ["UNPROTECTED_TURN", "NARROW_LANE_MUTUAL_YIELD"])
def test_interaction_scenarios_complete_safely(name):
    m, _ = _run(name, interactive=True, mode="sensors")     # the full closed loop, sensors included
    assert m.collision_count == 0
    assert m.scenario_completed, f"{name} did not finish: {m.termination_reason}"


def test_narrow_lane_does_not_deadlock():
    """Two vehicles meet where only one fits. Neither is scripted; the pair must still resolve it."""
    m, _ = _run("NARROW_LANE_MUTUAL_YIELD", interactive=True)
    assert m.scenario_completed and m.termination_reason == "GOAL_REACHED"
    assert m.collision_count == 0
    assert m.average_speed > 0.5, "ego crawled the whole way: effectively a deadlock"


def test_interactive_flag_changes_agent_behaviour():
    """With the flag off the agent cannot see the ego's speed, so it commits regardless. With it on
    a fast-approaching ego makes it wait. Same scenario, same seed, different traffic."""
    on, phases_on = _run("UNPROTECTED_TURN", interactive=True, ego_speed=16.0)
    off, phases_off = _run("UNPROTECTED_TURN", interactive=False, ego_speed=16.0)
    assert YIELD in phases_on, f"expected a yield with interaction on, saw {phases_on}"
    assert YIELD not in phases_off, f"baseline must not yield, saw {phases_off}"
    assert on.collision_count == 0 and off.collision_count == 0


def test_agent_commits_when_the_ego_is_slow_enough():
    """The mirror of the test above: a slow ego leaves a gap worth taking."""
    m, phases = _run("UNPROTECTED_TURN", interactive=True, ego_speed=8.0)
    assert phases[0] == COMMIT, f"expected an immediate commit, saw {phases}"
    assert m.collision_count == 0
