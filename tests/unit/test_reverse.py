"""Reversing recovery: vehicle reverse gear, reverse candidates, REVERSING state."""
import math

import pytest

from autonomy.behavior.state_machine import BehaviorStateMachine
from autonomy.core.config import AutonomyConfig, load_vehicle_parameters
from autonomy.core.types import (BehaviorDecision, BehaviorState, ControlCommand, ObjectType, RiskAssessment,
                                 RiskLevel, RiskSummary, SpeedPolicy, VehicleState)
from autonomy.planning.candidate_generator import CandidateGenerator
from autonomy.vehicle.models import KinematicBicycleModel
from simulation.world.road import DrivableSpace


@pytest.fixture(scope="module")
def cfg():
    return AutonomyConfig.load()


@pytest.fixture(scope="module")
def params():
    return load_vehicle_parameters()


def test_reverse_gear_moves_backwards_and_is_speed_limited(params):
    m = KinematicBicycleModel(params)
    s = VehicleState(0.0, 10.0, 0.0, 0.0, 0.0)
    for _ in range(200):                                   # 4 s of reverse "acceleration"
        s = m.step(s, ControlCommand(s.timestamp, 0.0, 1.0, 0.0, reverse=True), 0.02)
    assert s.longitudinal_velocity == pytest.approx(-params.max_reverse_speed)
    assert s.x < 10.0 - 4.0                                 # actually moved backwards
    # braking in reverse pulls |v| toward zero and never overshoots into forward motion
    for _ in range(200):
        s = m.step(s, ControlCommand(s.timestamp, 0.0, 0.0, 3.0, reverse=True), 0.02)
    assert s.longitudinal_velocity == pytest.approx(0.0)


def test_reverse_gear_is_not_engaged_while_rolling_forward(params):
    m = KinematicBicycleModel(params)
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 5.0)
    s = m.step(s, ControlCommand(0.0, 0.0, 2.0, 0.0, reverse=True), 0.1)
    assert 0.0 < s.longitudinal_velocity < 5.0            # braked, still forward


def test_reverse_candidates_only_when_allowed_and_stationary(cfg, params):
    road = DrivableSpace.straight(200.0, 3.5, 3.5, x0=-20.0)
    gen = CandidateGenerator(cfg.planning, params, road)
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    cands, _ = gen.generate(ego, SpeedPolicy(4.0, True, False, allow_reverse=True), 1.75, 0.0)
    rev = [c for c in cands if c.id.startswith("reverse_")]
    # Phase 5: one candidate per (leg length, lateral end offset). Before Phase 5 reverse was
    # straight-only, so this was one per leg length; the offsets are what curved reverse adds.
    offs = gen.reverse_offsets(gen.frame(ego))
    assert len(rev) == len(cfg.planning.reverse_distances_m) * len(offs)
    straight = [c for c in rev if abs(c.lateral_offset_end - 1.75) < 1e-6]
    assert straight, "the straight-back leg must still be offered"
    for c in rev:
        tr = c.trajectory
        assert tr.velocity.min() < 0 and tr.velocity[0] == 0.0
        # a short leg comes to rest inside the horizon; a long leg holds reverse speed to the end (planner re-plans)
        assert tr.velocity[-1] == 0.0 or abs(tr.velocity[-1]) == pytest.approx(cfg.planning.reverse_speed_mps)
        assert tr.x[-1] < tr.x[0]                                      # travels backwards along the corridor
        assert abs(tr.velocity.min()) <= cfg.planning.reverse_speed_mps + 1e-9
    for c in straight:
        assert abs(c.trajectory.y[-1] - 1.75) < 1e-6                   # straight back along the corridor
    # not allowed by policy -> none; allowed but moving -> none
    assert not [c for c in gen.generate(ego, SpeedPolicy(4.0, True, False), 1.75, 0.0)[0] if c.id.startswith("reverse_")]
    moving = VehicleState(0.0, 40.0, 1.75, 0.0, 3.0)
    assert not [c for c in gen.generate(moving, SpeedPolicy(4.0, True, False, allow_reverse=True), 1.75, 0.0)[0]
                if c.id.startswith("reverse_")]


def _risk(level):
    a = RiskAssessment("o", ObjectType.CATTLE, 3.0, 0.0, 0.0, math.inf, math.inf, 3.0, 0.0, False, 0.1, 0.3, level, 0.3, 1.4)
    s = RiskSummary([a], level, 0.3, math.inf, 3.0, "o", False)
    s.non_lead_max_level, s.non_lead_max_score = level, 0.3
    return s


def test_fsm_reverses_when_boxed_in_then_returns(cfg):
    fsm = BehaviorStateMachine(cfg.behavior, 8.0)
    stopped = VehicleState(0.0, 0, 0, 0, 0.0)
    fsm.decide(_risk(RiskLevel.CRITICAL), VehicleState(0.0, 0, 0, 0, 5.0), 0.0)     # -> EMERGENCY_BRAKE
    d = fsm.decide(_risk(RiskLevel.HIGH), stopped, 1.0)
    assert d.state == BehaviorState.STOPPED
    # boxed in: no forward candidate for longer than reverse_after_standstill_s
    d = fsm.decide(_risk(RiskLevel.HIGH), stopped, 2.0, planner_feasible=False, standstill_s=1.0)
    assert d.state == BehaviorState.STOPPED
    d = fsm.decide(_risk(RiskLevel.HIGH), stopped, 4.5, planner_feasible=False,
                   standstill_s=cfg.behavior.reverse_after_standstill_s + 0.1)
    assert d.state == BehaviorState.REVERSING and d.speed_policy.allow_reverse and "Boxed in" in d.reason
    # a forward trajectory becomes available again -> CAUTION
    d = fsm.decide(_risk(RiskLevel.MEDIUM), stopped, 5.2, planner_feasible=True)
    assert d.state == BehaviorState.CAUTION
