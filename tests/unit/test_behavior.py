import math

import pytest

from autonomy.behavior.state_machine import BehaviorStateMachine
from autonomy.core.config import AutonomyConfig
from autonomy.core.types import (BehaviorState, ObjectType, RiskAssessment, RiskLevel, RiskSummary,
                                 VehicleState)


@pytest.fixture
def cfg():
    return AutonomyConfig.load().behavior


def risk(level, score, ttc=math.inf, intersection=False, lead=None, lead_speed=0.0, lead_gap=math.inf,
         ttc_phys=None):
    a = RiskAssessment("obj", ObjectType.CATTLE, 20.0, 5.0, 5.0, ttc, ttc, 1.0, 2.0,
                       intersection, 0.5, score, level, 1.0, 1.4)
    s = RiskSummary([a], level, score, ttc, 1.0, "obj", intersection)
    s.lead_object_id, s.lead_speed, s.lead_gap = lead, lead_speed, lead_gap
    s.min_ttc_current_speed = ttc if ttc_phys is None else ttc_phys
    return s


def ego(v=10.0):
    return VehicleState(0.0, 0.0, 0.0, 0.0, v)


def test_cruise_to_caution_to_avoid_to_emergency(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    d = fsm.decide(RiskSummary.empty(), ego(), 0.0)
    assert d.state == BehaviorState.CRUISE and d.speed_policy.target_speed == 10.0

    d = fsm.decide(risk(RiskLevel.MEDIUM, 0.25, ttc=3.8, intersection=True), ego(), 0.1)
    assert d.state == BehaviorState.CAUTION
    assert d.speed_policy.target_speed == pytest.approx(10.0 * cfg.caution_speed_factor)
    assert "3.8 s" in d.reason

    d = fsm.decide(risk(RiskLevel.HIGH, 0.5, ttc=2.5, intersection=True), ego(), 0.2)
    assert d.state == BehaviorState.AVOID and "intersects" in d.reason
    assert d.speed_policy.allow_lateral_avoidance

    d = fsm.decide(risk(RiskLevel.CRITICAL, 0.9, ttc=0.8, intersection=True), ego(), 0.3)
    assert d.state == BehaviorState.EMERGENCY_BRAKE
    assert d.speed_policy.force_stop and d.speed_policy.target_speed == 0.0
    assert d.triggers["min_ttc"] == pytest.approx(0.8)


def test_escalation_ignores_dwell_but_release_respects_it(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    fsm.decide(risk(RiskLevel.HIGH, 0.5, ttc=2.5, intersection=True), ego(), 0.0)
    assert fsm.state == BehaviorState.AVOID
    # risk gone immediately: must wait min_dwell before releasing
    d = fsm.decide(risk(RiskLevel.NONE, 0.0), ego(), 0.1)
    assert d.state == BehaviorState.AVOID
    d = fsm.decide(risk(RiskLevel.NONE, 0.0), ego(), cfg.min_dwell_s + 0.05)
    assert d.state == BehaviorState.CAUTION
    d = fsm.decide(risk(RiskLevel.NONE, 0.0), ego(), 2 * cfg.min_dwell_s + 0.2)
    assert d.state == BehaviorState.CRUISE


def test_hysteresis_on_exit_thresholds(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    fsm.decide(risk(RiskLevel.HIGH, 0.5, ttc=2.5, intersection=True), ego(), 0.0)
    # intersection cleared but score still above avoid_exit -> stay in AVOID
    d = fsm.decide(risk(RiskLevel.MEDIUM, cfg.avoid_exit_score + 0.05), ego(), 1.0)
    assert d.state == BehaviorState.AVOID
    d = fsm.decide(risk(RiskLevel.MEDIUM, cfg.avoid_exit_score - 0.05), ego(), 2.0)
    assert d.state == BehaviorState.CAUTION


def test_stopped_and_release(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    fsm.decide(risk(RiskLevel.CRITICAL, 0.9, ttc=0.5, intersection=True), ego(), 0.0)
    assert fsm.state == BehaviorState.EMERGENCY_BRAKE
    d = fsm.decide(risk(RiskLevel.HIGH, 0.5, ttc=1.5, intersection=True), ego(0.0), 1.0)
    assert d.state == BehaviorState.STOPPED
    # stationary: never re-enters EMERGENCY_BRAKE, planner may search at caution speed
    d = fsm.decide(risk(RiskLevel.HIGH, 0.9, ttc=0.5, intersection=True), ego(0.0), 1.5)
    assert d.state == BehaviorState.STOPPED and not d.speed_policy.force_stop
    assert d.speed_policy.allow_lateral_avoidance
    d = fsm.decide(risk(RiskLevel.NONE, 0.0), ego(0.0), 2.0)
    assert d.state == BehaviorState.CAUTION
    d = fsm.decide(risk(RiskLevel.NONE, 0.0), ego(0.0), 3.0)
    assert d.state == BehaviorState.CRUISE


def test_stopped_moves_off_into_avoid_when_planner_finds_a_path(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    fsm.decide(risk(RiskLevel.CRITICAL, 0.9, ttc=0.5, intersection=True), ego(), 0.0)
    fsm.decide(risk(RiskLevel.HIGH, 0.5, ttc=1.5, intersection=True), ego(0.0), 1.0)
    assert fsm.state == BehaviorState.STOPPED
    d = fsm.decide(risk(RiskLevel.HIGH, 0.5, ttc=1.5, intersection=True), ego(1.0), 2.0)
    assert d.state == BehaviorState.AVOID


def test_no_feasible_plan_forces_emergency_only_while_moving(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    d = fsm.decide(RiskSummary.empty(), ego(), 0.0, planner_feasible=False)
    assert d.state == BehaviorState.EMERGENCY_BRAKE and "no feasible" in d.reason
    fsm2 = BehaviorStateMachine(cfg, 10.0)
    d = fsm2.decide(RiskSummary.empty(), ego(0.0), 0.0, planner_feasible=False)
    assert d.state == BehaviorState.CRUISE


def test_follow_policy_matches_lead(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    r = risk(RiskLevel.NONE, 0.0, lead="car", lead_speed=6.0, lead_gap=9.0)
    d = fsm.decide(r, ego(), 0.0)
    assert d.state == BehaviorState.FOLLOW
    desired_gap = cfg.follow_time_gap_s * 6.0
    expected = 6.0 + (9.0 - desired_gap) / cfg.follow_time_gap_s
    assert d.speed_policy.target_speed == pytest.approx(min(10.0, max(0.0, expected)))


def test_every_decision_is_explained(cfg):
    fsm = BehaviorStateMachine(cfg, 10.0)
    for r, t in ((RiskSummary.empty(), 0.0), (risk(RiskLevel.MEDIUM, 0.3), 0.1),
                 (risk(RiskLevel.HIGH, 0.5, 2.0, True), 0.2)):
        d = fsm.decide(r, ego(), t)
        assert d.reason and isinstance(d.triggers, dict) and "max_risk_level" in d.triggers
        assert d.to_dict()["state"] == d.state.value
