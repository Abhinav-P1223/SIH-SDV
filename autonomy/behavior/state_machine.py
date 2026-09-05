"""Behaviour decision layer: a declared finite state machine.

States: CRUISE, FOLLOW, CAUTION, AVOID, EMERGENCY_BRAKE, STOPPED.

EMERGENCY_BRAKE is entered only while moving, on a CRITICAL physical TTC or
when the planner has no feasible trajectory. STOPPED is entered only from
EMERGENCY_BRAKE once the vehicle is stationary; in STOPPED the planner keeps
searching for a clear path at caution speed, and the machine leaves STOPPED
when the vehicle moves off or the risk subsides.

All transitions live in one table (`TRANSITIONS`) as (from-states, to-state,
guard). Guards are pure functions of a `DecisionContext` and return a reason
string when they fire, otherwise None. The table is evaluated in order; the
first firing transition wins. Escalations (to a more severe state) are
immediate; de-escalations respect `min_dwell_s` and lower exit thresholds so
the machine does not chatter. This structure maps 1:1 onto a Stateflow chart.

Every decision returns a machine-readable explanation: the reason string and
the numeric triggers that produced it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from autonomy.core.config import BehaviorConfig
from autonomy.core.types import (BehaviorDecision, BehaviorState, RiskLevel, RiskSummary,
                                 SpeedPolicy, VehicleState)

SEVERITY = {
    BehaviorState.CRUISE: 0,
    BehaviorState.FOLLOW: 1,
    BehaviorState.CAUTION: 2,
    BehaviorState.AVOID: 3,
    BehaviorState.STOPPED: 4,
    BehaviorState.EMERGENCY_BRAKE: 5,
}


@dataclass
class DecisionContext:
    risk: RiskSummary
    ego: VehicleState
    cfg: BehaviorConfig
    desired_speed: float
    planner_feasible: bool          # did the last planning cycle find any feasible trajectory
    time_in_state: float

    @property
    def worst(self):
        if not self.risk.assessments:
            return None
        return max(self.risk.assessments, key=lambda a: (a.risk_level.value, a.risk_score))

    def describe_worst(self) -> str:
        w = self.worst
        if w is None:
            return "no objects"
        ttc = f"{w.ttc:.1f} s" if math.isfinite(w.ttc) else "inf"
        return (f"{w.object_id} ({w.object_type.value}): TTC {ttc}, min predicted distance "
                f"{w.min_predicted_distance:.2f} m at +{w.time_of_min_distance:.1f} s, "
                f"risk {w.risk_score:.2f} [{w.risk_level.name}]")


Guard = Callable[[DecisionContext], Optional[str]]


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def g_emergency(c: DecisionContext) -> Optional[str]:
    moving = c.ego.longitudinal_velocity >= c.cfg.stopped_speed_mps
    if not c.planner_feasible and moving:
        return "Planner found no feasible collision-free trajectory; emergency braking."
    if c.risk.max_level == RiskLevel.CRITICAL and moving:
        return (f"Physical TTC {c.risk.min_ttc_current_speed:.2f} s at current speed "
                f"{c.ego.longitudinal_velocity:.1f} m/s with {c.risk.worst_object_id}; emergency braking.")
    return None


def g_stopped(c: DecisionContext) -> Optional[str]:
    if c.ego.longitudinal_velocity < c.cfg.stopped_speed_mps:
        return f"Vehicle brought to a halt; holding while the planner looks for a clear path. {c.describe_worst()}."
    return None


def _has_slower_lead(c: DecisionContext) -> bool:
    r = c.risk
    return r.lead_object_id is not None and r.lead_speed < c.desired_speed - 0.5


def g_avoid(c: DecisionContext) -> Optional[str]:
    r = c.risk
    # threats other than a lead vehicle we are following, or the current plan itself being on a
    # collision course (whichever object), escalate to AVOID
    if r.non_lead_any_intersection and r.non_lead_max_level.value >= RiskLevel.HIGH.value:
        w = c.worst
        return (f"Predicted trajectory of {w.object_id} ({w.object_type.value}) intersects ego "
                f"trajectory within {w.ttc:.1f} s (risk {w.risk_score:.2f}).")
    if r.plan_any_intersection and r.plan_max_level.value >= RiskLevel.HIGH.value:
        return (f"Selected trajectory intersects {r.worst_object_id} within {r.plan_min_ttc:.1f} s "
                f"(plan risk {r.plan_max_score:.2f}); re-planning under avoidance.")
    return None


def g_caution(c: DecisionContext) -> Optional[str]:
    r = c.risk
    if r.non_lead_max_level.value >= RiskLevel.MEDIUM.value:
        if r.non_lead_any_intersection:
            return (f"Route at desired speed intersects a predicted object in {r.min_ttc:.1f} s "
                    f"(beyond the avoidance horizon); reducing speed. {c.describe_worst()}.")
        return f"Elevated collision risk without a predicted intersection: {c.describe_worst()}."
    return None


def g_follow(c: DecisionContext) -> Optional[str]:
    r = c.risk
    if _has_slower_lead(c) and r.non_lead_max_level.value <= RiskLevel.LOW.value:
        return (f"Slower object {r.lead_object_id} ahead at {r.lead_gap:.1f} m travelling "
                f"{r.lead_speed:.1f} m/s; matching speed.")
    return None


def g_release_to_cruise(c: DecisionContext) -> Optional[str]:
    r = c.risk
    if _has_slower_lead(c):
        return None
    if r.non_lead_max_score < c.cfg.caution_exit_score and not r.non_lead_any_intersection \
            and r.non_lead_max_level.value <= RiskLevel.LOW.value:
        return f"Risk cleared (max risk {r.non_lead_max_score:.2f}); resuming cruise."
    return None


def g_avoid_release(c: DecisionContext) -> Optional[str]:
    r = c.risk
    if not r.non_lead_any_intersection and r.non_lead_max_score < c.cfg.avoid_exit_score \
            and not r.plan_any_intersection:
        return f"Selected trajectory clear of predicted objects (max risk {r.non_lead_max_score:.2f})."
    return None


def g_stopped_to_avoid(c: DecisionContext) -> Optional[str]:
    moving = c.ego.longitudinal_velocity >= c.cfg.stopped_speed_mps
    if moving and c.risk.any_intersection and c.risk.max_level.value >= RiskLevel.HIGH.value:
        return f"Planner found a clear trajectory around {c.risk.worst_object_id}; moving off under avoidance."
    return None


def g_stopped_release(c: DecisionContext) -> Optional[str]:
    moving = c.ego.longitudinal_velocity >= c.cfg.stopped_speed_mps
    if c.risk.max_level.value <= RiskLevel.LOW.value:
        return f"Risk cleared while stopped (max risk {c.risk.max_score:.2f}); resuming."
    if moving or (c.risk.max_level == RiskLevel.MEDIUM and not c.risk.any_intersection):
        return "Residual risk without an imminent threat; proceeding with caution."
    return None


def g_eb_release(c: DecisionContext) -> Optional[str]:
    if c.risk.max_level.value < RiskLevel.CRITICAL.value and c.planner_feasible:
        return f"Critical condition cleared (level {c.risk.max_level.name}); handing back to planner."
    return None


ALL = tuple(BehaviorState)
S = BehaviorState

# (from_states, to_state, guard). Order = priority.
TRANSITIONS: list[tuple[tuple[BehaviorState, ...], BehaviorState, Guard]] = [
    (tuple(s for s in ALL if s not in (S.EMERGENCY_BRAKE, S.STOPPED)), S.EMERGENCY_BRAKE, g_emergency),
    ((S.EMERGENCY_BRAKE,), S.STOPPED, g_stopped),
    ((S.EMERGENCY_BRAKE,), S.CAUTION, g_eb_release),
    ((S.STOPPED,), S.AVOID, g_stopped_to_avoid),
    ((S.STOPPED,), S.CAUTION, g_stopped_release),
    ((S.CRUISE, S.FOLLOW, S.CAUTION), S.AVOID, g_avoid),
    ((S.AVOID,), S.CAUTION, g_avoid_release),
    ((S.CRUISE, S.FOLLOW), S.CAUTION, g_caution),
    ((S.CRUISE, S.CAUTION), S.FOLLOW, g_follow),
    ((S.AVOID,), S.FOLLOW, g_follow),
    ((S.CAUTION, S.FOLLOW), S.CRUISE, g_release_to_cruise),
]


# --------------------------------------------------------------------------- #
class BehaviorStateMachine:
    def __init__(self, cfg: BehaviorConfig, desired_speed: float):
        self.cfg = cfg
        self.desired_speed = desired_speed
        self.state = BehaviorState.CRUISE
        self.entered_at = 0.0
        self.last_reason = "Initial state."
        self.transition_count = 0

    def decide(self, risk: RiskSummary, ego: VehicleState, now: float,
               planner_feasible: bool = True) -> BehaviorDecision:
        ctx = DecisionContext(risk, ego, self.cfg, self.desired_speed, planner_feasible,
                              now - self.entered_at)
        previous = self.state
        reason = self.last_reason
        for from_states, to_state, guard in TRANSITIONS:
            if self.state not in from_states or to_state == self.state:
                continue
            escalation = SEVERITY[to_state] > SEVERITY[self.state]
            if not escalation and ctx.time_in_state < self.cfg.min_dwell_s:
                continue
            fired = guard(ctx)
            if fired:
                self.state = to_state
                self.entered_at = now
                self.transition_count += 1
                reason = fired
                break
        else:
            if self.state == BehaviorState.CRUISE and risk.max_level.value <= RiskLevel.LOW.value:
                reason = "No significant risk; cruising toward goal."
        self.last_reason = reason
        w = ctx.worst
        triggers = {
            "max_risk_score": risk.max_score,
            "max_risk_level": risk.max_level.name,
            "min_ttc": None if math.isinf(risk.min_ttc) else risk.min_ttc,
            "min_predicted_distance": None if math.isinf(risk.min_predicted_distance) else risk.min_predicted_distance,
            "any_intersection": risk.any_intersection,
            "worst_object": w.object_id if w else None,
            "lead_object": risk.lead_object_id,
            "ego_speed": ego.longitudinal_velocity,
            "planner_feasible": planner_feasible,
        }
        return BehaviorDecision(now, self.state, previous, reason, triggers,
                                self._policy(self.state, risk), now - self.entered_at)

    def _policy(self, state: BehaviorState, risk: RiskSummary) -> SpeedPolicy:
        v = self.desired_speed
        c = self.cfg
        if state == BehaviorState.CRUISE:
            return SpeedPolicy(v, True, False)
        if state == BehaviorState.FOLLOW:
            desired_gap = max(c.follow_time_gap_s * max(risk.lead_speed, 0.0), 2.0)
            gap_err = risk.lead_gap - desired_gap if math.isfinite(risk.lead_gap) else 0.0
            target = risk.lead_speed + gap_err / c.follow_time_gap_s
            return SpeedPolicy(float(min(v, max(0.0, target))), True, False)
        if state == BehaviorState.CAUTION:
            return SpeedPolicy(v * c.caution_speed_factor, True, False)
        if state == BehaviorState.AVOID:
            return SpeedPolicy(v * c.avoid_speed_factor, True, False)
        if state == BehaviorState.STOPPED:
            # stationary but not in danger: let the planner search for a clear path at caution speed;
            # the zero-speed candidate remains available if nothing is clear
            return SpeedPolicy(v * c.caution_speed_factor, True, False)
        return SpeedPolicy(0.0, False, True)   # EMERGENCY_BRAKE
