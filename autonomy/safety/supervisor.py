"""Safety supervisor: independent emergency-braking override.

Runs every dynamics step, after the tracker and before the vehicle model.
It does not depend on the planner's choices; it only looks at the risk
summary, the ego state and whether the planner had to fall back.

Activation when ANY of:
    min_ttc_current_speed < safety.ttc_critical_s   (physical TTC at current speed)
    max risk level     == CRITICAL
    planner fallback   (no feasible trajectory)

While active: acceleration = 0, brake = safety.brake_mps2, steering kept from
the nominal command (steering while braking is allowed). The override holds
for at least `hold_time_s` after the last trigger so it does not chatter.
Every activation is recorded with time and reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from autonomy.core.config import SafetyConfig
from autonomy.core.types import ControlCommand, PlannerOutput, RiskLevel, RiskSummary, SafetyStatus, VehicleState


@dataclass
class Activation:
    time: float
    reason: str
    released_at: Optional[float] = None


class SafetySupervisor:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.active = False
        self.last_trigger_time = -math.inf
        self.activations: list[Activation] = []
        self.current_reason = ""

    def _trigger(self, risk: RiskSummary, plan: Optional[PlannerOutput], ego: VehicleState) -> Optional[str]:
        if plan is not None and plan.selected.fallback and ego.longitudinal_velocity > 0.05:
            return "planner returned fallback stop (no feasible trajectory)"
        if risk.min_ttc_current_speed < self.cfg.ttc_critical_s:
            return (f"physical TTC {risk.min_ttc_current_speed:.2f} s < {self.cfg.ttc_critical_s:.2f} s "
                    f"to {risk.worst_object_id}")
        if risk.max_level == RiskLevel.CRITICAL and ego.longitudinal_velocity > 0.05:
            return f"critical risk level from {risk.worst_object_id}"
        return None

    def check(self, command: ControlCommand, risk: RiskSummary, plan: Optional[PlannerOutput],
              ego: VehicleState, now: float) -> tuple[ControlCommand, SafetyStatus]:
        reason = self._trigger(risk, plan, ego)
        if reason:
            if not self.active:
                self.activations.append(Activation(now, reason))
                self.active = True
            self.current_reason = reason
            self.last_trigger_time = now
        elif self.active and now - self.last_trigger_time >= self.cfg.hold_time_s:
            self.active = False
            self.activations[-1].released_at = now
            self.current_reason = ""

        if self.active:
            command = ControlCommand(now, command.steering_angle, 0.0, self.cfg.brake_mps2, source="safety")
        status = SafetyStatus(self.active, self.current_reason, len(self.activations),
                              self.activations[-1].time if self.active else None)
        return command, status
