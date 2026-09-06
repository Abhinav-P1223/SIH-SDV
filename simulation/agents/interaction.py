"""Ego-aware traffic interaction: IDM following plus APPROACH / YIELD / COMMIT.

Deliberately compact. Two textbook mechanisms, no framework:

  IDM             longitudinal following. A traffic agent with the ego (or anything else) ahead
                  in its own path closes the gap to a desired headway and brakes when it shrinks.
  Gap acceptance  a three-state decision for crossing or merging conflicts. The agent estimates
                  when each party reaches the conflict point, compares the time gap against a
                  critical gap, and either yields or commits. Commitment is one-way, which is what
                  human drivers actually do and what stops the state flapping.

Nothing here is scripted against a clock or a scenario. The agent reasons only from the ego state
it is given (position, velocity, heading), which is what a real road user can see. It never sees
the ego's future control commands, and the ego never sees the agent's internal state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple


class EgoView(NamedTuple):
    """What a traffic participant can observe about the ego.

    Indexable as (x, y) so every pre-existing behaviour that reads `ego_xy[0]` / `ego_xy[1]`
    keeps working unchanged -- that is the whole reason this is a NamedTuple and x, y come first.
    """
    x: float
    y: float
    speed: float = 0.0
    heading: float = 0.0

    @property
    def vx(self) -> float:
        return self.speed * math.cos(self.heading)

    @property
    def vy(self) -> float:
        return self.speed * math.sin(self.heading)


def idm_acceleration(v: float, v0: float, gap: float, dv: float, *, a_max: float, b_comf: float,
                     s0: float = 2.0, headway_s: float = 1.5) -> float:
    """Intelligent Driver Model acceleration.

        s* = s0 + max(0, v*T + v*dv / (2 sqrt(a b)))
        a  = a_max [ 1 - (v/v0)^4 - (s*/s)^2 ]

    `gap` is the clear distance to the obstruction ahead and `dv = v - v_lead` is the closing
    speed (positive when approaching). With no obstruction pass gap = inf, which reduces to free
    acceleration toward v0. The result is clamped to [-b_max, a_max]; `s` is floored so a zero gap
    cannot produce an infinite deceleration.
    """
    v = max(v, 0.0)
    v0 = max(v0, 1e-3)
    free = 1.0 - (v / v0) ** 4
    if not math.isfinite(gap):
        acc = a_max * free
    else:
        s = max(gap, 0.1)
        s_star = s0 + max(0.0, v * headway_s + v * dv / (2.0 * math.sqrt(a_max * b_comf)))
        acc = a_max * (free - (s_star / s) ** 2)
    return max(-b_comf * 3.0, min(a_max, acc))


APPROACH, YIELD, COMMIT = "APPROACH", "YIELD", "COMMIT"


@dataclass
class GapAcceptance:
    """Three-state conflict decision with one-way commitment.

    critical_gap_s   the time gap the agent insists on before crossing in front of the ego
    hysteresis_s     extra gap required to LEAVE the yield state, so a gap hovering at the
                     threshold does not flip the decision every cycle
    reaction_time_s  the agent acts on an ego state this old, so it cannot respond instantly
    """
    critical_gap_s: float = 2.5
    hysteresis_s: float = 0.8
    reaction_time_s: float = 0.5
    state: str = APPROACH
    last_gap_s: float = math.inf
    _delayed: list = field(default_factory=list)      # [(t, EgoView)] ring for the reaction delay

    def perceive(self, ego: EgoView, now: float) -> EgoView:
        """The ego as this agent currently sees it: delayed by its reaction time."""
        self._delayed.append((now, ego))
        cutoff = now - self.reaction_time_s
        while len(self._delayed) > 1 and self._delayed[1][0] <= cutoff:
            self._delayed.pop(0)
        return self._delayed[0][1]

    def decide(self, t_agent: float, t_ego: float) -> str:
        """Update the state from the two arrival times at the conflict point.

        `t_agent` and `t_ego` are seconds until each party reaches the conflict. The agent wants
        to be clear of it by `critical_gap_s` before the ego arrives.
        """
        gap = t_ego - t_agent
        self.last_gap_s = gap
        if self.state == COMMIT:
            return COMMIT                                  # one-way: committed is committed
        if not math.isfinite(t_ego):
            self.state = COMMIT                            # nothing coming: go
        elif self.state == YIELD:
            # Already yielding: only a gap that clears critical AND the hysteresis band releases
            # it. Without this the state drops back to APPROACH the moment the gap nudges above
            # critical, and the decision flaps from cycle to cycle on a gap that is barely moving.
            if gap >= self.critical_gap_s + self.hysteresis_s:
                self.state = COMMIT
        else:                                              # APPROACH
            self.state = COMMIT if gap >= self.critical_gap_s else YIELD
        return self.state

    def release(self) -> None:
        """The conflict is behind us: allow a fresh decision for the next one."""
        self.state = APPROACH
        self.last_gap_s = math.inf


def time_to_point(x: float, y: float, speed: float, heading: float,
                  px: float, py: float) -> float:
    """Seconds until a point-mass at (x, y) travelling at `speed` along `heading` reaches (px, py).

    Measured along the direction of travel, so a conflict already behind the mover returns a
    negative time and one it is not approaching returns infinity.
    """
    if speed <= 1e-3:
        return math.inf
    along = (px - x) * math.cos(heading) + (py - y) * math.sin(heading)
    return along / speed
