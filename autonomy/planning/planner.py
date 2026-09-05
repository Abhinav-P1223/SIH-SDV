"""Planner facade: generate -> check -> score -> select.

Replaceable as a unit: anything implementing `plan(ego, decision, predictions,
now)` and returning a `PlannerOutput` can stand in for it (lattice, MPC, ...).

Selection hierarchy when nothing is fully feasible:
1. candidates that are collision-free within the horizon and on the road but
   violate only the boundary *margin*, or whose END pose is exposed to a
   constant-velocity object beyond the horizon, are scored with the normal
   cost function and the cheapest is selected, flagged `degraded=True`;
2. otherwise the hard-stop candidate flagged `fallback=True`. The behaviour
   layer and the safety supervisor treat that flag as an emergency.
"""
from __future__ import annotations

import math
import time
from collections import Counter

from autonomy.core.config import PlanningConfig, RiskConfig
from autonomy.core.interfaces import RoadModel
from autonomy.core.types import (BehaviorDecision, CandidateTrajectory, ObjectPrediction,
                                 PlannerOutput, VehicleParameters, VehicleState)

from .candidate_generator import CandidateGenerator
from .collision_checker import CollisionChecker
from .scorer import TrajectoryScorer


class Planner:
    def __init__(self, cfg: PlanningConfig, risk_cfg: RiskConfig, params: VehicleParameters,
                 road: RoadModel, desired_speed: float, desired_lateral_offset: float):
        self.cfg = cfg
        self.params = params
        self.road = road
        self.desired_offset = desired_lateral_offset
        self.generator = CandidateGenerator(cfg, params, road)
        self.checker = CollisionChecker(cfg, params, road)
        self.scorer = TrajectoryScorer(cfg, params, desired_speed, risk_cfg.time_constant_s,
                                       cfg.safety_margin_m)
        self.last_output: PlannerOutput | None = None
        self.standstill_since: float | None = None
        self.reverse_target_s: float | None = None      # committed end of the current reversing leg
        self.reverse_leg_done = False                   # leg finished; do not start another until the
                                                        # behaviour layer leaves REVERSING
        self.previous_offset: float | None = None

    @staticmethod
    def _advance(c: CandidateTrajectory) -> float:
        tr = c.trajectory
        if tr.s is not None:
            return float(tr.s[-1] - tr.s[0])
        return math.hypot(float(tr.x[-1] - tr.x[0]), float(tr.y[-1] - tr.y[0]))

    @staticmethod
    def _leg_distance(c: CandidateTrajectory) -> float:
        """Requested length of a reversing leg (encoded in the id: reverse_<D>m)."""
        return float(c.id[len("reverse_"):-1])

    def _boxed_in(self, progressing: list[CandidateTrajectory], predictions: list[ObjectPrediction],
                  check, s0: float) -> bool:
        """True when the ego is stuck right up against a static blocker: it is close enough that the lateral
        transition no longer fits in front of it, no candidate aims at an offset that clears it, and yet there
        IS room to pass beside it. Reversing then buys the run-up the shift needs. Behind a queue of stopped
        traffic no such gap exists and the ego simply waits, so this stays False."""
        if not progressing or check is None or not predictions or check.obj_s.size == 0:
            return False
        ahead = [j for j in range(len(predictions)) if check.obj_s[j] > s0
                 and math.hypot(float(predictions[j].vx[0]), float(predictions[j].vy[0])) < 0.3]
        if not ahead:
            return False
        j = min(ahead, key=lambda k: check.obj_s[k])
        pr = predictions[j]
        half_obj = 0.5 * max(pr.length, pr.width)
        gap_to_blocker = float(check.obj_s[j]) - s0 - half_obj - 0.5 * self.params.length
        if gap_to_blocker > self.cfg.boxed_in_range_m:
            return False                                      # not up against it: just drive
        # Judge each candidate by the offset it AIMS at, not the one it has reached: a wide shift needs about
        # 9 m of travel and therefore outlasts the 4 s horizon, so the achieved offset always looks blocked.
        half_ego = 0.5 * self.params.width + self.cfg.safety_margin_m
        if any(abs(float(check.obj_d[j]) - c.lateral_offset_end) >= half_obj + half_ego for c in progressing):
            return False                                      # something genuinely aims past the blocker
        d_right, d_left = self.road.lateral_bounds_at(float(check.obj_s[j])) if hasattr(self.road, "lateral_bounds_at") \
            else (-math.inf, math.inf)
        need = self.params.width + self.cfg.safety_margin_m + self.cfg.boundary_margin_m   # object side + edge side
        gap_left = d_left - (float(check.obj_d[j]) + half_obj)
        gap_right = (float(check.obj_d[j]) - half_obj) - d_right
        return max(gap_left, gap_right) >= need

    def plan(self, ego: VehicleState, decision: BehaviorDecision,
             predictions: list[ObjectPrediction], now: float, tracking_error: float = 0.0) -> PlannerOutput:
        t0 = time.perf_counter()
        policy = decision.speed_policy
        if ego.longitudinal_velocity < self.cfg.standstill_speed_mps and not policy.force_stop:
            self.standstill_since = now if self.standstill_since is None else self.standstill_since
        else:
            self.standstill_since = None
        standstill = 0.0 if self.standstill_since is None else now - self.standstill_since
        if not policy.allow_reverse:                  # behaviour layer is not in REVERSING: episode over
            self.reverse_target_s, self.reverse_leg_done = None, False
        rev_dists = None
        rolling_back = ego.longitudinal_velocity < -0.3
        if policy.allow_reverse and self.reverse_target_s is not None:
            s_now, _, _ = self.road.project(ego.x, ego.y)
            remaining = s_now - self.reverse_target_s
            if rolling_back:
                rev_dists = [max(remaining, 0.0)]                       # finish the committed leg
            elif remaining <= self.cfg.progress_feasible_min_m:
                self.reverse_leg_done = True                            # arrived (or stopped short): re-evaluate
        if policy.allow_reverse and self.reverse_leg_done and not rolling_back:
            rev_dists = []                                              # no new leg this episode
        candidates, frame = self.generator.generate(ego, policy, self.desired_offset, now, reverse_distances=rev_dists)

        feasible: list[CandidateTrajectory] = []
        squeezed: list[CandidateTrajectory] = []
        extra = (self.cfg.tracking_margin_speed_gain_s * max(ego.longitudinal_velocity, 0.0)
                 + min(abs(tracking_error), self.cfg.tracking_margin_error_cap_m))
        checks = self.checker.check_all(candidates, predictions, extra_margin=extra)
        for cand, check in zip(candidates, checks):
            if cand.feasible or cand.margin_only:
                s_end, _, _ = self.road.project(float(cand.trajectory.x[-1]), float(cand.trajectory.y[-1]))
                self.scorer.score(cand, check, policy, self.desired_offset, frame.s0, s_end, standstill,
                                  self.previous_offset)
                (feasible if cand.feasible else squeezed).append(cand)

        forward_feasible = [c for c in feasible if not c.id.startswith("reverse_")]
        reverse_feasible = [c for c in feasible if c.id.startswith("reverse_")]
        at_rest = abs(ego.longitudinal_velocity) < self.cfg.standstill_speed_mps
        progressing = (lambda c: not at_rest or self._advance(c) >= self.cfg.progress_feasible_min_m)
        # boxed in: at rest, every progressing forward candidate only closes up on a static blocker that the
        # ego could pass beside if it had more room. Creeping up to the standoff is then not progress: hold
        # position instead so the behaviour layer sees "no forward plan" and can order a reversing leg.
        boxed_in = at_rest and policy.allow_lateral_avoidance and not self.reverse_leg_done and self._boxed_in(
            [c for c in forward_feasible if progressing(c)], predictions, checks[0] if checks else None, frame.s0)
        self.last_boxed_in = boxed_in
        if boxed_in:
            forward_feasible = [c for c in forward_feasible if not progressing(c)]
        forward_progressing = [c for c in forward_feasible if progressing(c)]
        degraded = False
        if rolling_back and reverse_feasible:                 # finish the committed leg (the stop is the fallback)
            selected = reverse_feasible[0]
            selected.total_cost = 0.0
        elif forward_progressing or (forward_feasible and not reverse_feasible):
            selected = min(forward_feasible, key=lambda c: c.total_cost)
        elif reverse_feasible:                                # boxed in: start a leg with the longest clear distance
            selected = max(reverse_feasible, key=lambda c: abs(float(c.trajectory.s[-1] - c.trajectory.s[0])))
            self.reverse_target_s = float(selected.trajectory.s[0]) - self._leg_distance(selected)
            selected.total_cost = 0.0
        else:
            if squeezed:
                degraded = True
                selected = min(squeezed, key=lambda c: c.total_cost)
            else:
                hard = [c for c in candidates if c.label.startswith("stop_hard")]
                selected = hard[-1] if hard else candidates[-1]
                selected.fallback = True
        selected.degraded = degraded

        # keep the last committed offset even when this cycle fell back to a hard stop: the ego is still
        # physically on that line, and forgetting it is what let the lattice flip sides every cycle
        if not selected.fallback:
            self.previous_offset = selected.lateral_offset_end
        if not selected.id.startswith("reverse_"):
            self.reverse_target_s = None
        hist = Counter(c.rejection_reason.value for c in candidates if not c.feasible)
        # what the behaviour layer is told: is there a forward plan that actually gets somewhere? At standstill a
        # candidate that merely holds position (a clear zero-speed candidate, or a degraded one whose standoff
        # violation is creep-guarded) is not a forward plan -> the ego is boxed in.
        if forward_feasible:
            n_feasible = sum(1 for c in forward_feasible if progressing(c))
        elif selected.degraded:
            n_feasible = 1 if progressing(selected) else 0
        else:
            n_feasible = 0
        out = PlannerOutput(
            timestamp=now, candidates=candidates, selected=selected,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            feasible_count=n_feasible,
            rejected_count=len(candidates) - len(feasible),
            rejection_histogram=dict(hist),
            frame_origin=(frame.s0, frame.d0, frame.heading_rel),
            standstill_s=standstill,
        )
        self.last_output = out
        return out
