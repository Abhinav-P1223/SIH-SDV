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
        self.previous_offset: float | None = None

    def plan(self, ego: VehicleState, decision: BehaviorDecision,
             predictions: list[ObjectPrediction], now: float) -> PlannerOutput:
        t0 = time.perf_counter()
        policy = decision.speed_policy
        if ego.longitudinal_velocity < self.cfg.standstill_speed_mps and not policy.force_stop:
            self.standstill_since = now if self.standstill_since is None else self.standstill_since
        else:
            self.standstill_since = None
        standstill = 0.0 if self.standstill_since is None else now - self.standstill_since
        candidates, frame = self.generator.generate(ego, policy, self.desired_offset, now)

        feasible: list[CandidateTrajectory] = []
        squeezed: list[CandidateTrajectory] = []
        checks = self.checker.check_all(candidates, predictions)
        for cand, check in zip(candidates, checks):
            if cand.feasible or cand.margin_only:
                s_end, _, _ = self.road.project(float(cand.trajectory.x[-1]), float(cand.trajectory.y[-1]))
                self.scorer.score(cand, check, policy, self.desired_offset, frame.s0, s_end, standstill,
                                  self.previous_offset)
                (feasible if cand.feasible else squeezed).append(cand)

        if feasible:
            selected = min(feasible, key=lambda c: c.total_cost)
        else:
            if squeezed:
                selected = min(squeezed, key=lambda c: c.total_cost)
                selected.degraded = True
            else:
                hard = [c for c in candidates if c.label.startswith("stop_hard")]
                selected = hard[-1] if hard else candidates[-1]
                selected.fallback = True

        self.previous_offset = None if selected.fallback else selected.lateral_offset_end
        hist = Counter(c.rejection_reason.value for c in candidates if not c.feasible)
        out = PlannerOutput(
            timestamp=now, candidates=candidates, selected=selected,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            feasible_count=len(feasible) if feasible else (1 if selected.degraded else 0),
            rejected_count=len(candidates) - len(feasible),
            rejection_histogram=dict(hist),
            frame_origin=(frame.s0, frame.d0, frame.heading_rel),
            standstill_s=standstill,
        )
        self.last_output = out
        return out
