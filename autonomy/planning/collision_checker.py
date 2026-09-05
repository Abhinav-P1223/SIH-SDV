"""Feasibility checks for candidate trajectories.

Hard rejections, in this order, each recorded with a reason:

    CURVATURE_LIMIT             |kappa| > tan(delta_max) / L
    LATERAL_ACCELERATION_LIMIT  v^2 |kappa| > a_lat_max
    OUTSIDE_DRIVABLE_SPACE      any footprint corner outside the corridor at any time, or
                                closer than boundary_margin after boundary_margin_grace_s
    PREDICTED_COLLISION         footprint distance to any predicted object <= safety_margin
                                at the same time index

All candidates share the same time grid, so the checker evaluates them in one
batched pass: every footprint of every candidate is stacked into a single
array and compared against each predicted object once. The per-object
distance and sigma series are returned so the scorer can compute soft risk
without recomputing geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from autonomy.core.config import PlanningConfig
from autonomy.core.geometry import box_corners, box_sequence_distance
from autonomy.core.interfaces import RoadModel
from autonomy.core.types import CandidateTrajectory, ObjectPrediction, RejectionReason, VehicleParameters


@dataclass
class CheckResult:
    distances: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # (M objects, N steps)
    sigmas: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))      # (M, N)
    risk_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))     # (M,)
    band_widths: np.ndarray = field(default_factory=lambda: np.zeros(0))      # (M,) max(object length, width)
    boundary_clearance: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (N,)


class CollisionChecker:
    def __init__(self, cfg: PlanningConfig, params: VehicleParameters, road: RoadModel):
        self.cfg = cfg
        self.params = params
        self.road = road

    def check(self, cand: CandidateTrajectory, predictions: list[ObjectPrediction]) -> CheckResult:
        return self.check_all([cand], predictions)[0]

    def check_all(self, cands: list[CandidateTrajectory],
                  predictions: list[ObjectPrediction]) -> list[CheckResult]:
        p = self.params
        cfg = self.cfg
        C = len(cands)
        if C == 0:
            return []
        N = len(cands[0].trajectory)
        T = np.stack([c.trajectory.t for c in cands])             # (C,N)
        X = np.stack([c.trajectory.x for c in cands])
        Y = np.stack([c.trajectory.y for c in cands])
        YAW = np.stack([c.trajectory.yaw for c in cands])
        V = np.stack([c.trajectory.velocity for c in cands])
        K = np.abs(np.stack([c.trajectory.curvature for c in cands]))
        rel_t = T - T[:, :1]
        results = [CheckResult() for _ in cands]
        alive = np.ones(C, dtype=bool)

        # 1. curvature / steering limit
        bad = K > p.max_curvature * (1.0 + 1e-6)
        for i in np.nonzero(bad.any(axis=1))[0]:
            k = int(np.argmax(K[i]))
            self._reject(cands[i], RejectionReason.CURVATURE,
                         f"|kappa|={K[i, k]:.3f} > {p.max_curvature:.3f} 1/m at +{rel_t[i, k]:.1f}s")
            alive[i] = False

        # 2. lateral acceleration
        a_lat = V ** 2 * K
        bad = (a_lat > cfg.max_lateral_acceleration_mps2) & alive[:, None]
        for i in np.nonzero(bad.any(axis=1))[0]:
            k = int(np.argmax(a_lat[i]))
            self._reject(cands[i], RejectionReason.LATERAL_ACCEL,
                         f"v^2*kappa={a_lat[i, k]:.2f} > {cfg.max_lateral_acceleration_mps2:.2f} m/s^2 at +{rel_t[i, k]:.1f}s")
            alive[i] = False

        # 3. drivable-space boundary (all footprints at once)
        off = p.footprint_center_offset
        FX = (X + off * np.cos(YAW)).ravel()
        FY = (Y + off * np.sin(YAW)).ravel()
        FYAW = YAW.ravel()
        corners = box_corners(FX, FY, FYAW, p.length, p.width)          # (C*N,4,2)
        clearance = self.road.boundary_clearance(corners).reshape(C, N)  # 0 when outside
        on_road = clearance > 0.0
        after_grace = rel_t >= cfg.boundary_margin_grace_s
        margin_ok = (clearance >= cfg.boundary_margin_m) | ~after_grace
        for i in range(C):
            results[i].boundary_clearance = clearance[i]
            cands[i].min_boundary_clearance = float(clearance[i].min())
            if not alive[i]:
                continue
            if not on_road[i].all():
                k = int(np.argmin(on_road[i]))
                self._reject(cands[i], RejectionReason.BOUNDARY,
                             f"footprint leaves corridor at +{rel_t[i, k]:.1f}s")
                alive[i] = False
            elif not margin_ok[i].all():
                k = int(np.argmin(margin_ok[i]))
                self._reject(cands[i], RejectionReason.BOUNDARY,
                             f"boundary clearance {clearance[i, k]:.2f} m < margin {cfg.boundary_margin_m:.2f} m "
                             f"at +{rel_t[i, k]:.1f}s")
                cands[i].margin_only = True
                alive[i] = False

        # 4. predicted collisions
        if predictions:
            M = len(predictions)
            dist = np.empty((M, C, N))
            sig = np.empty((M, C, N))
            w = np.array([pr.risk_weight for pr in predictions])
            bw = np.array([max(pr.length, pr.width) for pr in predictions])
            for j, pred in enumerate(predictions):
                ox = np.interp(T, pred.times, pred.x).ravel()
                oy = np.interp(T, pred.times, pred.y).ravel()
                oh = np.interp(T, pred.times, np.unwrap(pred.heading)).ravel()
                dist[j] = box_sequence_distance(FX, FY, FYAW, p.length, p.width,
                                                ox, oy, oh, pred.length, pred.width).reshape(C, N)
                sig[j] = np.interp(T, pred.times, pred.sigma())
            hit = dist <= cfg.safety_margin_m                                   # (M,C,N)
            for i in range(C):
                results[i].distances = dist[:, i, :]
                results[i].sigmas = sig[:, i, :]
                results[i].risk_weights = w
                results[i].band_widths = bw
                cands[i].min_clearance = float(dist[:, i, :].min())
                if alive[i] and hit[:, i, :].any():
                    any_t = hit[:, i, :].any(axis=0)
                    k = int(np.argmax(any_t))
                    j = int(np.argmax(hit[:, i, k]))
                    cands[i].collision_time = float(rel_t[i, k])
                    self._reject(cands[i], RejectionReason.COLLISION,
                                 f"distance {dist[j, i, k]:.2f} m <= margin {cfg.safety_margin_m:.2f} m "
                                 f"to {predictions[j].object_id} at +{rel_t[i, k]:.1f}s")
                    alive[i] = False
        return results

    @staticmethod
    def _reject(cand: CandidateTrajectory, reason: RejectionReason, detail: str) -> None:
        cand.feasible = False
        cand.rejection_reason = reason
        cand.rejection_detail = detail
        cand.total_cost = math.inf
