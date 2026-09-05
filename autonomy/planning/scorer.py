"""Weighted cost function for feasible candidate trajectories.

    J = sum_i w_i * C_i      with every C_i normalised to roughly [0, 1]

C_collision   soft collision risk: max over objects and time of
              w_type * p(t) * exp(-t/tau), with p(t) the band collision
              probability (autonomy/core/probability.py) of the object's
              Gaussian prediction lying inside the ego collision band.
              Non-zero even for candidates that pass the hard collision
              check, so the planner prefers routes clear of uncertain predictions.
C_clearance   exp(-min(min_clearance - margin, saturation) / clearance_scale)  (1 at the margin,
              flat beyond margin + saturation so 'far enough' is not rewarded further)
C_smoothness  0.5 * mean|v^2 kappa| / a_lat_max + 0.5 * mean|a| / a_max
C_curvature   max|kappa| / kappa_max
C_progress    1 - (s_end - s0) / (target_speed * horizon)     (0 when target_speed == 0);
              its weight is multiplied by min(1 + gain * t_standstill, max_factor) while the
              ego stands still, so a feasible bypass eventually beats freezing
C_boundary    exp(-min_boundary_clearance / (0.5 * clearance_scale))
C_speed       |v_end - target_speed| / max(desired_speed, 1)
C_uncertainty mean over time of max over objects of p(t)   (exposure to uncertain regions)
C_lateral     |d_end - desired_offset| / max|lateral_offsets|   (return toward route)
C_consistency |d_end - d_end of the previously selected plan| / max|lateral_offsets|: with noisy
              perception the cheapest side can flip every cycle; committing beats dithering
C_blocked     1 - t_meet / route_lookahead if the continuation from the end state meets an
              object within the look-ahead (0 otherwise): postponing a head-on meeting by
              slowing down is not a solution; a clear continuation is

Weights come from PlanningConfig.weights. Individual components and weighted
components are stored on the candidate for explanation and telemetry.
"""
from __future__ import annotations

import math

import numpy as np

from autonomy.core.config import PlanningConfig
from autonomy.core.probability import band_collision_probability
from autonomy.core.types import CandidateTrajectory, SpeedPolicy, VehicleParameters

from .collision_checker import CheckResult


class TrajectoryScorer:
    def __init__(self, cfg: PlanningConfig, params: VehicleParameters, desired_speed: float,
                 risk_time_constant_s: float, safety_margin_m: float):
        self.cfg = cfg
        self.params = params
        self.desired_speed = max(desired_speed, 1.0)
        self.tau = risk_time_constant_s
        self.margin = safety_margin_m
        self.d_max = max(cfg.lateral_cost_scale_m, 0.5)

    def score(self, cand: CandidateTrajectory, check: CheckResult, policy: SpeedPolicy,
              desired_offset: float, s0: float, s_end: float, standstill_s: float = 0.0,
              previous_offset: float | None = None) -> float:
        tr = cand.trajectory
        rel_t = tr.t - tr.t[0]
        p = self.params
        c: dict[str, float] = {}

        if check.distances.size:
            prob = np.stack([
                band_collision_probability(check.distances[j], check.sigmas[j], self.margin,
                                           p.width + check.band_widths[j] + 2.0 * self.margin)
                for j in range(check.distances.shape[0])])                            # (M,N)
            weighted = prob * check.risk_weights[:, None] * np.exp(-rel_t / self.tau)[None, :]
            c["collision"] = float(min(1.0, np.max(weighted)))
            c["uncertainty"] = float(np.mean(np.max(prob, axis=0)))
        else:
            c["collision"] = 0.0
            c["uncertainty"] = 0.0

        gap_c = min(max(cand.min_clearance - self.margin, 0.0), self.cfg.clearance_saturation_m)
        c["clearance"] = float(math.exp(-gap_c / self.cfg.clearance_scale_m)) \
            if math.isfinite(cand.min_clearance) else 0.0
        a_lat = np.abs(tr.velocity ** 2 * tr.curvature)
        a_max = max(p.max_acceleration, p.max_deceleration)
        c["smoothness"] = float(0.5 * np.mean(a_lat) / self.cfg.max_lateral_acceleration_mps2
                                + 0.5 * np.mean(np.abs(tr.acceleration)) / a_max)
        c["curvature"] = float(np.max(np.abs(tr.curvature)) / p.max_curvature)
        max_progress = policy.target_speed * self.cfg.horizon_s
        c["progress"] = float(np.clip(1.0 - (s_end - s0) / max_progress, 0.0, 1.0)) if max_progress > 0 else 0.0
        c["boundary"] = float(math.exp(-cand.min_boundary_clearance / (0.5 * self.cfg.clearance_scale_m))) \
            if math.isfinite(cand.min_boundary_clearance) else 0.0
        c["speed"] = float(abs(cand.target_speed - policy.target_speed) / self.desired_speed)
        c["lateral"] = float(abs(cand.lateral_offset_end - desired_offset) / self.d_max)
        c["consistency"] = 0.0 if previous_offset is None else float(
            abs(cand.lateral_offset_end - previous_offset) / self.d_max)
        c["blocked"] = 0.0 if cand.route_block_time is None else \
            float(np.clip(1.0 - cand.route_block_time / self.cfg.route_lookahead_s, 0.0, 1.0))

        w = dict(self.cfg.weights.as_dict())
        if standstill_s > 0.0:
            w["progress"] *= min(1.0 + self.cfg.standstill_progress_gain * standstill_s,
                                 self.cfg.standstill_progress_max_factor)
        cand.costs = c
        cand.weighted_costs = {k: w[k] * c[k] for k in c}
        cand.total_cost = float(sum(cand.weighted_costs.values()))
        return cand.total_cost
