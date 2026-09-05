"""Feasibility checks for candidate trajectories.

Hard rejections, in this order, each recorded with a reason:

    CURVATURE_LIMIT             |kappa| > tan(delta_max) / L
    LATERAL_ACCELERATION_LIMIT  v^2 |kappa| > a_lat_max
    OUTSIDE_DRIVABLE_SPACE      any footprint corner outside the corridor at any time, or
                                closer than boundary_margin after boundary_margin_grace_s
    PREDICTED_COLLISION         footprint distance to any predicted object <= safety_margin
                                at the same time index
    TERMINAL_STATE_EXPOSED      continuing from the candidate's END state (along the corridor at the
                                terminal speed; frozen in place for a stop) collides with a
                                constant-velocity extrapolation of an object before
                                terminal_exposure_horizon_s. This rejects both "stop where you will
                                be hit" and "creep toward a blocked route" candidates, so a bypass
                                is preferred over myopic waiting. Candidates rejected only for this
                                remain scorable (degraded tier) so the planner never returns nothing.
    Within the first `collision_margin_grace_s` a candidate is rejected only for actual overlap,
    not for being inside the margin: the current pose is not the planner's choice. Beyond it,
    being inside the margin is a hit only while the distance is still shrinking relative to now
    (sliding past at the clearance the ego already has is allowed; actual overlap never is).

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

        grace = rel_t < cfg.collision_margin_grace_s          # the current state is not the planner's choice

        # 1. curvature / steering limit (hard steering saturation is never excused)
        bad = K > p.max_curvature * (1.0 + 1e-6)
        for i in np.nonzero(bad.any(axis=1))[0]:
            k = int(np.argmax(K[i]))
            self._reject(cands[i], RejectionReason.CURVATURE,
                         f"|kappa|={K[i, k]:.3f} > {p.max_curvature:.3f} 1/m at +{rel_t[i, k]:.1f}s")
            alive[i] = False

        # 2. lateral acceleration (comfort limit; excused during the grace window, the ego is already turning)
        a_lat = V ** 2 * K
        bad = (a_lat > cfg.max_lateral_acceleration_mps2) & alive[:, None] & ~grace
        for i in np.nonzero(bad.any(axis=1))[0]:
            k = int(np.argmax(np.where(~grace[i], a_lat[i], -1.0)))
            self._reject(cands[i], RejectionReason.LATERAL_ACCEL,
                         f"v^2*kappa={a_lat[i, k]:.2f} > {cfg.max_lateral_acceleration_mps2:.2f} m/s^2 at +{rel_t[i, k]:.1f}s")
            alive[i] = False

        # 3. drivable-space boundary (all footprints at once)
        off = p.footprint_center_offset
        FX = (X + off * np.cos(YAW)).ravel()
        FY = (Y + off * np.sin(YAW)).ravel()
        FYAW = YAW.ravel()
        if all(c.trajectory.s is not None for c in cands):
            clearance = self._corridor_clearance(cands, C, N)             # fast path, corridor frame
        else:
            corners = box_corners(FX, FY, FYAW, p.length, p.width)      # (C*N,4,2)
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
                # directional sigma: interpolate the covariance onto the candidate grid, then project
                Pxx = np.interp(T, pred.times, pred.covariances[:, 0, 0]).ravel()
                Pyy = np.interp(T, pred.times, pred.covariances[:, 1, 1]).ravel()
                Pxy = np.interp(T, pred.times, pred.covariances[:, 0, 1]).ravel()
                ux, uy = FX - ox, FY - oy
                nrm = np.hypot(ux, uy)
                nrm = np.where(nrm > 1e-9, nrm, 1.0)
                ux, uy = ux / nrm, uy / nrm
                sig[j] = np.sqrt(np.maximum(Pxx * ux * ux + 2 * Pxy * ux * uy + Pyy * uy * uy, 1e-12)).reshape(C, N)
            # inside the margin counts as a hit, except during the grace window where only overlap does
            margin_ok = rel_t >= cfg.collision_margin_grace_s                      # (C,N)
            margins = np.array([cfg.safety_margin_m + min(cfg.uncertainty_margin_gain * pr.meas_sigma, cfg.uncertainty_margin_max_m)
                                for pr in predictions])
            # inside-margin counts as a hit only when CLOSING relative to the current distance (or overlapping):
            # a candidate that slides past an object at the clearance the ego already has is not a collision
            d0 = dist[:, :, :1]                                                    # (M,C,1) current distance
            closing = dist < d0 - 0.05
            hit = (dist <= 0.0) | ((dist <= margins[:, None, None]) & closing & margin_ok[None, :, :])   # (M,C,N)
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
                                 f"distance {dist[j, i, k]:.2f} m <= margin {margins[j]:.2f} m "
                                 f"to {predictions[j].object_id} at +{rel_t[i, k]:.1f}s")
                    alive[i] = False

            # 4b. standoff for near-stop candidates: do not come to rest closer than stop_standoff_m to any
            #     predicted object (room to manoeuvre later; no reversing in Stage 1). Scorable degraded tier.
            # measured against each object's predicted position at the end AND its current position: a crossing
            # animal may stop where it is, so the ego must not come to rest right at its current spot either
            if predictions:
                infl = np.array([min(pr.meas_sigma, cfg.exposure_sigma_cap_m) for pr in predictions])   # (M,)
                ex0 = FX.reshape(C, N)[:, -1]; ey0 = FY.reshape(C, N)[:, -1]; eyaw0 = YAW[:, -1]
                d_now = np.stack([box_sequence_distance(ex0, ey0, eyaw0, p.length, p.width,
                                                        np.full(C, pr.x[0]), np.full(C, pr.y[0]), np.full(C, pr.heading[0]),
                                                        pr.length + 2 * infl[j], pr.width + 2 * infl[j])
                                  for j, pr in enumerate(predictions)])                                 # (M,C)
            for i in range(C):
                if alive[i] and cands[i].target_speed < cfg.stop_standoff_speed_mps and predictions:
                    d_end_all = np.minimum(dist[:, i, -1], d_now[:, i])
                    j = int(np.argmin(d_end_all))
                    if d_end_all[j] < cfg.stop_standoff_m:
                        self._reject(cands[i], RejectionReason.TERMINAL_EXPOSURE,
                                     f"would come to rest {d_end_all[j]:.1f} m from {predictions[j].object_id} "
                                     f"(< standoff {cfg.stop_standoff_m:.1f} m)")
                        cands[i].margin_only = True
                        alive[i] = False

            # 5. terminal exposure: continue each surviving candidate beyond the horizon along the corridor
            #    at its terminal speed (frozen for a stop) and extrapolate objects at constant velocity
            T_end = float(rel_t[0, -1])
            if cfg.route_lookahead_s > T_end + 1e-9 and alive.any():
                ext_dt = 2.0 * cfg.dt_s                                                       # coarser beyond the horizon
                ext = np.arange(T_end + ext_dt, cfg.route_lookahead_s + 1e-9, ext_dt)               # (E,)
                idx = np.nonzero(alive)[0]
                E = len(ext)
                if all(cands[i].trajectory.s is not None for i in idx):
                    s_end = np.array([cands[i].trajectory.s[-1] for i in idx])
                    d_end = np.array([cands[i].trajectory.d[-1] for i in idx])
                    v_end = np.array([cands[i].trajectory.velocity[-1] for i in idx])
                    s_ext = (s_end[:, None] + v_end[:, None] * (ext - T_end)[None, :]).ravel()
                    d_ext = np.repeat(d_end, E)
                    cx, cy, ch = self.road.to_cartesian(s_ext, d_ext)
                    ex = cx + off * np.cos(ch)
                    ey = cy + off * np.sin(ch)
                    eyaw = ch
                else:
                    ex = np.repeat(FX.reshape(C, N)[idx, -1], E)
                    ey = np.repeat(FY.reshape(C, N)[idx, -1], E)
                    eyaw = np.repeat(YAW[idx, -1], E)
                t_abs = (T[idx, 0][:, None] + ext[None, :]).ravel()
                for j, pred in enumerate(predictions):
                    dt_ext = t_abs - pred.times[-1]
                    ox = pred.x[-1] + pred.vx[-1] * dt_ext
                    oy = pred.y[-1] + pred.vy[-1] * dt_ext
                    oh = np.full_like(ox, pred.heading[-1])
                    dist_ext = box_sequence_distance(ex, ey, eyaw, p.length, p.width,
                                                     ox, oy, oh, pred.length + 2 * infl[j], pred.width + 2 * infl[j]
                                                     ).reshape(len(idx), E)
                    hit_ext = dist_ext <= margins[j]
                    for r, i in enumerate(idx):
                        c = cands[i]
                        if not hit_ext[r].any():
                            continue
                        k = int(np.argmax(hit_ext[r]))
                        t_meet = float(ext[k])
                        if c.route_block_time is None or t_meet < c.route_block_time:
                            c.route_block_time = t_meet
                        if not alive[i]:
                            continue
                        if t_meet <= cfg.terminal_exposure_horizon_s:
                            self._reject(c, RejectionReason.TERMINAL_EXPOSURE,
                                         f"continuing at {c.target_speed:.1f} m/s from the end state meets "
                                         f"{pred.object_id} at +{t_meet:.1f}s")
                            c.margin_only = True          # scorable in the degraded tier
                            alive[i] = False
                        elif c.target_speed < cfg.stop_standoff_speed_mps and dist_ext[r, 0] < cfg.stop_standoff_m:
                            # stopping behind a blocked route: keep room to manoeuvre around the blocker later
                            self._reject(c, RejectionReason.TERMINAL_EXPOSURE,
                                         f"would stop {dist_ext[r, 0]:.1f} m behind {pred.object_id} "
                                         f"(< standoff {cfg.stop_standoff_m:.1f} m) with the route blocked")
                            c.margin_only = True
                            alive[i] = False
        return results

    def _corridor_clearance(self, cands: list[CandidateTrajectory], C: int, N: int) -> np.ndarray:
        """Boundary clearance from corridor coordinates.

        Each footprint corner's lateral offset is d_centre +- (W/2 cos th) +- (L/2 sin th)
        (th = heading relative to the reference); clearance is its distance to the
        corridor edge at that station. Exact on straight references; on a curve of
        radius R the error is of order (L/2)^2 / (2R): measured 6 cm at R = 60 m,
        well inside the 25 cm boundary margin.
        """
        p = self.params
        S = np.stack([c.trajectory.s for c in cands]).ravel()
        D = np.stack([c.trajectory.d for c in cands]).ravel()
        TH = np.stack([c.trajectory.heading_rel for c in cands]).ravel()
        off = p.footprint_center_offset
        sc = S + off * np.cos(TH)
        dc = D + off * np.sin(TH)
        half_w = 0.5 * p.width * np.abs(np.cos(TH)) + 0.5 * p.length * np.abs(np.sin(TH))
        d_right, d_left = self.road.lateral_bounds(sc)
        clear_left = d_left - (dc + half_w)
        clear_right = (dc - half_w) - d_right
        clearance = np.minimum(clear_left, clear_right)
        return np.where(clearance > 0.0, clearance, 0.0).reshape(C, N)

    @staticmethod
    def _reject(cand: CandidateTrajectory, reason: RejectionReason, detail: str) -> None:
        cand.feasible = False
        cand.rejection_reason = reason
        cand.rejection_detail = detail
        cand.total_cost = math.inf
