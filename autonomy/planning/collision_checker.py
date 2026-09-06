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
    A collision rejection that never involves actual footprint overlap (only the margin is
    entered) keeps the candidate scorable in the planner's degraded tier: a tight pass is
    preferred over a fallback stop that would itself be struck.
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
    obj_s: np.ndarray = field(default_factory=lambda: np.zeros(0))        # (M,) object arc length now
    obj_d: np.ndarray = field(default_factory=lambda: np.zeros(0))        # (M,) object lateral offset now
    obj_v_lat: np.ndarray = field(default_factory=lambda: np.zeros(0))    # (M,) object lateral velocity (+left)


class CollisionChecker:
    def __init__(self, cfg: PlanningConfig, params: VehicleParameters, road: RoadModel):
        self.cfg = cfg
        self.params = params
        self.road = road

    def check(self, cand: CandidateTrajectory, predictions: list[ObjectPrediction]) -> CheckResult:
        return self.check_all([cand], predictions)[0]

    def check_all(self, cands: list[CandidateTrajectory],
                  predictions: list[ObjectPrediction], extra_margin: float = 0.0) -> list[CheckResult]:
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
        soft = np.zeros(C, dtype=bool)     # margin-only (degraded-tier) candidates: still collision-checked

        grace = rel_t < cfg.collision_margin_grace_s          # the current state is not the planner's choice

        # 1. curvature / steering limit (hard steering saturation is never excused)
        bad = K > p.max_curvature * (1.0 + 1e-6)
        for i in np.nonzero(bad.any(axis=1))[0]:
            k = int(np.argmax(K[i]))
            self._reject(cands[i], RejectionReason.CURVATURE,
                         f"|kappa|={K[i, k]:.3f} > {p.max_curvature:.3f} 1/m at +{rel_t[i, k]:.1f}s")
            alive[i] = False

        # 2. lateral acceleration (comfort limit; excused during the grace window, the ego is already turning)
        a_lat = V ** 2 * K                                  # V may be negative (reverse); squared
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
                soft[i] = True                    # a squeezed candidate must still be clear of objects
                alive[i] = False

        # 4. predicted collisions
        if predictions:
            M = len(predictions)
            dist = np.empty((M, C, N))
            sig = np.empty((M, C, N))
            w = np.array([pr.risk_weight for pr in predictions])
            bw = np.array([max(pr.length, pr.width) for pr in predictions])
            obj_s = np.zeros(M); obj_d = np.zeros(M); obj_vl = np.zeros(M)
            for j, pr in enumerate(predictions):
                s_o, d_o, h_o = self.road.project(float(pr.x[0]), float(pr.y[0]))
                obj_s[j], obj_d[j] = s_o, d_o
                obj_vl[j] = -float(pr.vx[0]) * np.sin(h_o) + float(pr.vy[0]) * np.cos(h_o)
            # The ego footprint corners are identical for every object, so build them once instead
            # of once per object inside box_sequence_distance.
            ego_corners = box_corners(FX, FY, FYAW, p.length, p.width)
            for j, pred in enumerate(predictions):
                ox = np.interp(T, pred.times, pred.x).ravel()
                oy = np.interp(T, pred.times, pred.y).ravel()
                oh = np.interp(T, pred.times, np.unwrap(pred.heading)).ravel()
                dist[j] = box_sequence_distance(FX, FY, FYAW, p.length, p.width,
                                                ox, oy, oh, pred.length, pred.width,
                                                a_corners=ego_corners).reshape(C, N)
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
            # The margin grows with how uncertain the object's position is AT THE TIME the candidate would be
            # there, not just with the measurement noise now. An animal two seconds out has a much wider
            # distribution than the tracker's current error, and a fixed margin lets the ego accelerate past it
            # on a prediction it has no right to trust. `sig` is the predicted covariance projected onto the
            # line to the ego, so this is the real thing, capped by uncertainty_margin_max_m.
            unc = np.minimum(cfg.uncertainty_margin_gain * sig, cfg.uncertainty_margin_max_m)     # (M,C,N)
            margins_t = cfg.safety_margin_m + extra_margin + unc                                  # (M,C,N)
            margins = np.array([cfg.safety_margin_m + extra_margin                                # scalar view,
                                + min(cfg.uncertainty_margin_gain * pr.meas_sigma,                # used beyond
                                      cfg.uncertainty_margin_max_m)                               # the horizon
                                for pr in predictions])
            # inside-margin counts as a hit only when CLOSING relative to the current distance (or overlapping):
            # a candidate that slides past an object at the clearance the ego already has is not a collision
            d0 = dist[:, :, :1]                                                    # (M,C,1) current distance
            closing = dist < d0 - 0.05
            # creep guard. A candidate that violates a margin/standoff is only scorable in the degraded tier
            # if it does not end closer to the object than HOLDING POSITION would (allowing for the distance a
            # comfortable stop from the current speed needs). Otherwise the degraded tier lets the ego inch
            # into an obstacle one cycle at a time and hides that it is boxed in (which is what triggers the
            # reversing recovery).
            ex_s = FX.reshape(C, N)[:, 0]; ey_s = FY.reshape(C, N)[:, 0]; eyaw_s = YAW[:, 0]
            d_hold = np.stack([box_sequence_distance(ex_s, ey_s, eyaw_s, p.length, p.width,
                                                     np.interp(T[:, -1], pr.times, pr.x), np.interp(T[:, -1], pr.times, pr.y),
                                                     np.interp(T[:, -1], pr.times, np.unwrap(pr.heading)), pr.length, pr.width)
                               for pr in predictions])                                            # (M,C)
            allow = np.maximum(V[:, 0], 0.0) ** 2 / (2 * cfg.comfortable_deceleration_mps2) + cfg.creep_guard_slack_m  # (C,)
            hit = (dist <= 0.0) | ((dist <= margins_t) & closing & margin_ok[None, :, :])              # (M,C,N)
            for i in range(C):
                results[i].distances = dist[:, i, :]
                results[i].sigmas = sig[:, i, :]
                results[i].risk_weights = w
                results[i].band_widths = bw
                results[i].obj_s, results[i].obj_d, results[i].obj_v_lat = obj_s, obj_d, obj_vl
                cands[i].min_clearance = float(dist[:, i, :].min())
                if (alive[i] or soft[i]) and hit[:, i, :].any():
                    any_t = hit[:, i, :].any(axis=0)
                    k = int(np.argmax(any_t))
                    j = int(np.argmax(hit[:, i, k]))
                    cands[i].collision_time = float(rel_t[i, k])
                    self._reject(cands[i], RejectionReason.COLLISION,
                                 f"distance {dist[j, i, k]:.2f} m <= margin {margins_t[j, i, k]:.2f} m "
                                 f"to {predictions[j].object_id} at +{rel_t[i, k]:.1f}s")
                    # inside the margin but never actually touching (and not creeping): still scorable in the
                    # degraded tier, so a tight pass beats a stop that would itself be struck
                    creep = dist[j, i, -1] < d_hold[j, i] - allow[i]
                    touching = bool(np.any(dist[:, i, :] <= 0.0))
                    cands[i].margin_only = soft[i] = not touching and not creep
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
                if cands[i].id.startswith("reverse_"):
                    continue                                  # reversing ends at rest by construction; judged by collision only
                if (alive[i] or soft[i]) and cands[i].target_speed < cfg.stop_standoff_speed_mps and predictions:
                    d_end_all = np.minimum(dist[:, i, -1], d_now[:, i])
                    j = int(np.argmin(d_end_all))
                    if d_end_all[j] < cfg.stop_standoff_m:
                        creep = (dist[j, i, -1] < d_hold[j, i] - allow[i]) or (d_now[j, i] < dist[j, i, 0] - allow[i])
                        self._reject(cands[i], RejectionReason.TERMINAL_EXPOSURE,
                                     f"would come to rest {d_end_all[j]:.1f} m from {predictions[j].object_id} "
                                     f"(< standoff {cfg.stop_standoff_m:.1f} m)" + (" - creeping closer" if creep else ""))
                        cands[i].margin_only = soft[i] = not creep   # holding position is degraded-scorable, creeping is not
                        alive[i] = False

            # 5. terminal exposure: continue each surviving candidate beyond the horizon along the corridor
            #    at its terminal speed (frozen for a stop) and extrapolate objects at constant velocity.
            #    This runs for every candidate that will be SCORED, not only the still-feasible ones.
            #    Restricting it to `alive` left the graded `blocked` cost at exactly 0.0 for every
            #    candidate rejected in sections 3 and 4 (0 of 2113 boundary-rejected and 0 of 1420
            #    collision-rejected ever scored non-zero, against 61% of feasible ones). Since that
            #    weight is 3.0, the degraded tier was handing squeezed candidates an advantage of up
            #    to 3.0 purely from the order in which rejections happen.
            T_end = float(rel_t[0, -1])
            idx = np.zeros(0, dtype=int)
            scorable = np.array([alive[i] or cands[i].margin_only for i in range(C)])
            if cfg.route_lookahead_s > T_end + 1e-9 and scorable.any():
                # Both bodies travel in straight lines out here (constant velocity from the end
                # state), so the continuation is sampled coarsely: a finer grid costs time
                # linearly and tells us nothing a linear interpolation would not.
                ext_dt = cfg.exposure_step_s
                ext = np.arange(T_end + ext_dt, cfg.route_lookahead_s + 1e-9, ext_dt)               # (E,)
                # reversing candidates end at rest away from the blocker by construction: collision check only
                idx = np.array([i for i in np.nonzero(scorable)[0] if not cands[i].id.startswith("reverse_")], dtype=int)
                E = len(ext)
            if len(idx) > 0:
                if all(cands[i].trajectory.s is not None for i in idx):
                    s_end = np.array([cands[i].trajectory.s[-1] for i in idx])
                    d_end = np.array([cands[i].trajectory.d[-1] for i in idx])
                    # A candidate that ends at rest would otherwise extrapolate to a standstill and
                    # meet nothing at all, scoring `blocked` = 0 by construction. That is the same
                    # structural zero as the one above, mirrored: it made stopping the cheapest
                    # option the moment moving candidates started carrying the cost honestly. The
                    # ego does not stay stopped for ever, so the continuation resumes at a walking
                    # pace and the term measures where the candidate LEAVES you, not the fact that
                    # it stopped. Coming to rest too close is the stop-standoff rule's job, below.
                    v_end = np.maximum(np.array([cands[i].trajectory.velocity[-1] for i in idx]),
                                       cfg.exposure_resume_speed_mps)
                    s_grid = s_end[:, None] + v_end[:, None] * (ext - T_end)[None, :]              # (I,E)
                    # the lateral transition usually outlasts the horizon (a wide shift needs ~9 m of travel).
                    # Continue it at the rate the candidate ended with, up to its target offset, instead of
                    # freezing d: otherwise a candidate half-way around an obstacle looks like it hits it.
                    h_end = np.array([cands[i].trajectory.heading_rel[-1] if cands[i].trajectory.heading_rel
                                      is not None else 0.0 for i in idx])
                    d_tgt = np.array([cands[i].lateral_offset_end for i in idx])
                    d_grid = d_end[:, None] + np.tan(h_end)[:, None] * (s_grid - s_end[:, None])
                    lo_d = np.minimum(d_end, d_tgt)[:, None]
                    hi_d = np.maximum(d_end, d_tgt)[:, None]
                    d_grid = np.clip(d_grid, lo_d, hi_d)
                    s_ext, d_ext = s_grid.ravel(), d_grid.ravel()
                    cx, cy, ch = self.road.to_cartesian(s_ext, d_ext)
                    ex = cx + off * np.cos(ch)
                    ey = cy + off * np.sin(ch)
                    eyaw = ch
                else:
                    ex = np.repeat(FX.reshape(C, N)[idx, -1], E)
                    ey = np.repeat(FY.reshape(C, N)[idx, -1], E)
                    eyaw = np.repeat(YAW[idx, -1], E)
                t_abs = (T[idx, 0][:, None] + ext[None, :]).ravel()
                ext_corners = box_corners(ex, ey, eyaw, p.length, p.width)
                for j, pred in enumerate(predictions):
                    dt_ext = t_abs - pred.times[-1]
                    ox = pred.x[-1] + pred.vx[-1] * dt_ext
                    oy = pred.y[-1] + pred.vy[-1] * dt_ext
                    oh = np.full_like(ox, pred.heading[-1])
                    # Coarse by construction: constant-velocity extrapolation 4-15 s ahead with the
                    # object already inflated by its measured sigma. Exact polygon refinement out
                    # here is spurious precision and was the single largest cost in the planner, so
                    # only pairs within 1 m of contact get it. The fallback bound UNDER-estimates
                    # distance, so this errs toward more exposure, never less.
                    dist_ext = box_sequence_distance(ex, ey, eyaw, p.length, p.width,
                                                     ox, oy, oh, pred.length + 2 * infl[j], pred.width + 2 * infl[j],
                                                     exact_within=1.0,
                                                     a_corners=ext_corners).reshape(len(idx), E)
                    # Grade the exposure instead of testing a binary "does it meet". Beyond the horizon the
                    # object is extrapolated for many seconds from a noisy tracked velocity, so a yes/no test
                    # sitting on the margin flips from cycle to cycle and takes whole groups of candidates in
                    # and out of the feasible set with it. Two continuous factors instead: how far inside the
                    # margin the closest approach comes, and how soon it comes.
                    scale = max(cfg.exposure_proximity_scale_m, 1e-6)
                    d_min = dist_ext.min(axis=1)                                              # (I,)
                    prox = np.clip((margins[j] + scale - dist_ext) / scale, 0.0, 1.0)         # (I,E)
                    recency = np.clip(1.0 - ext / cfg.route_lookahead_s, 0.0, 1.0)            # (E,)
                    severity = (prox * recency[None, :]).max(axis=1)   # worst mix of "how close" and "how soon"
                    hit_ext = dist_ext <= margins[j]
                    for r, i in enumerate(idx):
                        c = cands[i]
                        c.route_block_severity = max(c.route_block_severity, float(severity[r]))
                        if not hit_ext[r].any():
                            continue
                        k = int(np.argmax(hit_ext[r]))
                        t_meet = float(ext[k])
                        if c.route_block_time is None or t_meet < c.route_block_time:
                            c.route_block_time = t_meet
                        if not alive[i]:
                            continue
                        # The hard rejection is kept for exposure the ego is committed to, but it now needs a
                        # clear violation, not a graze: a candidate that merely brushes the margin stays
                        # feasible and carries the graded cost above, which is what stops the flip-flopping.
                        if t_meet <= cfg.terminal_exposure_horizon_s and d_min[r] <= margins[j] - cfg.exposure_reject_slack_m:
                            self._reject(c, RejectionReason.TERMINAL_EXPOSURE,
                                         f"continuing at {c.target_speed:.1f} m/s from the end state meets "
                                         f"{pred.object_id} at +{t_meet:.1f}s (closest {d_min[r]:.2f} m)")
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
