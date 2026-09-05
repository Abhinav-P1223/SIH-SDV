"""Candidate trajectory generation in the corridor (Frenet) frame.

For each lateral end offset d_end and each terminal speed v_end a candidate is
built from two decoupled profiles:

Lateral, as a function of arc length sigma = s - s0 over a transition length
S = max(v0 * T_lat, S_min, S_kappa), with S_kappa = sqrt(5.77 |d_end - d0| / (0.9 kappa_max))
the shortest length at which the quintic's peak curvature stays inside the
steering limit, so a wide shift from a crawl is generated feasibly instead of
being generated too sharp and rejected:

    d(sigma) = quintic polynomial with
        d(0) = d0, d'(0) = tan(theta_rel), d''(0) = kappa_ego,
        d(S) = d_end, d'(S) = 0, d''(S) = 0
    d(sigma) = d_end for sigma > S

Longitudinal, as a function of time:

    v(t) = v0 + a*t saturated at v_end, a = +a_accel (comfortable) or -a_decel
    s(t) = s0 + integral v dt        (trapezoidal)

Conversion to Cartesian uses the road's reference heading h_ref(s):

    yaw     = h_ref + atan(d')
    kappa   = d'' / (1 + d'^2)^(3/2)        (reference curvature assumed ~0 per segment)

The generator never rejects anything; feasibility is the collision checker's
job so that rejections are recorded with reasons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autonomy.core.config import PlanningConfig
from autonomy.core.geometry import wrap_angle
from autonomy.core.interfaces import RoadModel
from autonomy.core.types import (CandidateTrajectory, SpeedPolicy, Trajectory, VehicleParameters,
                                 VehicleState)


@dataclass
class FrameOrigin:
    s0: float
    d0: float
    heading_rel: float
    h_ref: float


def quintic_coefficients(d0: float, dd0: float, ddd0: float,
                         d1: float, dd1: float, ddd1: float, S: float) -> np.ndarray:
    """Coefficients c0..c5 of d(sigma) satisfying the six boundary conditions."""
    c0, c1, c2 = d0, dd0, 0.5 * ddd0
    S2, S3, S4, S5 = S ** 2, S ** 3, S ** 4, S ** 5
    A = np.array([[S3, S4, S5],
                  [3 * S2, 4 * S3, 5 * S4],
                  [6 * S, 12 * S2, 20 * S3]])
    b = np.array([d1 - (c0 + c1 * S + c2 * S2),
                  dd1 - (c1 + 2 * c2 * S),
                  ddd1 - 2 * c2])
    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5])


def eval_quintic(c: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = c[0] + c[1] * sigma + c[2] * sigma ** 2 + c[3] * sigma ** 3 + c[4] * sigma ** 4 + c[5] * sigma ** 5
    dd = c[1] + 2 * c[2] * sigma + 3 * c[3] * sigma ** 2 + 4 * c[4] * sigma ** 3 + 5 * c[5] * sigma ** 4
    ddd = 2 * c[2] + 6 * c[3] * sigma + 12 * c[4] * sigma ** 2 + 20 * c[5] * sigma ** 3
    return d, dd, ddd


class CandidateGenerator:
    def __init__(self, cfg: PlanningConfig, params: VehicleParameters, road: RoadModel):
        self.cfg = cfg
        self.params = params
        self.road = road
        self.rel_times = np.linspace(0.0, cfg.horizon_s, cfg.steps)
        self._profile_cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    def frame(self, ego: VehicleState) -> FrameOrigin:
        s0, d0, h_ref = self.road.project(ego.x, ego.y)
        return FrameOrigin(s0, d0, float(wrap_angle(ego.yaw - h_ref)), h_ref)

    def speed_profile(self, v0: float, v_end: float, decel: float,
                      a0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(v(t), s_rel(t), a(t)) for the configured time grid. Cached per (v0, v_end, decel, a0).

        Comfortable profiles are jerk-limited S-curves: the acceleration starts at the ego's CURRENT
        acceleration a0, ramps toward the target (max_acceleration or -decel) at most max_jerk_mps3 per
        second, and is wound back so that a = 0 exactly when v_end is reached (a_allow = sqrt(2 J |dv|)).
        Seeding with a0 matters: the planner re-plans at 10 Hz and the controller tracks the first samples of
        each new profile, so a profile that always restarted from a = 0 would hold the vehicle near zero
        acceleration for ever. A hard stop (decel == max_deceleration) keeps the instantaneous full-braking
        ramp: safety beats comfort."""
        key = (round(v0, 4), round(v_end, 4), round(decel, 4), round(a0, 2))
        hit = self._profile_cache.get(key)
        if hit is not None:
            return hit
        t = self.rel_times
        a_target = self.params.max_acceleration if v_end >= v0 else -decel
        J = self.cfg.max_jerk_mps3
        if decel >= self.params.max_deceleration - 1e-9 or J <= 0.0:
            v_raw = v0 + a_target * t
            v = np.minimum(v_raw, v_end) if a_target >= 0 else np.maximum(v_raw, v_end)
        else:
            v = np.empty_like(t)
            v[0] = max(v0, 0.0)
            a = min(max(a0, -self.params.max_deceleration), self.params.max_acceleration)
            for k in range(1, len(t)):
                dt = t[k] - t[k - 1]
                dv = v_end - v[k - 1]
                a_cmd = math.copysign(min(abs(a_target), math.sqrt(2.0 * J * abs(dv))), dv) if abs(dv) > 1e-9 else 0.0
                a = a + max(-J * dt, min(J * dt, a_cmd - a))
                v_new = v[k - 1] + a * dt
                if (dv > 0 and v_new > v_end) or (dv < 0 and v_new < v_end):
                    v_new, a = v_end, 0.0
                v[k] = v_new
        v = np.clip(v, 0.0, self.params.max_speed)
        # acceleration over each interval (forward difference); last sample holds 0
        acc = np.concatenate([np.diff(v) / np.diff(t), [0.0]])
        s_rel = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
        out = (v, s_rel, acc)
        if len(self._profile_cache) > 256:
            self._profile_cache.clear()
        self._profile_cache[key] = out
        return out

    def build(self, ego: VehicleState, fr: FrameOrigin, d_end: float, v_end: float,
              decel: float, now: float, label: str) -> CandidateTrajectory:
        v0 = ego.longitudinal_velocity
        v, s_rel, acc = self.speed_profile(v0, v_end, decel, ego.longitudinal_acceleration)
        shift = abs(d_end - fr.d0)
        # transition length: long enough for the quintic's peak curvature (5.77*shift/S^2) to respect the steering
        # limit AND the lateral-acceleration comfort limit at the current speed (v0^2 * kappa)
        s_kappa = math.sqrt(5.77 * shift / (0.8 * self.params.max_curvature)) if shift > 1e-6 else 0.0
        s_alat = max(v0, 0.0) * math.sqrt(5.77 * shift / (0.9 * self.cfg.max_lateral_acceleration_mps2)) if shift > 1e-6 else 0.0
        S = max(v0 * self.cfg.lateral_transition_time_s, self.cfg.min_lateral_transition_length_m, s_kappa, s_alat)
        kappa0 = math.tan(ego.steering_angle) / self.params.wheelbase
        c = quintic_coefficients(fr.d0, math.tan(fr.heading_rel), kappa0, d_end, 0.0, 0.0, S)
        sigma = np.minimum(s_rel, S)
        d, dd, ddd = eval_quintic(c, sigma)
        held = s_rel > S
        d[held], dd[held], ddd[held] = d_end, 0.0, 0.0

        x, y, h_ref = self.road.to_cartesian(fr.s0 + s_rel, d)
        yaw = wrap_angle(h_ref + np.arctan(dd))
        kappa = ddd / np.power(1.0 + dd ** 2, 1.5)
        traj = Trajectory(t=now + self.rel_times, x=x, y=y, yaw=np.asarray(yaw), velocity=v,
                          curvature=kappa, acceleration=acc, id=label,
                          s=fr.s0 + s_rel, d=d, heading_rel=np.arctan(dd))
        return CandidateTrajectory(id=label, label=label, trajectory=traj,
                                   lateral_offset_end=d_end, target_speed=v_end)

    def build_reverse(self, ego: VehicleState, fr: FrameOrigin, distance: float, now: float) -> CandidateTrajectory:
        """Straight reversing recovery along the corridor at the current offset. Speed magnitude follows a
        trapezoid that starts at the current reverse speed |v0|, accelerates to reverse_speed_mps, holds and
        decelerates to rest after `distance` metres. A leg longer than the horizon can cover simply holds
        reverse speed to the end of the horizon (the planner commits to the leg and re-plans the remaining
        distance every cycle). Velocity is negative; the body keeps facing forward."""
        t = self.rel_times
        a = self.cfg.reverse_acceleration_mps2
        v_max = self.cfg.reverse_speed_mps
        m0 = max(-ego.longitudinal_velocity, 0.0)             # current reverse speed magnitude
        horizon = float(t[-1])
        d_stop = m0 ** 2 / (2 * a)                              # cannot stop sooner than this
        # longest manoeuvre that comes to rest inside the horizon
        t_acc_full, t_dec_full = max(v_max - m0, 0.0) / a, v_max / a
        if t_acc_full + t_dec_full <= horizon:
            d_fit = (m0 * t_acc_full + 0.5 * a * t_acc_full ** 2) + v_max * (horizon - t_acc_full - t_dec_full) \
                + v_max ** 2 / (2 * a)
        else:
            v_p = 0.5 * (a * horizon + m0)
            d_fit = (v_p ** 2 - m0 ** 2) / (2 * a) + v_p ** 2 / (2 * a)
        requested = distance
        distance = min(max(distance, d_stop), d_fit)
        v_tri = math.sqrt(max(a * distance + 0.5 * m0 ** 2, 0.0))   # triangular peak for this distance
        if v_tri <= v_max:
            v_peak, t_hold = max(v_tri, m0), 0.0
        else:
            v_peak = v_max
            d_acc = (v_max ** 2 - m0 ** 2) / (2 * a)
            t_hold = (distance - d_acc - v_max ** 2 / (2 * a)) / v_max
        t_acc = (v_peak - m0) / a
        t1, t2 = t_acc, t_acc + t_hold
        mag = np.where(t < t1, m0 + a * t, np.where(t < t2, v_peak, np.maximum(v_peak - a * (t - t2), 0.0)))
        if requested > d_fit + 1e-6:                            # leg continues beyond the horizon: no stop yet
            mag = np.minimum(m0 + a * t, v_max)
        v = -mag
        s_rel = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
        acc = np.concatenate([np.diff(v) / np.diff(t), [0.0]])
        x, y, h_ref = self.road.to_cartesian(fr.s0 + s_rel, np.full_like(t, fr.d0))
        yaw = np.full_like(t, fr.h_ref)
        traj = Trajectory(t=now + t, x=x, y=y, yaw=yaw, velocity=v, curvature=np.zeros_like(t),
                          acceleration=acc, id=f"reverse_{requested:.1f}m",
                          s=fr.s0 + s_rel, d=np.full_like(t, fr.d0), heading_rel=np.zeros_like(t))
        return CandidateTrajectory(id=traj.id, label=traj.id, trajectory=traj,
                                   lateral_offset_end=fr.d0, target_speed=0.0)

    def lateral_offsets(self, fr: FrameOrigin, desired_offset: float) -> list[float]:
        """End offsets desired + k*step for every k whose footprint fits inside the corridor at s0."""
        d_right, d_left = self.road.lateral_bounds_at(fr.s0) if hasattr(self.road, "lateral_bounds_at") \
            else (-math.inf, math.inf)
        half = 0.5 * self.params.width + self.cfg.boundary_margin_m
        lo, hi = d_right + half, d_left - half
        step = self.cfg.lateral_step_m
        if step <= 0 or not (math.isfinite(lo) and math.isfinite(hi)):
            offs = [desired_offset + o for o in self.cfg.lateral_offsets_m]
        else:
            k_min = math.ceil((lo - desired_offset) / step - 1e-9)
            k_max = math.floor((hi - desired_offset) / step + 1e-9)
            offs = [desired_offset + k * step for k in range(k_min, k_max + 1)]
            # the grid is anchored on the desired offset, so both corridor extremes are usually quantised away;
            # squeezing past an obstacle needs them, so add each edge when the grid does not already reach it
            offs += [e for e in (lo, hi) if all(abs(e - o) > 0.1 * step for o in offs)]
        offs = sorted(o for o in offs if lo - 1e-9 <= o <= hi + 1e-9)
        return offs if offs else [min(max(fr.d0, lo), hi) if math.isfinite(lo) else fr.d0]

    # ------------------------------------------------------------------ #
    def generate(self, ego: VehicleState, policy: SpeedPolicy, desired_offset: float,
                 now: float, reverse_distances: list[float] | None = None) -> tuple[list[CandidateTrajectory], FrameOrigin]:
        """reverse_distances: overrides cfg.reverse_distances_m (the planner passes the committed remaining
        distance while the ego is already rolling backwards)."""
        fr = self.frame(ego)
        v0 = ego.longitudinal_velocity
        cands: list[CandidateTrajectory] = []
        comfortable = self.cfg.comfortable_deceleration_mps2
        hard = self.params.max_deceleration

        if policy.force_stop:
            for name, dec in (("stop_soft", comfortable), ("stop_hard", hard)):
                cands.append(self.build(ego, fr, fr.d0, 0.0, dec, now, f"{name}_d{fr.d0:+.1f}"))
            return cands, fr
        rolling_back = v0 < -0.3
        if policy.allow_reverse and v0 <= 0.3:
            for D in (reverse_distances if reverse_distances is not None else self.cfg.reverse_distances_m):
                cands.append(self.build_reverse(ego, fr, D, now))
        if rolling_back:
            # no forward candidates while rolling backwards: finish (or stop) the manoeuvre first
            cands.append(self.build(ego, fr, fr.d0, 0.0, hard, now, f"stop_hard_d{fr.d0:+.2f}"))
            return cands, fr

        offsets = self.lateral_offsets(fr, desired_offset) if policy.allow_lateral_avoidance else [desired_offset]

        speeds = sorted({round(max(0.0, f * policy.target_speed), 3) for f in self.cfg.speed_fractions} | {0.0},
                        reverse=True)
        for d_end in offsets:
            for v_end in speeds:
                cands.append(self.build(ego, fr, d_end, v_end, comfortable, now,
                                        f"d{d_end:+.2f}_v{v_end:.1f}"))
        # hard stop along the current offset is always available as a last resort
        cands.append(self.build(ego, fr, fr.d0, 0.0, hard, now, f"stop_hard_d{fr.d0:+.2f}"))
        return cands, fr
