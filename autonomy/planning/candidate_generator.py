"""Candidate trajectory generation in the corridor (Frenet) frame.

For each lateral end offset d_end and each terminal speed v_end a candidate is
built from two decoupled profiles:

Lateral, as a function of arc length sigma = s - s0 over a transition length
S = max(v0 * T_lat, S_min):

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

    def speed_profile(self, v0: float, v_end: float, decel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(v(t), s_rel(t), a(t)) for the configured time grid. Cached per (v0, v_end, decel)."""
        key = (round(v0, 4), round(v_end, 4), round(decel, 4))
        hit = self._profile_cache.get(key)
        if hit is not None:
            return hit
        t = self.rel_times
        a = self.params.max_acceleration if v_end >= v0 else -decel
        v_raw = v0 + a * t
        v = np.minimum(v_raw, v_end) if a >= 0 else np.maximum(v_raw, v_end)
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
        v, s_rel, acc = self.speed_profile(v0, v_end, decel)
        S = max(v0 * self.cfg.lateral_transition_time_s, self.cfg.min_lateral_transition_length_m)
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
                          curvature=kappa, acceleration=acc, id=label)
        return CandidateTrajectory(id=label, label=label, trajectory=traj,
                                   lateral_offset_end=d_end, target_speed=v_end)

    # ------------------------------------------------------------------ #
    def generate(self, ego: VehicleState, policy: SpeedPolicy, desired_offset: float,
                 now: float) -> tuple[list[CandidateTrajectory], FrameOrigin]:
        fr = self.frame(ego)
        v0 = ego.longitudinal_velocity
        cands: list[CandidateTrajectory] = []
        comfortable = self.cfg.comfortable_deceleration_mps2
        hard = self.params.max_deceleration

        if policy.force_stop:
            for name, dec in (("stop_soft", comfortable), ("stop_hard", hard)):
                cands.append(self.build(ego, fr, fr.d0, 0.0, dec, now, f"{name}_d{fr.d0:+.1f}"))
            return cands, fr

        offsets = [desired_offset + o for o in self.cfg.lateral_offsets_m] if policy.allow_lateral_avoidance \
            else [desired_offset]
        # keep candidates whose end offset can physically fit inside the corridor
        d_right, d_left = self.road.lateral_bounds_at(fr.s0) if hasattr(self.road, "lateral_bounds_at") \
            else (-math.inf, math.inf)
        half = 0.5 * self.params.width + self.cfg.boundary_margin_m
        offsets = [o for o in offsets if d_right + half <= o <= d_left - half]
        if not offsets:
            offsets = [fr.d0]

        speeds = sorted({round(max(0.0, f * policy.target_speed), 3) for f in self.cfg.speed_fractions} | {0.0},
                        reverse=True)
        for d_end in offsets:
            for v_end in speeds:
                cands.append(self.build(ego, fr, d_end, v_end, comfortable, now,
                                        f"d{d_end:+.2f}_v{v_end:.1f}"))
        # hard stop along the current offset is always available as a last resort
        cands.append(self.build(ego, fr, fr.d0, 0.0, hard, now, f"stop_hard_d{fr.d0:+.2f}"))
        return cands, fr
