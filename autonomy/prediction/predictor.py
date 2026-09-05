"""Short-horizon motion prediction for surrounding objects.

Mean path: constant heading, piecewise constant-acceleration (CA) speed. The
predictor is the only component that sees successive ObjectStates, so it
estimates each object's acceleration itself from the velocity history of the
last `velocity_history_s` seconds of `predict` calls (least-squares slope,
first-order low-pass with time constant `acceleration_filter_s`, magnitude
clamped to the per-class `max_acceleration_mps2`). With fewer than
`min_history_samples` samples the acceleration is zero and the model reduces
exactly to constant velocity.

Signed 1-D propagation of the speed component v0 with acceleration a:

    t_e  = T_a                       if a pushes along v0 (or v0 = 0, a > 0)
         = min(T_a, -v0 / a)         if a opposes v0 (the object stops at t_e)
    v(t) = v0 + a * min(t, t_e)
    s(t) = v0 * tc + 0.5 * a * tc^2 + v(t_e) * (t - tc),   tc = min(t, t_e)

T_a = `acceleration_horizon_s`: CA is not credible for long, so the velocity is
held constant beyond it. A decelerating object stops and stays stopped; a
speed sign reversal is never predicted.

Uncertainty grows with time per behaviour profile:

    sigma_l^2(t) = sigma_pos0^2 + f_i * f_s^2 * [ (sigma_vel * t)^2 + (0.5 * sigma_acc * t^2)^2 ]
    sigma_t^2(t) = sigma_pos0^2 + lateral_factor^2 * f_s^2 * [ ... ]      (across the heading)

f_s = static_factor while the object is (near) stationary, else 1. f_i is the
intent prior for free movers (classes without `follows_road`): an object that
is braking harder than `intent_accel_threshold_mps2` along its heading is
about to stop (f_i = `intent_stopping_growth_factor` < 1); one that is
speeding up is committing to its crossing (f_i = `intent_accelerating_growth_factor`
> 1); otherwise f_i = 1. The initial term is the larger of the profile's
sigma_pos0 and the tracked object's own positional covariance.

Road-following prior: when a road model is available and the object's class
`follows_road` and its heading is within `road_following_heading_tol_deg` of
the corridor direction (either way), the mean path follows the corridor with
the CA speed law applied to the along-road velocity component only, while the
lateral velocity decays exponentially (`lateral_velocity_decay_s`). A tracked
vehicle's small lateral velocity is mostly estimation noise. UNKNOWN tracks are
treated as road-following only while they move faster than 3 m/s.

Latency compensation: an ObjectState whose `timestamp` is older than `now` is
first propagated to `now` at its own velocity.

Output is time-indexed with absolute times t0, t0+dt, ..., t0+horizon. The
velocity arrays carry v(t): index 0 is the current velocity, index -1 the
held terminal velocity used for beyond-horizon extrapolation downstream.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from autonomy.core.config import ObjectProfile, ObjectProfiles, PredictionConfig
from autonomy.core.geometry import wrap_angle
from autonomy.core.interfaces import RoadModel
from autonomy.core.types import ObjectPrediction, ObjectState, ObjectType


class Predictor(ABC):
    @abstractmethod
    def predict(self, objects: list[ObjectState], now: float) -> list[ObjectPrediction]: ...


@dataclass
class _VelocityHistory:
    """Recent velocity samples of one object and its filtered acceleration estimate (world frame)."""
    times: deque[float] = field(default_factory=deque)
    vx: deque[float] = field(default_factory=deque)
    vy: deque[float] = field(default_factory=deque)
    ax: float = 0.0
    ay: float = 0.0


class ConstantVelocityPredictor(Predictor):
    """Constant-heading predictor with self-estimated acceleration (see module docstring)."""

    def __init__(self, config: PredictionConfig, profiles: ObjectProfiles, road: Optional[RoadModel] = None):
        self.cfg = config
        self.profiles = profiles
        self.road = road
        self.rel_times = np.linspace(0.0, config.horizon_s, config.steps)
        self.a_max_by_type = {ObjectType(k): float(v) for k, v in config.max_acceleration_by_type.items()}
        self.history: dict[str, _VelocityHistory] = {}

    # ------------------------------------------------------------ acceleration estimation
    def _max_acceleration(self, object_type: ObjectType) -> float:
        return self.a_max_by_type.get(object_type, self.cfg.max_acceleration_mps2)

    def _update_acceleration(self, obj: ObjectState, t: float) -> tuple[float, float]:
        """Record the object's velocity at time t and return its filtered, clamped acceleration.

        The raw estimate is the least-squares slope of (t_i, v_i) over the history
        window; it is low-passed (a += alpha (a_ls - a), alpha = dt / (tau + dt)) and
        clamped to |a| <= a_max of the class. A track whose time does not advance
        (re-query of the same instant) or that has fewer than `min_history_samples`
        samples reports zero acceleration.
        """
        cfg = self.cfg
        hist = self.history.get(obj.id)
        if hist is None or t <= hist.times[-1] + 1e-9:
            hist = _VelocityHistory()
            self.history[obj.id] = hist
        dt = t - hist.times[-1] if hist.times else 0.0
        hist.times.append(t)
        hist.vx.append(obj.vx)
        hist.vy.append(obj.vy)
        while hist.times[0] < t - cfg.velocity_history_s - 1e-9:
            hist.times.popleft()
            hist.vx.popleft()
            hist.vy.popleft()
        if len(hist.times) < cfg.min_history_samples:
            hist.ax = hist.ay = 0.0
            return 0.0, 0.0
        tt = np.fromiter(hist.times, dtype=float)
        tm = tt - tt.mean()
        den = float(np.dot(tm, tm))
        if den > 1e-12:
            vx = np.fromiter(hist.vx, dtype=float)
            vy = np.fromiter(hist.vy, dtype=float)
            ax_ls = float(np.dot(tm, vx - vx.mean()) / den)
            ay_ls = float(np.dot(tm, vy - vy.mean()) / den)
            alpha = dt / (cfg.acceleration_filter_s + dt)
            hist.ax += alpha * (ax_ls - hist.ax)
            hist.ay += alpha * (ay_ls - hist.ay)
            a_max = self._max_acceleration(obj.object_type)
            mag = math.hypot(hist.ax, hist.ay)
            if mag > a_max:
                hist.ax *= a_max / mag
                hist.ay *= a_max / mag
        return hist.ax, hist.ay

    def _forget_stale(self, seen: set[str], now: float) -> None:
        cutoff = now - self.cfg.velocity_history_s
        for oid in [k for k, h in self.history.items() if k not in seen and h.times[-1] < cutoff]:
            del self.history[oid]

    # ------------------------------------------------------------ propagation
    def _propagate_speed(self, v0: float, a: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed 1-D CA motion with a finite acceleration horizon and no sign reversal.

        Returns (s(t), v(t)) per the module docstring. Exactly s = v0 t when a = 0.
        """
        if abs(a) < 1e-9:
            return v0 * t, np.full_like(t, v0)
        opposing = (a < 0.0) if v0 >= 0.0 else (a > 0.0)
        t_e = min(self.cfg.acceleration_horizon_s, -v0 / a) if opposing else self.cfg.acceleration_horizon_s
        tc = np.minimum(t, t_e)
        v = v0 + a * tc
        s = v0 * tc + 0.5 * a * tc ** 2 + (v0 + a * t_e) * (t - tc)
        return s, v

    def _road_following(self, obj: ObjectState, prof: ObjectProfile, ax: float, ay: float
                        ) -> Optional[tuple[np.ndarray, ...]]:
        """Corridor-following mean path (x, y, heading, vx, vy) or None if the prior does not apply."""
        if self.road is None or not prof.follows_road:
            return None
        if obj.object_type == ObjectType.UNKNOWN and obj.speed < 3.0:
            return None
        s0, d0, h_ref = self.road.project(obj.x, obj.y)
        rel = float(wrap_angle(obj.heading - h_ref))
        tol = math.radians(self.cfg.road_following_heading_tol_deg)
        if abs(rel) > tol and abs(abs(rel) - math.pi) > tol:
            return None
        t = self.rel_times
        c_ref, s_ref = math.cos(h_ref), math.sin(h_ref)
        v_along = obj.vx * c_ref + obj.vy * s_ref
        v_lat0 = -obj.vx * s_ref + obj.vy * c_ref
        a_along = ax * c_ref + ay * s_ref
        ds, v_s = self._propagate_speed(v_along, a_along, t)
        tau = max(self.cfg.lateral_velocity_decay_s, 1e-3)
        v_lat = v_lat0 * np.exp(-t / tau)
        d = d0 + v_lat0 * tau * (1.0 - np.exp(-t / tau))
        x, y, h = self.road.to_cartesian(s0 + ds, d)
        heading = np.where(np.cos(rel) >= 0, h, wrap_angle(h + math.pi))
        vx = v_s * np.cos(h) - v_lat * np.sin(h)
        vy = v_s * np.sin(h) + v_lat * np.cos(h)
        return x, y, heading, vx, vy

    def _free_motion(self, obj: ObjectState, ax: float, ay: float) -> tuple[np.ndarray, ...]:
        """Constant-heading CA mean path (x, y, heading, vx, vy, a_along) for a free mover.

        The CA law acts on the speed along the direction of travel: the velocity
        direction, or for a stationary object the acceleration direction (it is
        starting to move) else the heading. The acceleration component across the
        direction of travel is dropped: the heading is held.
        """
        t = self.rel_times
        speed = obj.speed
        if speed > 1e-6:
            ux, uy = obj.vx / speed, obj.vy / speed
        elif math.hypot(ax, ay) > 1e-9:
            n = math.hypot(ax, ay)
            ux, uy = ax / n, ay / n
        else:
            ux, uy = math.cos(obj.heading), math.sin(obj.heading)
        a_along = ax * ux + ay * uy
        s, v = self._propagate_speed(speed, a_along, t)
        return obj.x + ux * s, obj.y + uy * s, np.full_like(t, obj.heading), ux * v, uy * v, a_along

    def _intent_factor(self, a_along: float) -> float:
        """Growth multiplier of the along-heading uncertainty from the object's speed trend."""
        if a_along < -self.cfg.intent_accel_threshold_mps2:
            return self.cfg.intent_stopping_growth_factor
        if a_along > self.cfg.intent_accel_threshold_mps2:
            return self.cfg.intent_accelerating_growth_factor
        return 1.0

    # ------------------------------------------------------------ interface
    def predict(self, objects: list[ObjectState], now: float) -> list[ObjectPrediction]:
        out: list[ObjectPrediction] = []
        t = self.rel_times
        seen: set[str] = set()
        for obj_raw in objects:
            seen.add(obj_raw.id)
            t_meas = obj_raw.timestamp if obj_raw.timestamp is not None else now
            ax, ay = self._update_acceleration(obj_raw, t_meas) if self.cfg.estimate_acceleration else (0.0, 0.0)
            # latency compensation: a measurement older than `now` is propagated to now at its own velocity
            age = max(now - t_meas, 0.0)
            obj = obj_raw if age < 1e-6 else replace(obj_raw, x=obj_raw.x + obj_raw.vx * age,
                                                     y=obj_raw.y + obj_raw.vy * age, timestamp=now)
            prof = self.profiles.get(obj.object_type)
            f_i = 1.0
            rf = self._road_following(obj, prof, ax, ay)
            if rf is not None:
                x, y, heading, vx, vy = rf
            else:
                x, y, heading, vx, vy, a_along = self._free_motion(obj, ax, ay)
                f_i = self._intent_factor(a_along)
            cov0 = np.asarray(obj.covariance, dtype=float)
            meas_var = float(np.max(np.linalg.eigvalsh(cov0))) if cov0.size == 4 else 0.0
            sigma0_sq = max(prof.sigma_pos0_m ** 2, meas_var)
            f_s = prof.static_factor if obj.speed < 0.2 else 1.0
            growth = (f_s ** 2) * ((prof.sigma_vel_mps * t) ** 2 + (0.5 * prof.sigma_acc_mps2 * t ** 2) ** 2)
            var_l = sigma0_sq + f_i * growth
            var_t = sigma0_sq + (prof.lateral_factor ** 2) * growth
            c, s = math.cos(obj.heading), math.sin(obj.heading)
            # rotate diag(var_l, var_t) from the object frame into the world frame
            cov = np.zeros((t.shape[0], 2, 2))
            cov[:, 0, 0] = var_l * c * c + var_t * s * s
            cov[:, 1, 1] = var_l * s * s + var_t * c * c
            cov[:, 0, 1] = cov[:, 1, 0] = (var_l - var_t) * s * c
            out.append(ObjectPrediction(
                object_id=obj.id, object_type=obj.object_type,
                times=now + t, x=x, y=y,
                heading=heading, vx=vx, vy=vy,
                covariances=cov, length=obj.length, width=obj.width,
                risk_weight=prof.risk_weight, meas_sigma=math.sqrt(max(meas_var, 0.0)),
            ))
        self._forget_stale(seen, now)
        return out
