"""Short-horizon motion prediction for surrounding objects.

Stage 1: constant velocity, constant heading, with behaviour-profile-dependent
uncertainty that grows with time:

    sigma_l^2(t) = sigma_pos0^2 + f_s^2 * [ (sigma_vel * t)^2 + (0.5 * sigma_acc * t^2)^2 ]
    sigma_t(t)   = lateral_factor * sigma_l(t)        (across the object's heading)

f_s = static_factor while the object is (near) stationary, else 1. The
covariance is anisotropic: a road-following vehicle mostly deviates along its
heading, a pedestrian or cow can go anywhere (lateral_factor 1.0). The initial
term is the larger of the profile's sigma_pos0 and the tracked object's own
positional covariance (sensor fusion supplies it; ground truth supplies zero).

Road-following prior: when a road model is available and the object's class
`follows_road` and its heading is within `road_following_heading_tol_deg` of
the corridor direction (either way), the mean path follows the corridor at the
object's along-road speed while its lateral velocity decays exponentially
(`lateral_velocity_decay_s`). A tracked vehicle's small lateral velocity is
mostly estimation noise; extrapolating it linearly would drift the prediction
across the road and make the planner dither. Pedestrians, cattle and pushcarts
keep free constant-velocity motion. UNKNOWN tracks (not yet classified) are
treated as road-following only while they move faster than 3 m/s.

Latency compensation: an ObjectState whose `timestamp` is older than `now` is
first propagated to `now` at its own velocity (the age is what a delayed sensor
or a slow fusion cycle would introduce).

Output is time-indexed with absolute times t0, t0+dt, ..., t0+horizon.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from typing import Optional

from autonomy.core.config import ObjectProfiles, PredictionConfig
from autonomy.core.geometry import wrap_angle
from autonomy.core.interfaces import RoadModel
from autonomy.core.types import ObjectPrediction, ObjectState, ObjectType


class Predictor(ABC):
    @abstractmethod
    def predict(self, objects: list[ObjectState], now: float) -> list[ObjectPrediction]: ...


class ConstantVelocityPredictor(Predictor):
    def __init__(self, config: PredictionConfig, profiles: ObjectProfiles, road: Optional[RoadModel] = None):
        self.cfg = config
        self.profiles = profiles
        self.road = road
        self.rel_times = np.linspace(0.0, config.horizon_s, config.steps)

    def _road_following(self, obj: ObjectState, prof) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Corridor-following mean path (x, y, heading arrays) or None if the prior does not apply."""
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
        v_along = obj.vx * math.cos(h_ref) + obj.vy * math.sin(h_ref)
        v_lat = -obj.vx * math.sin(h_ref) + obj.vy * math.cos(h_ref)
        tau = max(self.cfg.lateral_velocity_decay_s, 1e-3)
        d = d0 + v_lat * tau * (1.0 - np.exp(-t / tau))
        x, y, h = self.road.to_cartesian(s0 + v_along * t, d)
        heading = np.where(np.cos(rel) >= 0, h, wrap_angle(h + math.pi))
        return x, y, heading

    def predict(self, objects: list[ObjectState], now: float) -> list[ObjectPrediction]:
        out: list[ObjectPrediction] = []
        t = self.rel_times
        for obj_raw in objects:
            # latency compensation: a measurement older than `now` is propagated to now at its own velocity
            age = max(now - obj_raw.timestamp, 0.0) if obj_raw.timestamp is not None else 0.0
            obj = obj_raw if age < 1e-6 else replace(obj_raw, x=obj_raw.x + obj_raw.vx * age,
                                                     y=obj_raw.y + obj_raw.vy * age, timestamp=now)
            prof = self.profiles.get(obj.object_type)
            rf = self._road_following(obj, prof)
            if rf is not None:
                x, y, heading = rf
            else:
                x, y = obj.x + obj.vx * t, obj.y + obj.vy * t
                heading = np.full_like(t, obj.heading)
            cov0 = np.asarray(obj.covariance, dtype=float)
            meas_var = float(np.max(np.linalg.eigvalsh(cov0))) if cov0.size == 4 else 0.0
            sigma0_sq = max(prof.sigma_pos0_m ** 2, meas_var)
            f_s = prof.static_factor if obj.speed < 0.2 else 1.0
            growth = (f_s ** 2) * ((prof.sigma_vel_mps * t) ** 2 + (0.5 * prof.sigma_acc_mps2 * t ** 2) ** 2)
            var_l = sigma0_sq + growth
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
                heading=heading,
                vx=np.full_like(t, obj.vx), vy=np.full_like(t, obj.vy),
                covariances=cov, length=obj.length, width=obj.width,
                risk_weight=prof.risk_weight, meas_sigma=math.sqrt(max(meas_var, 0.0)),
            ))
        return out
