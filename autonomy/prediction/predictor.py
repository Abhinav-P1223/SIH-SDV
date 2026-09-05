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

Output is time-indexed with absolute times t0, t0+dt, ..., t0+horizon.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from autonomy.core.config import ObjectProfiles, PredictionConfig
from autonomy.core.types import ObjectPrediction, ObjectState


class Predictor(ABC):
    @abstractmethod
    def predict(self, objects: list[ObjectState], now: float) -> list[ObjectPrediction]: ...


class ConstantVelocityPredictor(Predictor):
    def __init__(self, config: PredictionConfig, profiles: ObjectProfiles):
        self.cfg = config
        self.profiles = profiles
        self.rel_times = np.linspace(0.0, config.horizon_s, config.steps)

    def predict(self, objects: list[ObjectState], now: float) -> list[ObjectPrediction]:
        out: list[ObjectPrediction] = []
        t = self.rel_times
        for obj in objects:
            prof = self.profiles.get(obj.object_type)
            x = obj.x + obj.vx * t
            y = obj.y + obj.vy * t
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
                heading=np.full_like(t, obj.heading),
                vx=np.full_like(t, obj.vx), vy=np.full_like(t, obj.vy),
                covariances=cov, length=obj.length, width=obj.width,
                risk_weight=prof.risk_weight, meas_sigma=math.sqrt(max(meas_var, 0.0)),
            ))
        return out
