"""Short-horizon motion prediction for surrounding objects.

Stage 1: constant velocity, constant heading, with behaviour-profile-dependent
uncertainty that grows with time:

    sigma^2(t) = sigma_pos0^2 + (sigma_vel * t)^2 + (0.5 * sigma_acc * t^2)^2

The covariance is isotropic. The initial term is the larger of the profile's
sigma_pos0 and the tracked object's own positional covariance (Stage 2 fusion
will supply a meaningful covariance; ground truth supplies zero).

Output is time-indexed with absolute times t0, t0+dt, ..., t0+horizon.
"""
from __future__ import annotations

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
            sigma0_sq = max(prof.sigma_pos0_m ** 2, float(np.max(np.diag(cov0))) if cov0.size else 0.0)
            var = sigma0_sq + (prof.sigma_vel_mps * t) ** 2 + (0.5 * prof.sigma_acc_mps2 * t ** 2) ** 2
            cov = np.zeros((t.shape[0], 2, 2))
            cov[:, 0, 0] = var
            cov[:, 1, 1] = var
            out.append(ObjectPrediction(
                object_id=obj.id, object_type=obj.object_type,
                times=now + t, x=x, y=y,
                heading=np.full_like(t, obj.heading),
                vx=np.full_like(t, obj.vx), vy=np.full_like(t, obj.vy),
                covariances=cov, length=obj.length, width=obj.width,
                risk_weight=prof.risk_weight,
            ))
        return out
