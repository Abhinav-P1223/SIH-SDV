"""Lateral trajectory tracking: Stanley controller.

    e       = cross-track error of the FRONT axle to the nearest trajectory point,
              positive when the path lies to the LEFT of the vehicle
    theta_e = heading error = yaw_path(look-ahead) - yaw_vehicle  (wrapped)
    delta_ff= atan(L * kappa_path(look-ahead))        kinematic curvature feed-forward
    delta   = heading_gain * theta_e + atan( k * e / (k_soft + v) ) + delta_ff
    delta   = clip(delta, -delta_lat, delta_lat)   with delta_lat = atan(L * a_lat_max / v^2)
    delta   = clip(delta, -delta_max, delta_max)

The look-ahead point sits max(lookahead_time_s * v, min_lookahead_m) along the
path beyond the nearest point. Because the planner re-plans from the ego pose
every cycle, the nearest point always has ~zero error; the look-ahead and the
curvature feed-forward are what make the vehicle actually bend onto the path.

The interface (`LateralController.compute`) is what an MPC would implement.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from autonomy.core.config import StanleyConfig
from autonomy.core.geometry import wrap_angle
from autonomy.core.types import Trajectory, VehicleParameters, VehicleState


@dataclass
class LateralDebug:
    target_index: int
    target_x: float
    target_y: float
    cross_track_error: float
    heading_error: float
    steering_command: float


class LateralController(ABC):
    @abstractmethod
    def compute(self, ego: VehicleState, traj: Trajectory) -> tuple[float, LateralDebug]: ...

    def reset(self) -> None: ...


class StanleyController(LateralController):
    def __init__(self, cfg: StanleyConfig, params: VehicleParameters):
        self.cfg = cfg
        self.params = params

    def compute(self, ego: VehicleState, traj: Trajectory) -> tuple[float, LateralDebug]:
        lf = self.params.cg_to_front_axle
        fx = ego.x + lf * math.cos(ego.yaw)
        fy = ego.y + lf * math.sin(ego.yaw)
        d2 = (traj.x - fx) ** 2 + (traj.y - fy) ** 2
        i = int(np.argmin(d2))
        px, py = float(traj.x[i]), float(traj.y[i])
        # signed lateral offset of the nearest path point relative to the vehicle heading
        dx, dy = px - fx, py - fy
        e = -math.sin(ego.yaw) * dx + math.cos(ego.yaw) * dy        # +ve: path is to the left
        v = abs(ego.longitudinal_velocity)
        if ego.longitudinal_velocity < -0.05 or float(traj.velocity[min(1, len(traj) - 1)]) < -1e-6:
            # reversing: hold the path heading; steering acts with inverted sign on the travel direction
            theta_e = float(wrap_angle(float(traj.yaw[i]) - ego.yaw))
            delta = -self.cfg.heading_gain * theta_e
            delta = max(-self.params.max_steering_angle, min(self.params.max_steering_angle, delta))
            return delta, LateralDebug(i, px, py, e, theta_e, delta)

        # look-ahead point for heading and curvature
        la = max(self.cfg.lookahead_time_s * v, self.cfg.min_lookahead_m)
        seg = np.hypot(np.diff(traj.x[i:]), np.diff(traj.y[i:]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        j = i + int(np.searchsorted(cum, la))
        j = min(j, len(traj) - 1)
        theta_e = float(wrap_angle(float(traj.yaw[j]) - ego.yaw))
        delta_ff = math.atan(self.params.wheelbase * float(traj.curvature[j]))

        delta = self.cfg.heading_gain * theta_e + math.atan2(self.cfg.k_gain * e, self.cfg.k_soft + v) + delta_ff
        if v > 1.0:
            delta_lat = math.atan(self.params.wheelbase * self.cfg.max_lateral_acceleration_mps2 / (v * v))
            delta = max(-delta_lat, min(delta_lat, delta))
        delta = max(-self.params.max_steering_angle, min(self.params.max_steering_angle, delta))
        return delta, LateralDebug(j, float(traj.x[j]), float(traj.y[j]), e, theta_e, delta)
