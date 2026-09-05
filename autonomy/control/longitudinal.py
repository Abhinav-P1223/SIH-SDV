"""Longitudinal tracking: PID speed control with trajectory feed-forward.

    v_ref  = trajectory velocity at the current time
    a_ff   = trajectory acceleration at the current time (optional)
    e      = v_ref - v
    u      = a_ff + kp e + ki * integral(e) + kd * de/dt
    acceleration = max(u, 0),  brake = max(-u, 0)

Anti-windup: the integral is clamped and is not accumulated while the output
is saturated in the direction of the error. When the reference is (near) zero and
the vehicle has stopped, a holding brake is applied so the car does not creep.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from autonomy.core.config import PIDConfig
from autonomy.core.types import Trajectory, VehicleParameters, VehicleState


@dataclass
class LongitudinalDebug:
    target_speed: float
    speed_error: float
    feedforward: float
    integral: float
    acceleration: float
    brake: float


class LongitudinalController(ABC):
    @abstractmethod
    def compute(self, ego: VehicleState, traj: Trajectory, now: float, dt: float) -> tuple[float, float, LongitudinalDebug]: ...

    def reset(self) -> None: ...


class PIDSpeedController(LongitudinalController):
    def __init__(self, cfg: PIDConfig, params: VehicleParameters, stop_speed: float, hold_brake: float = 2.0):
        self.cfg = cfg
        self.params = params
        self.stop_speed = stop_speed
        self.hold_brake = hold_brake
        self.integral = 0.0
        self.prev_error: float | None = None
        self.reversing = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def compute(self, ego: VehicleState, traj: Trajectory, now: float, dt: float) -> tuple[float, float, LongitudinalDebug]:
        v_ref = float(np.interp(now, traj.t, traj.velocity))
        v_ahead = float(np.interp(now + self.cfg.preview_s, traj.t, traj.velocity))   # move-off detection
        a_ff = float(np.interp(now, traj.t, traj.acceleration)) if self.cfg.use_feedforward else 0.0
        v = ego.longitudinal_velocity
        reversing = v_ref < -1e-6 or v_ahead < -1e-6 or (v < -0.05 and v_ref <= 0.0)
        if reversing:
            # track speed magnitudes; the tracker sets the reverse gear
            v_ref, v_ahead, v, a_ff = -v_ref, -v_ahead, -v, -a_ff
        e = v_ref - v

        lim = self.cfg.integral_limit
        de = 0.0 if self.prev_error is None or dt <= 0 else (e - self.prev_error) / dt
        self.prev_error = e

        u_raw = a_ff + self.cfg.kp * e + self.cfg.ki * self.integral + self.cfg.kd * de
        u = max(-self.params.max_deceleration, min(self.params.max_acceleration, u_raw))
        saturated_same_sign = (u_raw > u and e > 0) or (u_raw < u and e < 0)
        if not saturated_same_sign:
            self.integral = max(-lim, min(lim, self.integral + e * dt))

        if max(v_ref, v_ahead) < self.stop_speed and v < self.stop_speed:      # genuine hold, not a move-off
            acc, brake = 0.0, self.hold_brake
            self.integral = 0.0
        elif u >= 0:
            acc, brake = u, 0.0
        else:
            acc, brake = 0.0, -u
        self.reversing = reversing
        return acc, brake, LongitudinalDebug(v_ref, e, a_ff, self.integral, acc, brake)
