"""Actuator saturation model.

Sits between the (possibly unrealistic) requested ControlCommand and the
vehicle dynamics. Implements:

    delta_target = clip(delta_cmd, -delta_max, delta_max)
    delta[k+1]   = delta[k] + clip(delta_target - delta[k], -rate*dt, +rate*dt)
    a_net        = clip(a_cmd - b_cmd, -a_brake_max, +a_accel_max)

The planner and controller therefore never have access to instantaneous
steering changes or unbounded acceleration.
"""
from __future__ import annotations

from dataclasses import dataclass

from autonomy.core.types import ControlCommand, VehicleParameters


@dataclass
class ActuatedCommand:
    steering_angle: float          # rad, after angle + rate saturation
    net_acceleration: float        # m/s^2, after saturation (negative = braking)
    steering_saturated: bool
    steering_rate_saturated: bool
    acceleration_saturated: bool


class ActuatorModel:
    def __init__(self, params: VehicleParameters):
        self.p = params

    def apply(self, command: ControlCommand, current_steering: float, dt: float) -> ActuatedCommand:
        p = self.p
        target = max(-p.max_steering_angle, min(p.max_steering_angle, command.steering_angle))
        steer_sat = target != command.steering_angle

        max_delta = p.max_steering_rate * dt
        change = target - current_steering
        rate_sat = abs(change) > max_delta
        if rate_sat:
            change = max_delta if change > 0 else -max_delta
        new_steering = current_steering + change

        requested = command.acceleration - command.brake
        a_net = max(-p.max_deceleration, min(p.max_acceleration, requested))
        acc_sat = a_net != requested

        return ActuatedCommand(new_steering, a_net, steer_sat, rate_sat, acc_sat)
