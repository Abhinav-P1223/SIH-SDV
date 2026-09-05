"""Ego vehicle dynamics.

`VehicleModel` is the only code allowed to change the ego pose. It consumes a
ControlCommand, runs it through the ActuatorModel, and integrates the motion
equations. `KinematicBicycleModel` is the Stage 1 implementation; a
`DynamicBicycleModel` with tyre slip can implement the same interface later.

Kinematic bicycle model, CG reference point (all SI, radians):

    beta    = atan( lr / L * tan(delta) )          slip angle at CG
    x_dot   = v * cos(psi + beta)
    y_dot   = v * sin(psi + beta)
    psi_dot = v * cos(beta) * tan(delta) / L
    v_dot   = a_net
    v_lat   = v * sin(beta)

Integration: forward Euler by default (dt = 0.02 s is small relative to the
vehicle time constants). RK4 is available via `integrator="rk4"` for checks.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import replace

from autonomy.core.geometry import OrientedBox, wrap_angle
from autonomy.core.types import ControlCommand, VehicleParameters, VehicleState

from .actuators import ActuatedCommand, ActuatorModel


class VehicleModel(ABC):
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.actuators = ActuatorModel(params)
        self.last_actuated: ActuatedCommand | None = None

    @abstractmethod
    def step(self, state: VehicleState, command: ControlCommand, dt: float) -> VehicleState:
        """Advance the vehicle by dt under the (to-be-saturated) command."""

    def footprint(self, state: VehicleState) -> OrientedBox:
        """Oriented rectangle of the vehicle body for the given state."""
        return footprint_for(self.params, state.x, state.y, state.yaw)


def footprint_for(params: VehicleParameters, x: float, y: float, yaw: float) -> OrientedBox:
    off = params.footprint_center_offset
    return OrientedBox(x + off * math.cos(yaw), y + off * math.sin(yaw), yaw,
                       params.length, params.width)


class KinematicBicycleModel(VehicleModel):
    def __init__(self, params: VehicleParameters, integrator: str = "euler"):
        super().__init__(params)
        if integrator not in ("euler", "rk4"):
            raise ValueError("integrator must be 'euler' or 'rk4'")
        self.integrator = integrator

    # -- continuous-time derivatives -------------------------------------- #
    def _derivatives(self, x: float, y: float, psi: float, v: float,
                     delta: float, a_net: float) -> tuple[float, float, float, float]:
        p = self.params
        beta = math.atan(p.cg_to_rear_axle / p.wheelbase * math.tan(delta))
        return (v * math.cos(psi + beta),
                v * math.sin(psi + beta),
                v * math.cos(beta) * math.tan(delta) / p.wheelbase,
                a_net)

    def step(self, state: VehicleState, command: ControlCommand, dt: float) -> VehicleState:
        p = self.params
        act = self.actuators.apply(command, state.steering_angle, dt)
        self.last_actuated = act
        delta = act.steering_angle
        a_net = act.net_acceleration

        # a vehicle at rest cannot be braked into reverse
        if state.longitudinal_velocity <= 0.0 and a_net < 0.0:
            a_net = 0.0

        x, y, psi, v = state.x, state.y, state.yaw, state.longitudinal_velocity
        if self.integrator == "euler":
            dx, dy, dpsi, dv = self._derivatives(x, y, psi, v, delta, a_net)
            x, y, psi, v = x + dx * dt, y + dy * dt, psi + dpsi * dt, v + dv * dt
        else:
            k1 = self._derivatives(x, y, psi, v, delta, a_net)
            k2 = self._derivatives(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], psi + 0.5 * dt * k1[2], v + 0.5 * dt * k1[3], delta, a_net)
            k3 = self._derivatives(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], psi + 0.5 * dt * k2[2], v + 0.5 * dt * k2[3], delta, a_net)
            k4 = self._derivatives(x + dt * k3[0], y + dt * k3[1], psi + dt * k3[2], v + dt * k3[3], delta, a_net)
            x += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            psi += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            v += dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        v = min(max(v, 0.0), p.max_speed)
        beta = math.atan(p.cg_to_rear_axle / p.wheelbase * math.tan(delta))
        yaw_rate = v * math.cos(beta) * math.tan(delta) / p.wheelbase
        realised_acc = (v - state.longitudinal_velocity) / dt if dt > 0 else 0.0

        return VehicleState(
            timestamp=state.timestamp + dt,
            x=x, y=y, yaw=float(wrap_angle(psi)),
            longitudinal_velocity=v,
            lateral_velocity=v * math.sin(beta),
            yaw_rate=yaw_rate,
            longitudinal_acceleration=realised_acc,
            steering_angle=delta,
        )
