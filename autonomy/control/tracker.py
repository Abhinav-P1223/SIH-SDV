"""Trajectory tracker: combines lateral and longitudinal controllers.

The tracker is the ONLY producer of nominal ControlCommands. It never modifies
the vehicle state; the command goes through the safety supervisor and then the
vehicle model's actuator limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from autonomy.core.config import ControlConfig
from autonomy.core.types import ControlCommand, Trajectory, VehicleParameters, VehicleState

from .lateral import LateralController, LateralDebug, StanleyController
from .longitudinal import LongitudinalController, LongitudinalDebug, PIDSpeedController


@dataclass
class TrackerDebug:
    lateral: LateralDebug
    longitudinal: LongitudinalDebug

    def to_dict(self) -> dict[str, Any]:
        return {"lateral": asdict(self.lateral), "longitudinal": asdict(self.longitudinal)}


class TrajectoryTracker:
    def __init__(self, cfg: ControlConfig, params: VehicleParameters,
                 lateral: LateralController | None = None,
                 longitudinal: LongitudinalController | None = None):
        self.lateral = lateral or StanleyController(cfg.stanley, params)
        self.longitudinal = longitudinal or PIDSpeedController(cfg.pid, params, cfg.stop_speed_mps)
        self.last_debug: TrackerDebug | None = None

    def reset(self) -> None:
        self.lateral.reset()
        self.longitudinal.reset()

    def track(self, ego: VehicleState, traj: Trajectory, now: float, dt: float) -> ControlCommand:
        steer, ldbg = self.lateral.compute(ego, traj)
        acc, brake, gdbg = self.longitudinal.compute(ego, traj, now, dt)
        self.last_debug = TrackerDebug(ldbg, gdbg)
        return ControlCommand(now, steer, acc, brake, source="tracker")
