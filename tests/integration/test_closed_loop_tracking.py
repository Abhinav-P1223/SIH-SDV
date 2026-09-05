"""Planner + tracker + vehicle on an empty corridor: the ego must converge to the
desired lateral offset and speed and hold them through the full closed loop.
"""
import math

import numpy as np
import pytest

from autonomy.core.config import AutonomyConfig, ObjectProfiles, load_vehicle_parameters
from autonomy.core.types import VehicleState
from autonomy.telemetry.telemetry import InMemorySink, TelemetryPublisher
from simulation.runner import Simulation
from simulation.scenarios.loader import build_scenario


def empty_corridor_scenario(ego_y=0.0, ego_yaw_deg=8.0, speed=6.0, desired=10.0, offset=1.75):
    return build_scenario({
        "name": "EMPTY_CORRIDOR", "seed": 1,
        "road": {"reference": [[-20, 0], [200, 0]], "left_boundary": [[-20, 3.5], [200, 3.5]],
                 "right_boundary": [[-20, -3.5], [200, -3.5]]},
        "ego": {"x": 0.0, "y": ego_y, "yaw_deg": ego_yaw_deg, "speed_mps": speed,
                "desired_speed_mps": desired, "desired_lateral_offset_m": offset},
        "goal": {"s_m": 150.0},
        "agents": [],
    })


def test_converges_to_route_and_speed_without_teleporting():
    cfg = AutonomyConfig.load()
    params = load_vehicle_parameters()
    sc = empty_corridor_scenario()
    mem = InMemorySink()
    sim = Simulation(sc, cfg, params, ObjectProfiles.load(), TelemetryPublisher([mem]))
    metrics = sim.run()

    assert metrics.scenario_completed and metrics.collision_count == 0
    frames = mem.frames
    ys = np.array([f.ego.y for f in frames])
    vs = np.array([f.ego.longitudinal_velocity for f in frames])
    # settled after the first 6 s
    settled = [f for f in frames if f.timestamp > 6.0]
    assert all(abs(f.ego.y - 1.75) < 0.15 for f in settled)
    assert all(abs(f.ego.longitudinal_velocity - 10.0) < 0.3 for f in settled)
    # the ego moved continuously: no per-step jump larger than v_max * dt
    dx = np.hypot(np.diff([f.ego.x for f in frames]), np.diff(ys))
    assert np.max(dx) <= params.max_speed * cfg.simulation.dt_s + 1e-9
    # steering rate and acceleration never exceeded actuator limits
    assert metrics.max_steering_rate <= params.max_steering_rate + 1e-9
    assert metrics.max_acceleration <= params.max_acceleration + 1e-6
    assert metrics.max_deceleration <= params.max_deceleration + 1e-6
    # planner ran at the configured rate
    expected_cycles = math.ceil(metrics.time_to_completion / cfg.planning.period_s)
    assert abs(metrics.replanning_count - expected_cycles) <= 1
    assert all(f.decision.state.value == "CRUISE" for f in settled)
