import math

import pytest

from autonomy.core.config import load_vehicle_parameters
from autonomy.core.types import ControlCommand, VehicleState
from autonomy.vehicle.models import KinematicBicycleModel


@pytest.fixture
def params():
    return load_vehicle_parameters()


@pytest.fixture
def model(params):
    return KinematicBicycleModel(params)


def cmd(steer=0.0, acc=0.0, brake=0.0, t=0.0):
    return ControlCommand(t, steer, acc, brake)


def test_straight_line_constant_speed(model):
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 10.0)
    dt = 0.02
    for _ in range(100):
        s = model.step(s, cmd(), dt)
    assert s.x == pytest.approx(20.0, abs=1e-6)
    assert s.y == pytest.approx(0.0, abs=1e-9)
    assert s.yaw == pytest.approx(0.0)
    assert s.longitudinal_velocity == pytest.approx(10.0)


def test_constant_acceleration(model):
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
    dt = 0.02
    for _ in range(100):  # 2 s at 2 m/s^2
        s = model.step(s, cmd(acc=2.0), dt)
    assert s.longitudinal_velocity == pytest.approx(4.0, abs=1e-6)
    # Euler: x = sum v_k dt slightly above v t /2 -> allow small tolerance
    assert s.x == pytest.approx(4.0, abs=0.1)


def test_steady_state_turn_radius(model, params):
    """With constant steering the CG traces a circle of radius L / (cos(beta) tan(delta))."""
    delta = math.radians(10.0)
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 5.0, steering_angle=delta)
    dt = 0.01
    beta = math.atan(params.cg_to_rear_axle / params.wheelbase * math.tan(delta))
    expected_r = params.wheelbase / (math.cos(beta) * math.tan(delta))
    yaw_rate = None
    for _ in range(200):
        s = model.step(s, cmd(steer=delta), dt)
        yaw_rate = s.yaw_rate
    assert yaw_rate == pytest.approx(5.0 / expected_r, rel=1e-6)


def test_steering_angle_saturation(model, params):
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 5.0)
    for _ in range(500):
        s = model.step(s, cmd(steer=math.radians(80.0)), 0.02)
    assert s.steering_angle == pytest.approx(params.max_steering_angle)


def test_steering_rate_saturation(model, params):
    """A 30 degree step request must be realised gradually at max_steering_rate."""
    dt = 0.02
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 5.0)
    s = model.step(s, cmd(steer=math.radians(30.0)), dt)
    assert s.steering_angle == pytest.approx(params.max_steering_rate * dt)
    assert model.last_actuated.steering_rate_saturated
    n = 0
    while s.steering_angle < math.radians(30.0) - 1e-9 and n < 1000:
        s = model.step(s, cmd(steer=math.radians(30.0)), dt)
        n += 1
    expected_steps = math.ceil(math.radians(30.0) / (params.max_steering_rate * dt))
    assert n + 1 == pytest.approx(expected_steps, abs=1)


def test_acceleration_saturation(model, params):
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 5.0)
    s = model.step(s, cmd(acc=50.0), 0.1)
    assert s.longitudinal_velocity == pytest.approx(5.0 + params.max_acceleration * 0.1)
    assert model.last_actuated.acceleration_saturated


def test_braking_saturation_and_no_reverse(model, params):
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 1.0)
    s = model.step(s, cmd(brake=100.0), 0.1)
    assert model.last_actuated.net_acceleration == pytest.approx(-params.max_deceleration)
    assert s.longitudinal_velocity == pytest.approx(max(0.0, 1.0 - params.max_deceleration * 0.1))
    for _ in range(50):
        s = model.step(s, cmd(brake=100.0), 0.1)
    assert s.longitudinal_velocity == 0.0
    assert s.x >= 0.0


def test_speed_cap(model, params):
    s = VehicleState(0.0, 0.0, 0.0, 0.0, params.max_speed - 0.1)
    for _ in range(100):
        s = model.step(s, cmd(acc=3.0), 0.1)
    assert s.longitudinal_velocity == pytest.approx(params.max_speed)


def test_rk4_matches_euler_for_small_dt(params):
    e = KinematicBicycleModel(params, "euler")
    r = KinematicBicycleModel(params, "rk4")
    se = sr = VehicleState(0.0, 0.0, 0.0, 0.0, 8.0, steering_angle=math.radians(5))
    for _ in range(200):
        se = e.step(se, cmd(steer=math.radians(5), acc=0.5), 0.01)
        sr = r.step(sr, cmd(steer=math.radians(5), acc=0.5), 0.01)
    assert se.x == pytest.approx(sr.x, abs=0.05)
    assert se.y == pytest.approx(sr.y, abs=0.05)


def test_footprint_geometry(model, params):
    s = VehicleState(0.0, 10.0, 5.0, 0.0, 0.0)
    fp = model.footprint(s)
    assert fp.length == params.length and fp.width == params.width
    assert fp.cx == pytest.approx(10.0 + params.footprint_center_offset)
    assert fp.cy == pytest.approx(5.0)
