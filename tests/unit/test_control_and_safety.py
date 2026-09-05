import math

import numpy as np
import pytest

from autonomy.control.lateral import StanleyController
from autonomy.control.longitudinal import PIDSpeedController
from autonomy.control.tracker import TrajectoryTracker
from autonomy.core.config import AutonomyConfig, load_vehicle_parameters
from autonomy.core.types import (CandidateTrajectory, ControlCommand, ObjectType, PlannerOutput,
                                 RiskAssessment, RiskLevel, RiskSummary, Trajectory, VehicleState)
from autonomy.safety.supervisor import SafetySupervisor
from autonomy.vehicle.models import KinematicBicycleModel


@pytest.fixture(scope="module")
def cfg():
    return AutonomyConfig.load()


@pytest.fixture(scope="module")
def params():
    return load_vehicle_parameters()


def straight_traj(y=0.0, v=8.0, length=200.0, dt=0.1, a=0.0):
    t = np.arange(0.0, length / max(v, 1.0), dt)
    x = v * t
    return Trajectory(t, x, np.full_like(t, y), np.zeros_like(t), np.full_like(t, v),
                      np.zeros_like(t), np.full_like(t, a), id="ref")


# -------------------------------------------------------------- lateral --
def test_stanley_steers_toward_path(cfg, params):
    ctrl = StanleyController(cfg.control.stanley, params)
    traj = straight_traj(y=0.0)
    left_of_path = VehicleState(0.0, 5.0, 1.0, 0.0, 8.0)     # vehicle is left (+y) of the path
    right_of_path = VehicleState(0.0, 5.0, -1.0, 0.0, 8.0)
    d_left, dbg = ctrl.compute(left_of_path, traj)
    d_right, _ = ctrl.compute(right_of_path, traj)
    assert d_left < 0 < d_right            # steer right when left of path, and vice versa
    assert dbg.cross_track_error < 0        # path is to the right of the vehicle
    assert dbg.target_x >= 5.0


def test_stanley_converges_in_closed_loop(cfg, params):
    ctrl = StanleyController(cfg.control.stanley, params)
    model = KinematicBicycleModel(params)
    traj = straight_traj(y=0.0, v=8.0)
    s = VehicleState(0.0, 0.0, 1.5, math.radians(10), 8.0)
    errors = []
    for _ in range(400):                         # 8 s
        delta, dbg = ctrl.compute(s, traj)
        s = model.step(s, ControlCommand(s.timestamp, delta, 0.0, 0.0), 0.02)
        errors.append(abs(s.y))
    assert errors[-1] < 0.05
    assert abs(s.yaw) < math.radians(1.0)
    assert max(errors[200:]) < 0.2                # no sustained oscillation


def test_stanley_respects_steering_limit(cfg, params):
    ctrl = StanleyController(cfg.control.stanley, params)
    s = VehicleState(0.0, 0.0, 6.0, math.radians(90), 2.0)
    delta, _ = ctrl.compute(s, straight_traj())
    assert abs(delta) <= params.max_steering_angle + 1e-12


# --------------------------------------------------------- longitudinal --
def test_pid_converges_to_target_speed(cfg, params):
    pid = PIDSpeedController(cfg.control.pid, params, cfg.control.stop_speed_mps)
    model = KinematicBicycleModel(params)
    traj = straight_traj(v=8.0)
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 3.0)
    for _ in range(400):
        acc, brake, dbg = pid.compute(s, traj, s.timestamp, 0.02)
        assert acc >= 0 and brake >= 0 and not (acc > 0 and brake > 0)
        s = model.step(s, ControlCommand(s.timestamp, 0.0, acc, brake), 0.02)
    assert s.longitudinal_velocity == pytest.approx(8.0, abs=0.1)


def test_pid_brakes_when_too_fast_and_holds_when_stopped(cfg, params):
    pid = PIDSpeedController(cfg.control.pid, params, cfg.control.stop_speed_mps)
    traj = straight_traj(v=2.0)
    acc, brake, _ = pid.compute(VehicleState(0.0, 0, 0, 0, 6.0), traj, 0.0, 0.02)
    assert acc == 0.0 and brake > 0
    stop_traj = straight_traj(v=0.0)
    acc, brake, _ = pid.compute(VehicleState(0.0, 0, 0, 0, 0.0), stop_traj, 0.0, 0.02)
    assert acc == 0.0 and brake > 0


def test_pid_uses_feedforward(cfg, params):
    pid = PIDSpeedController(cfg.control.pid, params, cfg.control.stop_speed_mps)
    traj = straight_traj(v=5.0, a=2.0)
    acc, brake, dbg = pid.compute(VehicleState(0.0, 0, 0, 0, 5.0), traj, 0.0, 0.02)
    assert dbg.feedforward == pytest.approx(2.0) and acc == pytest.approx(2.0, abs=1e-6)


def test_tracker_never_touches_vehicle_state(cfg, params):
    tracker = TrajectoryTracker(cfg.control, params)
    s = VehicleState(0.0, 1.0, 2.0, 0.1, 5.0)
    cmd = tracker.track(s, straight_traj(), 0.0, 0.02)
    assert isinstance(cmd, ControlCommand) and cmd.source == "tracker"
    assert (s.x, s.y, s.yaw, s.longitudinal_velocity) == (1.0, 2.0, 0.1, 5.0)


# --------------------------------------------------------------- safety --
def summary(ttc_phys=math.inf, level=RiskLevel.NONE, score=0.0):
    a = RiskAssessment("o", ObjectType.CATTLE, 10.0, 5.0, 5.0, math.inf, math.inf, 5.0, 1.0,
                       False, 0.0, score, level, 0.3, 1.0)
    s = RiskSummary([a], level, score, math.inf, 5.0, "o", False)
    s.min_ttc_current_speed = ttc_phys
    return s


def fake_plan(fallback=False):
    tr = straight_traj()
    cand = CandidateTrajectory("c", "c", tr, 0.0, 8.0, feasible=not fallback, fallback=fallback)
    return PlannerOutput(0.0, [cand], cand, 1.0, 0 if fallback else 1, 1 if fallback else 0, {}, (0, 0, 0))


def test_safety_overrides_on_critical_ttc_and_records(cfg):
    sup = SafetySupervisor(cfg.safety)
    ego = VehicleState(0.0, 0, 0, 0, 10.0)
    nominal = ControlCommand(0.0, 0.1, 2.0, 0.0)
    cmd, status = sup.check(nominal, summary(ttc_phys=0.6), fake_plan(), ego, 0.0)
    assert status.override_active and cmd.source == "safety"
    assert cmd.brake == cfg.safety.brake_mps2 and cmd.acceleration == 0.0
    assert cmd.steering_angle == 0.1                     # steering is kept
    assert status.activation_count == 1 and "TTC" in status.reason


def test_safety_holds_then_releases(cfg):
    sup = SafetySupervisor(cfg.safety)
    ego = VehicleState(0.0, 0, 0, 0, 10.0)
    nominal = ControlCommand(0.0, 0.0, 1.0, 0.0)
    sup.check(nominal, summary(ttc_phys=0.5), fake_plan(), ego, 0.0)
    cmd, st = sup.check(nominal, summary(), fake_plan(), ego, cfg.safety.hold_time_s * 0.5)
    assert st.override_active and cmd.source == "safety"
    cmd, st = sup.check(nominal, summary(), fake_plan(), ego, cfg.safety.hold_time_s + 0.01)
    assert not st.override_active and cmd is nominal
    assert st.activation_count == 1 and sup.activations[0].released_at is not None


def test_safety_triggers_on_planner_fallback_and_critical_level(cfg):
    sup = SafetySupervisor(cfg.safety)
    ego = VehicleState(0.0, 0, 0, 0, 10.0)
    nominal = ControlCommand(0.0, 0.0, 1.0, 0.0)
    _, st = sup.check(nominal, summary(), fake_plan(fallback=True), ego, 0.0)
    assert st.override_active and "fallback" in st.reason
    sup2 = SafetySupervisor(cfg.safety)
    _, st2 = sup2.check(nominal, summary(level=RiskLevel.CRITICAL, score=0.95), fake_plan(), ego, 0.0)
    assert st2.override_active and "critical" in st2.reason
    # a stationary vehicle is never emergency-braked
    sup3 = SafetySupervisor(cfg.safety)
    _, st3 = sup3.check(nominal, summary(level=RiskLevel.CRITICAL, score=0.95), fake_plan(fallback=True),
                        VehicleState(0.0, 0, 0, 0, 0.0), 0.0)
    assert not st3.override_active


def test_safety_does_not_trigger_when_clear(cfg):
    sup = SafetySupervisor(cfg.safety)
    nominal = ControlCommand(0.0, 0.0, 1.0, 0.0)
    cmd, st = sup.check(nominal, summary(ttc_phys=3.0, level=RiskLevel.HIGH, score=0.5), fake_plan(),
                        VehicleState(0.0, 0, 0, 0, 10.0), 0.0)
    assert not st.override_active and cmd is nominal
