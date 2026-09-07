"""Phase 5: curved reverse manoeuvring, and the emergency-brake behaviour that was left alone.

The reverse tests drive the real generator, the real collision checker, the real controller and the
real vehicle model. Nothing here asserts that a function was called; each one asserts a physical
property of the trajectory or of the motion that results from it.

The emergency-brake tests exist to PROTECT the current behaviour. The Phase 5 audit measured all
four brake events in the suite and classified them as one necessary and three conservative, with no
false positives and no oscillation, so this phase changed nothing about braking. These tests pin
that: they fail if someone later weakens the override to make a count look better.
"""
import math

import numpy as np
import pytest

from autonomy.core.config import AutonomyConfig, ObjectProfiles, PerceptionConfig, load_vehicle_parameters
from autonomy.core.types import (ControlCommand, ObjectPrediction, ObjectState, ObjectType,
                                 SpeedPolicy, VehicleState)
from autonomy.control.lateral import StanleyController
from autonomy.planning.candidate_generator import CandidateGenerator
from autonomy.planning.collision_checker import CollisionChecker
from autonomy.vehicle.actuators import ActuatorModel
from autonomy.vehicle.models import KinematicBicycleModel
from simulation.world.road import DrivableSpace


@pytest.fixture(scope="module")
def cfg():
    return AutonomyConfig.load()


@pytest.fixture(scope="module")
def params():
    return load_vehicle_parameters()


@pytest.fixture
def road():
    return DrivableSpace.straight(200.0, 3.5, 3.5, x0=-20.0)


def reverse_candidates(cfg, params, road, ego, desired=1.75):
    gen = CandidateGenerator(cfg.planning, params, road)
    cands, fr = gen.generate(ego, SpeedPolicy(4.0, True, False, allow_reverse=True), desired, 0.0)
    return [c for c in cands if c.id.startswith("reverse_")], gen, fr


# ----------------------------------------------------- 1. reverse candidate generation -------- #
def test_reverse_candidates_now_span_several_lateral_offsets(cfg, params, road):
    """Before Phase 5 every reverse candidate ended at the CURRENT offset: straight back, always."""
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, gen, fr = reverse_candidates(cfg, params, road, ego)
    ends = sorted({round(c.lateral_offset_end, 3) for c in rev})
    assert len(ends) >= 3, f"expected a fan of reverse end offsets, got {ends}"
    assert any(e < fr.d0 - 0.5 for e in ends) and any(e > fr.d0 + 0.5 for e in ends)
    assert any(abs(e - fr.d0) < 1e-6 for e in ends), "the straight leg must still be offered"


def test_a_curved_reverse_actually_moves_sideways(cfg, params, road):
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, _, fr = reverse_candidates(cfg, params, road, ego)
    curved = max(rev, key=lambda c: abs(c.lateral_offset_end - fr.d0))
    tr = curved.trajectory
    assert tr.x[-1] < tr.x[0], "reverse must travel backwards"
    assert abs(tr.y[-1] - fr.d0) > 0.3, "the curved leg did not move laterally at all"
    assert np.all(np.isfinite(tr.x)) and np.all(np.isfinite(tr.y))


def test_the_reverse_body_yaw_leans_the_opposite_way_to_a_forward_shift(cfg, params, road):
    """Reversing does not turn the car around, so the yaw offset has the opposite sign.

    This is the sign that makes the difference between backing into the gap and backing away from
    it, and it is the one a refactor is most likely to get wrong.
    """
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    gen = CandidateGenerator(cfg.planning, params, road)
    fr = gen.frame(ego)
    back = gen.build_reverse(ego, fr, 6.0, 0.0, fr.d0 - 1.0)
    fwd = gen.build(ego, fr, fr.d0 - 1.0, 2.0, 2.0, 0.0, "fwd")
    i = len(back.trajectory) // 2
    assert back.trajectory.yaw[i] > fr.h_ref, "reverse to the right must yaw left"
    assert fwd.trajectory.yaw[i] < fr.h_ref, "forward to the right must yaw right"


# ------------------------------------------------------------ 2/3. speed and acceleration ----- #
def test_reverse_speed_stays_inside_the_configured_bound(cfg, params, road):
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, _, _ = reverse_candidates(cfg, params, road, ego)
    for c in rev:
        v = c.trajectory.velocity
        assert v.max() <= 1e-9, "a reverse candidate must never command forward motion"
        assert abs(v.min()) <= cfg.planning.reverse_speed_mps + 1e-9
        assert abs(v.min()) <= params.max_reverse_speed + 1e-9


def test_reverse_acceleration_stays_inside_the_configured_bound(cfg, params, road):
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, _, _ = reverse_candidates(cfg, params, road, ego)
    for c in rev:
        a = np.abs(c.trajectory.acceleration)
        assert a.max() <= cfg.planning.reverse_acceleration_mps2 + 1e-6, c.id


# ------------------------------------------------------------ 4/5. steering and its rate ------ #
def test_curved_reverse_curvature_respects_the_steering_limit(cfg, params, road):
    """tan(delta) = L * kappa holds in reverse exactly as it does forwards, so the same bound applies."""
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, _, _ = reverse_candidates(cfg, params, road, ego)
    for c in rev:
        k = np.abs(c.trajectory.curvature)
        assert k.max() <= params.max_curvature + 1e-6, f"{c.id} peaks at kappa {k.max():.3f}"
        delta = np.arctan(params.wheelbase * k)
        assert delta.max() <= params.max_steering_angle + 1e-6


def test_the_reverse_cross_track_contribution_is_clamped(cfg, params):
    """A huge lateral error must not produce an unbounded steering command while reversing."""
    stan = StanleyController(cfg.control.stanley, params)
    from autonomy.core.types import Trajectory
    n = 20
    t = np.linspace(0.0, 2.0, n)
    traj = Trajectory(t=t, x=np.linspace(0.0, -3.0, n), y=np.full(n, 8.0),   # path 8 m to the side
                      yaw=np.zeros(n), velocity=np.full(n, -1.0),
                      curvature=np.zeros(n), acceleration=np.zeros(n), id="rev")
    ego = VehicleState(0.0, 0.0, 0.0, 0.0, -1.0)
    delta, dbg = stan.compute(ego, traj)
    assert abs(delta) <= params.max_steering_angle + 1e-9
    # heading error is zero here, so the whole command is the clamped cross-track term
    assert abs(delta) <= cfg.control.stanley.reverse_crosstrack_limit_rad + 1e-9


def test_the_steering_rate_limit_still_binds_while_reversing(cfg, params):
    act = ActuatorModel(params)
    dt = 0.02
    out = act.apply(ControlCommand(0.0, params.max_steering_angle, 0.0, 0.0, reverse=True), 0.0, dt)
    assert abs(out.steering_angle) <= params.max_steering_rate * dt + 1e-12
    assert out.steering_rate_saturated


# ------------------------------------------------------------ 6/7. collision checking --------- #
def _blocker(x, y, t0=0.0):
    """A stationary pushcart, predicted to stay put for the whole horizon."""
    n = 41
    return ObjectPrediction(
        object_id="blk", object_type=ObjectType.PUSHCART,
        times=t0 + np.linspace(0.0, 4.0, n), x=np.full(n, float(x)), y=np.full(n, float(y)),
        heading=np.zeros(n), vx=np.zeros(n), vy=np.zeros(n),
        covariances=np.tile(np.eye(2) * 0.01, (n, 1, 1)),
        length=2.0, width=1.0, risk_weight=1.2, meas_sigma=0.05)


def test_a_reverse_leg_into_an_obstacle_is_rejected(cfg, params, road):
    """Reverse candidates go through the SAME swept-footprint check as forward ones."""
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, gen, fr = reverse_candidates(cfg, params, road, ego)
    check = CollisionChecker(cfg.planning, params, road)
    behind = _blocker(36.0, 1.75)                       # squarely in the reversing path
    check.check_all(rev, [behind])
    assert sum(c.feasible for c in rev) < len(rev), "an obstacle directly behind rejected nothing"
    rejected = [c for c in rev if not c.feasible]
    from autonomy.core.types import RejectionReason
    assert rejected and all(c.rejection_reason is not RejectionReason.NONE for c in rejected)


def test_a_clear_reverse_leg_survives_the_check(cfg, params, road):
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, gen, fr = reverse_candidates(cfg, params, road, ego)
    check = CollisionChecker(cfg.planning, params, road)
    check.check_all(rev, [_blocker(80.0, -3.0)])                 # far away and to the side
    assert sum(c.feasible for c in rev) > 0


def test_no_reverse_leg_crosses_an_obstacle_without_being_seen(cfg, params, road):
    """Sample the footprint along every ACCEPTED reverse leg and confirm it never overlaps."""
    ego = VehicleState(0.0, 40.0, 1.75, 0.0, 0.0)
    rev, gen, fr = reverse_candidates(cfg, params, road, ego)
    check = CollisionChecker(cfg.planning, params, road)
    blocker = _blocker(37.0, 1.0)
    check.check_all(rev, [blocker])
    half_l, half_w = 0.5 * params.length, 0.5 * params.width
    for c in rev:
        if not c.feasible:
            continue
        tr = c.trajectory
        for k in range(len(tr)):
            dx, dy = tr.x[k] - blocker.x[0], tr.y[k] - blocker.y[0]
            c_, s_ = math.cos(tr.yaw[k]), math.sin(tr.yaw[k])
            lx, ly = abs(dx * c_ + dy * s_), abs(-dx * s_ + dy * c_)
            assert lx > half_l + 1.0 or ly > half_w + 0.5, \
                f"{c.id} was accepted yet its footprint sits on the blocker at sample {k}"


# ------------------------------------------------------ 8/9. forward <-> reverse transitions -- #
def test_the_vehicle_model_actually_reverses_and_curves(params):
    """Motion must come from the shipped bicycle model, with yaw rate following signed speed."""
    m = KinematicBicycleModel(params)
    s = VehicleState(0.0, 0.0, 0.0, 0.0, 0.0)
    yaw0 = s.yaw
    for _ in range(60):
        s = m.step(s, ControlCommand(0.0, 0.25, 1.0, 0.0, reverse=True), 0.05)
    assert s.longitudinal_velocity < 0.0
    assert s.x < 0.0, "the vehicle did not move backwards"
    assert s.yaw != yaw0, "steering while reversing produced no rotation"
    assert abs(s.longitudinal_velocity) <= params.max_reverse_speed + 1e-9


def test_reverse_to_forward_transition_passes_through_zero(params):
    """No teleport and no sign jump: the vehicle must decelerate through rest to go forward again."""
    m = KinematicBicycleModel(params)
    s = VehicleState(0.0, 0.0, 0.0, 0.0, -1.0)
    speeds = []
    for _ in range(80):
        s = m.step(s, ControlCommand(0.0, 0.0, 1.5, 0.0), 0.05)
        speeds.append(s.longitudinal_velocity)
    assert speeds[0] < 0.0 and speeds[-1] > 0.0
    assert min(abs(v) for v in speeds) < 0.2, "the speed jumped across zero"
    jumps = [abs(b - a) for a, b in zip(speeds, speeds[1:])]
    assert max(jumps) < 0.5, "velocity discontinuity: the vehicle teleported through the transition"


def test_forward_to_reverse_needs_a_near_standstill(cfg, params, road):
    """Reverse candidates are not offered while the ego is still rolling forward."""
    rolling = VehicleState(0.0, 40.0, 1.75, 0.0, 3.0)
    rev, _, _ = reverse_candidates(cfg, params, road, rolling)
    assert not rev


# ---------------------------------------------------------- 10. deterministic recovery -------- #
@pytest.mark.parametrize("mode", ["ground_truth", "sensors"])
def test_boxed_in_recovery_reverses_and_still_reaches_the_goal(mode):
    """The whole point of Phase 5's scenario: reverse is REQUIRED in both perception modes.

    Slow, so it is one test per mode rather than a battery.
    """
    from simulation.runner import run_scenario
    r = run_scenario("NARROW_LANE_REVERSE_RECOVERY", log_dir=None, console=False,
                     perception=PerceptionConfig.load().with_mode(mode))
    m = r.metrics
    assert m.collision_count == 0
    assert m.termination_reason == "GOAL_REACHED"
    assert m.reverse_manoeuvres >= 1, "the scenario no longer requires reversing"
    assert 0.0 < m.reverse_distance_m < 20.0
    states = [f.decision.state.name for f in r.frames if f.decision]
    assert "REVERSING" in states
    last_rev = max(i for i, s in enumerate(states) if s == "REVERSING")
    assert any(s not in ("REVERSING", "STOPPED") for s in states[last_rev + 1:]), \
        "the ego never returned to forward motion after reversing"


# =============================================================================================== #
# Emergency braking. Phase 5 changed NOTHING here; these tests pin the audited behaviour.
# =============================================================================================== #
def test_a_genuine_emergency_still_brakes():
    """The pedestrian dart fires the override at 0.7 s time-to-collision with zero feasible plans."""
    from simulation.runner import run_scenario
    r = run_scenario("SUDDEN_PEDESTRIAN_DART", log_dir=None, console=False,
                     perception=PerceptionConfig.load().with_mode("sensors"))
    assert r.metrics.emergency_brake_activations >= 1
    assert r.metrics.collision_count == 0
    overrides = [f for f in r.frames if f.safety.override_active]
    assert overrides, "the safety override never fired in a scenario that needs it"
    assert all(f.control.source == "safety" for f in overrides)
    assert all(f.control.brake > 0 and f.control.acceleration == 0 for f in overrides)


def test_the_safety_supervisor_outranks_the_tracker():
    """Whenever the override is active, the command the vehicle receives is the supervisor's."""
    from simulation.runner import run_scenario
    r = run_scenario("DENSE_MARKET_MIXED_TRAFFIC", log_dir=None, console=False,
                     perception=PerceptionConfig.load().with_mode("sensors"))
    overrides = [f for f in r.frames if f.safety.override_active]
    assert overrides
    for f in overrides:
        assert f.control.source == "safety"
        assert f.control.brake > 0.0
        assert f.control.acceleration == 0.0
        assert f.safety.reason, "an override with no recorded reason is not auditable"


def test_braking_does_not_oscillate_under_sensor_noise():
    """The audit found no chattering; this fails if a future change introduces some.

    An override EPISODE is a contiguous run of active frames. Rapid release-and-reapply would show
    up as many episodes separated by a handful of frames. The measured separations were 1.8 s and
    6.6 s, so a 0.5 s floor has real headroom while still catching genuine chatter.
    """
    from simulation.runner import run_scenario
    cfg = AutonomyConfig.load()
    for scen in ("DENSE_MARKET_MIXED_TRAFFIC", "SUDDEN_PEDESTRIAN_DART"):
        r = run_scenario(scen, log_dir=None, console=False,
                         perception=PerceptionConfig.load().with_mode("sensors"))
        starts = [i for i, f in enumerate(r.frames)
                  if f.safety.override_active and (i == 0 or not r.frames[i - 1].safety.override_active)]
        gaps = [(b - a) * cfg.simulation.dt_s for a, b in zip(starts, starts[1:])]
        assert all(g >= 0.5 for g in gaps), f"{scen}: override re-fired after {min(gaps):.2f} s"


def test_the_brake_is_released_once_the_hazard_clears():
    """A brake that never releases is as much a failure as one that never fires."""
    from simulation.runner import run_scenario
    r = run_scenario("SUDDEN_PEDESTRIAN_DART", log_dir=None, console=False,
                     perception=PerceptionConfig.load().with_mode("sensors"))
    assert not r.frames[-1].safety.override_active
    assert r.metrics.termination_reason == "GOAL_REACHED"
    assert r.frames[-1].ego.longitudinal_velocity > 0.5, "the ego never resumed after braking"


def test_noisy_perception_does_not_make_the_quiet_scenarios_brake():
    """Sensor noise alone must not manufacture emergencies where there is no hazard."""
    from simulation.runner import run_scenario
    for scen in ("UNMARKED_VILLAGE_ROAD", "MIXED_TRAFFIC_CURVE"):
        m = run_scenario(scen, log_dir=None, console=False,
                         perception=PerceptionConfig.load().with_mode("sensors")).metrics
        assert m.emergency_brake_activations == 0, f"{scen} braked with no hazard present"
        assert m.collision_count == 0
