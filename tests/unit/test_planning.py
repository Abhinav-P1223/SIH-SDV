import math

import numpy as np
import pytest

from autonomy.core.config import AutonomyConfig, ObjectProfiles, load_vehicle_parameters
from autonomy.core.types import (BehaviorDecision, BehaviorState, ObjectState, ObjectType,
                                 RejectionReason, SpeedPolicy, VehicleState)
from autonomy.planning.candidate_generator import CandidateGenerator, eval_quintic, quintic_coefficients
from autonomy.planning.collision_checker import CollisionChecker
from autonomy.planning.planner import Planner
from autonomy.prediction.predictor import ConstantVelocityPredictor
from simulation.world.road import DrivableSpace


@pytest.fixture(scope="module")
def cfg():
    return AutonomyConfig.load()


@pytest.fixture(scope="module")
def params():
    return load_vehicle_parameters()


@pytest.fixture(scope="module")
def profiles():
    return ObjectProfiles.load()


@pytest.fixture
def road():
    return DrivableSpace.straight(200.0, 3.5, 3.5, x0=-20.0)


def decision(target=10.0, lateral=True, stop=False):
    return BehaviorDecision(0.0, BehaviorState.CRUISE, BehaviorState.CRUISE, "test", {},
                            SpeedPolicy(target, lateral, stop))


def ego(x=0.0, y=1.75, v=10.0, yaw=0.0, steer=0.0):
    return VehicleState(0.0, x, y, yaw, v, steering_angle=steer)


# ------------------------------------------------------------ generator --
def test_quintic_boundary_conditions():
    c = quintic_coefficients(1.0, 0.1, 0.02, -0.5, 0.0, 0.0, 20.0)
    d, dd, ddd = eval_quintic(c, np.array([0.0, 20.0]))
    assert d[0] == pytest.approx(1.0) and dd[0] == pytest.approx(0.1) and ddd[0] == pytest.approx(0.02)
    assert d[1] == pytest.approx(-0.5) and dd[1] == pytest.approx(0.0) and ddd[1] == pytest.approx(0.0)


def test_candidates_are_time_indexed_and_start_at_ego(cfg, params, road):
    gen = CandidateGenerator(cfg.planning, params, road)
    cands, fr = gen.generate(ego(), decision().speed_policy, 1.75, now=5.0)
    half = 0.5 * params.width + cfg.planning.boundary_margin_m
    fitting = [o for o in cfg.planning.lateral_offsets_m if -3.5 + half <= 1.75 + o <= 3.5 - half]
    assert len(cands) == len(fitting) * len(set(cfg.planning.speed_fractions) | {0.0}) + 1
    for c in cands:
        tr = c.trajectory
        assert tr.t[0] == 5.0 and len(tr) == cfg.planning.steps
        assert (tr.x[0], tr.y[0]) == pytest.approx((0.0, 1.75), abs=1e-9)
        assert tr.velocity[0] == pytest.approx(10.0)
        assert np.all(tr.velocity >= 0)
    ends = sorted({round(c.lateral_offset_end, 2) for c in cands})
    assert ends == sorted({round(1.75 + o, 2) for o in fitting})


def test_candidates_reach_lateral_target_and_speed(cfg, params, road):
    gen = CandidateGenerator(cfg.planning, params, road)
    cands, _ = gen.generate(ego(), decision().speed_policy, 1.75, now=0.0)
    c = next(c for c in cands if abs(c.lateral_offset_end - 0.75) < 1e-6 and abs(c.target_speed - 6.0) < 1e-6)
    assert c.trajectory.y[-1] == pytest.approx(0.75, abs=1e-6)
    assert c.trajectory.velocity[-1] == pytest.approx(6.0)
    # decelerates at the comfortable rate, never harder
    assert np.min(c.trajectory.acceleration) >= -cfg.planning.comfortable_deceleration_mps2 - 1e-6


def test_force_stop_generates_only_stop_candidates(cfg, params, road):
    gen = CandidateGenerator(cfg.planning, params, road)
    cands, _ = gen.generate(ego(), SpeedPolicy(0.0, False, True), 1.75, now=0.0)
    assert len(cands) == 2 and all(c.target_speed == 0.0 for c in cands)
    hard = next(c for c in cands if "hard" in c.label)
    assert np.min(hard.trajectory.acceleration) == pytest.approx(-params.max_deceleration)


def test_offsets_outside_corridor_are_not_generated(cfg, params):
    narrow = DrivableSpace.straight(200.0, 1.5, 1.5, x0=-20.0)   # 3 m wide: only d=0 fits a 1.8 m car
    gen = CandidateGenerator(cfg.planning, params, narrow)
    cands, _ = gen.generate(ego(y=0.0), decision().speed_policy, 0.0, now=0.0)
    assert {round(c.lateral_offset_end, 2) for c in cands} == {0.0}


# -------------------------------------------------------------- checker --
def test_curvature_rejection(cfg, params, road):
    gen = CandidateGenerator(cfg.planning, params, road)
    chk = CollisionChecker(cfg.planning, params, road)
    # a sharp 4 m lateral shift within 10 m at low speed exceeds the curvature limit
    cand = gen.build(ego(v=2.0), gen.frame(ego(v=2.0)), 1.75 + 4.5, 2.0,
                     cfg.planning.comfortable_deceleration_mps2, 0.0, "sharp")
    cand.trajectory.curvature[:] = params.max_curvature * 1.5
    chk.check(cand, [])
    assert not cand.feasible and cand.rejection_reason == RejectionReason.CURVATURE


def test_lateral_acceleration_rejection(cfg, params, road):
    gen = CandidateGenerator(cfg.planning, params, road)
    chk = CollisionChecker(cfg.planning, params, road)
    cands, _ = gen.generate(ego(v=10.0), decision().speed_policy, 1.75, now=0.0)
    chk.check_all(cands, [])
    fast_wide = [c for c in cands if abs(c.lateral_offset_end - 1.75) >= 2.4 and c.target_speed == 10.0]
    assert fast_wide and all(c.rejection_reason == RejectionReason.LATERAL_ACCEL for c in fast_wide)


def test_boundary_rejection(cfg, params, road):
    gen = CandidateGenerator(cfg.planning, params, road)
    chk = CollisionChecker(cfg.planning, params, road)
    cand = gen.build(ego(v=5.0), gen.frame(ego(v=5.0)), 3.2, 5.0, 3.0, 0.0, "edge")  # body to 4.1 > 3.5
    chk.check(cand, [])
    assert cand.rejection_reason == RejectionReason.BOUNDARY
    assert cand.min_boundary_clearance == 0.0


def test_collision_rejection_records_time_and_object(cfg, params, road, profiles):
    gen = CandidateGenerator(cfg.planning, params, road)
    chk = CollisionChecker(cfg.planning, params, road)
    o = ObjectState("cow", ObjectType.CATTLE, 0.0, 30.0, 1.75, 0.0, 0.0, math.pi / 2, 2.0, 0.7)
    preds = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], 0.0)
    straight = gen.build(ego(), gen.frame(ego()), 1.75, 10.0, 3.0, 0.0, "straight")
    swerve = gen.build(ego(), gen.frame(ego()), -1.25, 6.0, 3.0, 0.0, "swerve")   # 3 m shift while slowing
    stop = gen.build(ego(), gen.frame(ego()), 1.75, 0.0, params.max_deceleration, 0.0, "stop")
    chk.check_all([straight, swerve, stop], preds)
    assert straight.rejection_reason == RejectionReason.COLLISION
    assert "cow" in straight.rejection_detail
    assert 2.0 < straight.collision_time < 3.2
    assert swerve.feasible and swerve.min_clearance > cfg.planning.safety_margin_m
    assert stop.feasible


def test_corridor_clearance_matches_geometry(cfg, params):
    """Frenet fast path vs exact polygon geometry: equal on a straight, close on a curve."""
    from autonomy.core.geometry import box_corners
    for road, tol in ((DrivableSpace.straight(200.0, 3.5, 3.5, x0=-20.0), 1e-6),
                      (DrivableSpace.from_segments([{"straight": 40}, {"arc": {"radius": 60, "angle_deg": 40}},
                                                    {"straight": 40}], 3.5, 3.5, x0=-20.0), 0.08)):
        gen = CandidateGenerator(cfg.planning, params, road)
        chk = CollisionChecker(cfg.planning, params, road)
        e = VehicleState(0.0, 20.0, 1.0, 0.0, 8.0)
        cands, _ = gen.generate(e, decision(target=8.0).speed_policy, 1.0, now=0.0)
        C, N = len(cands), len(cands[0].trajectory)
        fast = chk._corridor_clearance(cands, C, N)
        X = np.stack([c.trajectory.x for c in cands]); Y = np.stack([c.trajectory.y for c in cands])
        YAW = np.stack([c.trajectory.yaw for c in cands])
        off = params.footprint_center_offset
        corners = box_corners((X + off * np.cos(YAW)).ravel(), (Y + off * np.sin(YAW)).ravel(), YAW.ravel(),
                              params.length, params.width)
        exact = road.boundary_clearance(corners).reshape(C, N)
        assert np.max(np.abs(fast - exact)) < tol


# -------------------------------------------------------------- planner --
def make_planner(cfg, params, road):
    return Planner(cfg.planning, cfg.risk, params, road, 10.0, 1.75)


def test_planner_prefers_route_when_clear(cfg, params, road):
    out = make_planner(cfg, params, road).plan(ego(), decision(), [], 0.0)
    assert out.selected.lateral_offset_end == pytest.approx(1.75)
    assert out.selected.target_speed == pytest.approx(10.0)
    assert out.feasible_count > 0 and set(out.selected.costs) == {
        "collision", "uncertainty", "clearance", "smoothness", "curvature", "progress", "boundary", "speed",
        "lateral", "blocked"}


def test_planner_rejects_colliding_candidates_and_explains(cfg, params, road, profiles):
    o = ObjectState("cow", ObjectType.CATTLE, 0.0, 30.0, 1.75, 0.0, 0.0, math.pi / 2, 2.0, 0.7)
    preds = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], 0.0)
    out = make_planner(cfg, params, road).plan(ego(), decision(), preds, 0.0)
    assert out.rejection_histogram.get("PREDICTED_COLLISION", 0) > 0
    assert out.selected.feasible and not out.selected.fallback
    assert out.selected.min_clearance > cfg.planning.safety_margin_m
    # selected has the lowest total cost among feasible candidates
    feasible = [c for c in out.candidates if c.feasible]
    assert out.selected.total_cost == min(c.total_cost for c in feasible)
    # weighted costs sum to the total
    assert sum(out.selected.weighted_costs.values()) == pytest.approx(out.selected.total_cost)


def test_planner_falls_back_to_hard_stop_when_boxed_in(cfg, params, profiles):
    wall_road = DrivableSpace.straight(200.0, 1.5, 1.5, x0=-20.0)      # too narrow to swerve
    o = ObjectState("truck", ObjectType.TRUCK, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 9.0, 2.5)  # right in front
    preds = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], 0.0)
    out = Planner(cfg.planning, cfg.risk, params, wall_road, 10.0, 0.0).plan(ego(y=0.0), decision(), preds, 0.0)
    assert out.feasible_count == 0
    assert out.selected.fallback and out.selected.label.startswith("stop_hard")


def test_scoring_penalises_lateral_deviation_and_low_speed(cfg, params, road):
    out = make_planner(cfg, params, road).plan(ego(), decision(), [], 0.0)
    by = {(round(c.lateral_offset_end, 2), round(c.target_speed, 1)): c for c in out.candidates if c.feasible}
    base = by[(1.75, 10.0)]
    assert by[(0.75, 10.0)].costs["lateral"] > base.costs["lateral"] == 0.0
    assert by[(1.75, 6.0)].costs["speed"] > base.costs["speed"]
    assert by[(1.75, 6.0)].costs["progress"] > base.costs["progress"]
