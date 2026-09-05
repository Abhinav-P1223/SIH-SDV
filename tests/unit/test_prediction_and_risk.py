import math

import numpy as np
import pytest

from autonomy.core.config import AutonomyConfig, ObjectProfiles, load_vehicle_parameters
from autonomy.core.types import ObjectState, ObjectType, RiskLevel, VehicleState
from autonomy.prediction.predictor import ConstantVelocityPredictor
from autonomy.risk.risk_engine import RiskEngine, first_overlap_time, kinematic_ttc
from simulation.world.road import DrivableSpace


@pytest.fixture(scope="module")
def cfg():
    return AutonomyConfig.load()


@pytest.fixture(scope="module")
def profiles():
    return ObjectProfiles.load()


@pytest.fixture(scope="module")
def params():
    return load_vehicle_parameters()


def obj(id_, x, y, vx, vy, otype=ObjectType.CATTLE, heading=None):
    h = math.atan2(vy, vx) if heading is None else heading
    return ObjectState(id_, otype, 0.0, x, y, vx, vy, h, 2.0, 0.7)


# ------------------------------------------------------------- prediction --
def test_prediction_is_time_indexed_and_constant_velocity(cfg, profiles):
    pred = ConstantVelocityPredictor(cfg.prediction, profiles).predict([obj("c", 10, 0, 1.0, -0.5)], now=3.0)[0]
    assert pred.times[0] == 3.0 and pred.times[-1] == pytest.approx(3.0 + cfg.prediction.horizon_s)
    assert len(pred.times) == cfg.prediction.steps
    k = 10
    assert pred.x[k] == pytest.approx(10 + 1.0 * (pred.times[k] - 3.0))
    assert pred.y[k] == pytest.approx(0 - 0.5 * (pred.times[k] - 3.0))


def test_prediction_uncertainty_grows_monotonically(cfg, profiles):
    pred = ConstantVelocityPredictor(cfg.prediction, profiles).predict([obj("c", 0, 0, 1, 0)], now=0.0)[0]
    s = pred.sigma()
    assert np.all(np.diff(s) > 0)
    prof = profiles.get(ObjectType.CATTLE)
    assert s[0] == pytest.approx(prof.sigma_pos0_m)
    assert pred.risk_weight == prof.risk_weight


def test_prediction_uses_profile_by_type(cfg, profiles):
    p = ConstantVelocityPredictor(cfg.prediction, profiles)
    ped = p.predict([obj("p", 0, 0, 1, 0, ObjectType.PEDESTRIAN)], 0.0)[0]
    truck = p.predict([obj("t", 0, 0, 1, 0, ObjectType.TRUCK)], 0.0)[0]
    assert ped.sigma()[-1] > truck.sigma()[-1]


# -------------------------------------------------------------------- ttc --
def test_kinematic_ttc():
    assert kinematic_ttc(20.0, 10.0) == pytest.approx(2.0)
    assert kinematic_ttc(20.0, 0.0) == math.inf
    assert kinematic_ttc(20.0, -3.0) == math.inf


def test_first_overlap_time():
    t = np.linspace(0, 4, 41)
    d = np.maximum(5.0 - 2.0 * t, 0.0)
    assert first_overlap_time(t, d, 0.5) == pytest.approx(2.3, abs=0.051)
    assert first_overlap_time(t, d + 10, 0.5) == math.inf


# ------------------------------------------------------------------ risk --
def make_engine(cfg, params):
    return RiskEngine(cfg.risk, cfg.behavior, params)


def test_head_on_static_object_produces_finite_ttc_and_high_risk(cfg, profiles, params):
    ego = VehicleState(0.0, 0.0, 0.0, 0.0, 10.0)
    o = obj("c", 30.0, 0.0, 0.0, 0.0, heading=math.pi / 2)
    preds = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], 0.0)
    summary = make_engine(cfg, params).evaluate(ego, None, [o], preds)
    a = summary.assessments[0]
    # gap = 30 - (front bumper ~ 2.0 m ahead of CG) - half cattle width 0.35 -> ~27.65 m; closing 10 m/s
    assert a.ttc_kinematic == pytest.approx(a.distance / 10.0)
    assert 2.4 < a.ttc < 3.0
    assert a.trajectory_intersection
    assert a.risk_level.value >= RiskLevel.HIGH.value
    assert summary.min_ttc == a.ttc


def test_far_lateral_object_is_low_risk(cfg, profiles, params):
    ego = VehicleState(0.0, 0.0, 0.0, 0.0, 10.0)
    o = obj("c", 30.0, 6.0, 0.0, 0.0, heading=0.0)
    preds = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], 0.0)
    a = make_engine(cfg, params).evaluate(ego, None, [o], preds).assessments[0]
    assert a.ttc == math.inf
    assert not a.trajectory_intersection
    assert a.risk_level.value <= RiskLevel.LOW.value
    assert a.min_predicted_distance > 4.0


def test_crossing_object_risk_rises_as_it_enters_path(cfg, profiles, params):
    ego = VehicleState(0.0, 0.0, 0.0, 0.0, 10.0)
    predictor = ConstantVelocityPredictor(cfg.prediction, profiles)
    engine = make_engine(cfg, params)
    scores = []
    for y in (9.0, 7.0, 5.5):
        o = obj("c", 30.0, y, 0.0, -0.8)
        s = engine.evaluate(ego, None, [o], predictor.predict([o], 0.0)).assessments[0]
        scores.append(s.risk_score)
    assert scores[0] < scores[1] < scores[2]
    # once the predicted paths intersect the score is governed by time-to-collision
    o = obj("c", 30.0, 1.0, 0.0, -0.8)
    s = engine.evaluate(ego, None, [o], predictor.predict([o], 0.0)).assessments[0]
    assert s.trajectory_intersection and s.risk_score >= scores[-1]


def test_risk_weight_by_type(cfg, profiles, params):
    ego = VehicleState(0.0, 0.0, 0.0, 0.0, 10.0)
    predictor = ConstantVelocityPredictor(cfg.prediction, profiles)
    engine = make_engine(cfg, params)
    out = []
    for otype in (ObjectType.PEDESTRIAN, ObjectType.TRUCK):
        o = ObjectState("o", otype, 0.0, 30.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.7)
        out.append(engine.evaluate(ego, None, [o], predictor.predict([o], 0.0)).assessments[0].risk_score)
    assert out[0] > out[1]


def test_lead_object_detection(cfg, profiles, params):
    road = DrivableSpace.straight(200, 3.5, 3.5)
    ego = VehicleState(0.0, 0.0, 1.75, 0.0, 10.0)
    lead = ObjectState("lead", ObjectType.CAR, 0.0, 25.0, 1.75, 6.0, 0.0, 0.0, 4.3, 1.8)
    oncoming = ObjectState("onc", ObjectType.CAR, 0.0, 40.0, -1.75, -6.0, 0.0, math.pi, 4.3, 1.8)
    objs = [lead, oncoming]
    preds = ConstantVelocityPredictor(cfg.prediction, profiles).predict(objs, 0.0)
    summary = make_engine(cfg, params).evaluate(ego, None, objs, preds, road)
    assert summary.lead_object_id == "lead"
    assert summary.lead_speed == pytest.approx(6.0)
    assert summary.lead_gap == pytest.approx(25.0 - 0.5 * (params.length + 4.3))
