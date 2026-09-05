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
    o = obj("c", 10, 0, 1.0, -0.5)
    o.timestamp = 3.0                                         # fresh measurement
    pred = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], now=3.0)[0]
    assert pred.times[0] == 3.0 and pred.times[-1] == pytest.approx(3.0 + cfg.prediction.horizon_s)
    assert len(pred.times) == cfg.prediction.steps
    k = 10
    assert pred.x[k] == pytest.approx(10 + 1.0 * (pred.times[k] - 3.0))
    assert pred.y[k] == pytest.approx(0 - 0.5 * (pred.times[k] - 3.0))


def test_prediction_compensates_measurement_latency(cfg, profiles):
    """A measurement 0.4 s old is propagated to now before prediction."""
    stale = obj("c", 10, 0, 2.0, 0.0)
    stale.timestamp = 2.6
    pred = ConstantVelocityPredictor(cfg.prediction, profiles).predict([stale], now=3.0)[0]
    assert pred.x[0] == pytest.approx(10 + 2.0 * 0.4)


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
    assert a.risk_level == RiskLevel.HIGH          # CRITICAL is reserved for the physical view
    assert summary.min_ttc == a.ttc
    assert summary.min_ttc_current_speed == pytest.approx(a.ttc)   # current speed == route speed here


def test_critical_only_from_physical_ttc(cfg, profiles, params):
    o = obj("c", 12.0, 0.0, 0.0, 0.0, heading=math.pi / 2)
    preds = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], 0.0)
    engine = make_engine(cfg, params)
    fast = engine.evaluate(VehicleState(0.0, 0.0, 0.0, 0.0, 10.0), None, [o], preds)
    assert fast.max_level == RiskLevel.CRITICAL and fast.min_ttc_current_speed < cfg.risk.ttc_critical_s
    stopped = engine.evaluate(VehicleState(0.0, 0.0, 0.0, 0.0, 0.0), None, [o], preds)
    assert stopped.max_level.value <= RiskLevel.HIGH.value
    assert stopped.min_ttc_current_speed == math.inf


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


def test_road_following_prior_damps_lateral_velocity_for_vehicles_only(cfg, profiles):
    """A tracked car with a noisy lateral velocity must be predicted along the corridor; a cow must not."""
    road = DrivableSpace.straight(200.0, 3.5, 3.5, x0=-20.0)
    p_road = ConstantVelocityPredictor(cfg.prediction, profiles, road)
    p_free = ConstantVelocityPredictor(cfg.prediction, profiles)
    car = ObjectState("car", ObjectType.CAR, 0.0, 50.0, 1.2, -8.0, -0.5, math.atan2(-0.5, -8.0), 4.3, 1.8)
    cow = ObjectState("cow", ObjectType.CATTLE, 0.0, 50.0, 1.2, 0.0, -0.8, -math.pi / 2, 2.0, 0.7)
    car_road, cow_road = p_road.predict([car, cow], 0.0)
    car_free, cow_free = p_free.predict([car, cow], 0.0)
    # oncoming car: lateral drift over 4 s is bounded by v_lat * tau (0.5 m) instead of 2 m
    assert abs(car_road.y[-1] - 1.2) < 0.55 and abs(car_free.y[-1] - 1.2) > 1.9
    assert car_road.x[-1] == pytest.approx(50.0 - 8.0 * 4.0, abs=0.2)     # along-road speed preserved
    assert math.cos(car_road.heading[0]) == pytest.approx(-1.0, abs=1e-6)  # heading snapped to the corridor (reverse)
    # the cow keeps free constant-velocity motion
    assert np.allclose(cow_road.y, cow_free.y) and np.allclose(cow_road.x, cow_free.x)
    # a vehicle crossing the corridor (heading off the road axis) is NOT forced to follow it
    crossing_car = ObjectState("xc", ObjectType.CAR, 0.0, 50.0, -3.0, 0.0, 6.0, math.pi / 2, 4.3, 1.8)
    xc = p_road.predict([crossing_car], 0.0)[0]
    assert xc.y[-1] == pytest.approx(-3.0 + 6.0 * 4.0)


# ------------------------------------------------- acceleration-aware prediction --
def feed_history(predictor, id_, otype, x0, y0, heading, speeds, dt=0.1):
    """Present the object at successive cycles moving along `heading` with the given speeds.

    Returns the prediction made at the last cycle and the object state that produced it.
    """
    c, s = math.cos(heading), math.sin(heading)
    x, y, t, pred, o = x0, y0, 0.0, None, None
    for i, v in enumerate(speeds):
        if i:
            x, y, t = x + speeds[i - 1] * c * dt, y + speeds[i - 1] * s * dt, t + dt
        o = ObjectState(id_, otype, t, x, y, v * c, v * s, heading, 0.6, 0.6)
        pred = predictor.predict([o], now=t)[0]
    return pred, o


def along(pred, o):
    """Signed displacement of the predicted mean along the object's heading."""
    return (pred.x - o.x) * math.cos(o.heading) + (pred.y - o.y) * math.sin(o.heading)


def test_constant_velocity_history_reproduces_pure_cv_prediction(cfg, profiles):
    """(a) With a steady velocity the estimated acceleration is exactly zero: x = x0 + v t."""
    p = ConstantVelocityPredictor(cfg.prediction, profiles)
    pred, o = feed_history(p, "cv", ObjectType.CATTLE, 10.0, -2.0, math.atan2(-0.5, 1.0), [math.hypot(1.0, 0.5)] * 8)
    t = pred.times - o.timestamp
    assert np.allclose(pred.x, o.x + o.vx * t) and np.allclose(pred.y, o.y + o.vy * t)
    assert np.allclose(pred.vx, o.vx) and np.allclose(pred.vy, o.vy)
    assert pred.sigma()[0] == pytest.approx(profiles.get(ObjectType.CATTLE).sigma_pos0_m)


def test_decelerating_pedestrian_stops_short_and_never_reverses(cfg, profiles):
    """(b) A pedestrian braking at 2 m/s^2 is predicted to come to rest, not to walk backwards."""
    p = ConstantVelocityPredictor(cfg.prediction, profiles)
    speeds = [max(2.5 - 2.0 * 0.1 * k, 0.0) for k in range(10)]           # 2.5 -> 0.7 m/s over 0.9 s
    pred, o = feed_history(p, "ped", ObjectType.PEDESTRIAN, 30.0, -3.0, math.pi / 2, speeds)
    s = along(pred, o)
    t = pred.times - o.timestamp
    assert np.all(s[1:] < o.speed * t[1:])                                   # short of the CV path
    assert np.all(np.diff(s) >= -1e-9)                                      # never reverses
    v_along = pred.vx * math.cos(o.heading) + pred.vy * math.sin(o.heading)
    assert np.all(v_along >= -1e-9) and v_along[-1] == pytest.approx(0.0)  # comes to rest and stays
    assert s[-1] < o.speed ** 2 / (2 * 1.0) + 0.05                         # stopping distance for |a| >= 1 m/s^2
    # intent prior: a braking pedestrian is about to stop -> along-heading uncertainty grows less than CV
    cv = ConstantVelocityPredictor(cfg.prediction, profiles).predict([o], now=o.timestamp)[0]
    assert pred.covariances[-1, 1, 1] < cv.covariances[-1, 1, 1]           # heading is +y
    assert pred.covariances[-1, 0, 0] == pytest.approx(cv.covariances[-1, 0, 0])   # lateral growth unchanged


def test_accelerating_vehicle_is_predicted_ahead_of_cv(cfg, profiles):
    """(c) A car speeding up along the corridor is predicted further ahead than constant velocity at 1 s."""
    road = DrivableSpace.straight(200.0, 3.5, 3.5, x0=-20.0)
    p = ConstantVelocityPredictor(cfg.prediction, profiles, road)
    speeds = [8.0 + 1.5 * 0.1 * k for k in range(10)]                     # +1.5 m/s^2
    pred, o = feed_history(p, "car", ObjectType.CAR, 20.0, 1.75, 0.0, speeds)
    k = int(round(1.0 / cfg.prediction.dt_s))
    assert pred.times[k] - o.timestamp == pytest.approx(1.0)
    assert pred.x[k] > o.x + o.vx * 1.0 + 0.3                              # ahead of CV
    assert pred.x[k] < o.x + o.vx * 1.0 + 0.5 * 1.5 * 1.0 ** 2 + 1e-6      # but not beyond the true CA
    assert np.allclose(pred.y, 1.75)                                       # stays in its lane
    assert pred.vx[-1] == pytest.approx(pred.vx[k + 5]) and pred.vx[-1] > o.vx  # velocity held after the CA horizon


def test_noisy_velocity_history_respects_acceleration_clamp(cfg, profiles):
    """(d) Velocity noise of 1 m/s per cycle cannot inject more than max_acceleration_mps2."""
    rng = np.random.default_rng(3)
    p = ConstantVelocityPredictor(cfg.prediction, profiles)
    a_max = cfg.prediction.max_acceleration_mps2
    t = 0.0
    for _ in range(40):
        vx, vy = 1.0 + rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)
        o = ObjectState("noisy", ObjectType.CATTLE, t, 5.0, 5.0, vx, vy, math.atan2(vy, vx), 2.0, 0.7)
        pred = p.predict([o], now=t)[0]
        dv = math.hypot(pred.vx[-1] - pred.vx[0], pred.vy[-1] - pred.vy[0])
        assert dv <= a_max * cfg.prediction.acceleration_horizon_s + 1e-9
        t += 0.1
