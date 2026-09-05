"""Sensor models and the fusion tracker: FOV, range, occlusion, noise, dropout,
latency, association, id persistence, lifecycle, class/dimension fusion and
the sensor ablations that MUST change the output."""
import math

import numpy as np
import pytest

from autonomy.core.config import ObjectProfiles, PerceptionConfig
from autonomy.core.types import Detection, ObjectType, SensorType, VehicleState
from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig
from simulation.agents.agent import Agent
from simulation.sensors.models import CameraModel, LidarModel, RadarModel, SensorConfig, SensorSuite
from autonomy.core.types import AgentBehaviorType


def agent(id_, x, y, heading=0.0, speed=0.0, otype=ObjectType.CAR, length=4.3, width=1.8):
    return Agent(id_, otype, x, y, heading, speed, length, width, AgentBehaviorType.STATIC, {}, seed=1)


def ego(x=0.0, y=0.0, yaw=0.0, v=0.0):
    return VehicleState(0.0, x, y, yaw, v)


# ------------------------------------------------------------------ sensors --
def test_fov_and_range_limits():
    cam = CameraModel(SensorConfig(fov_deg=90.0, range_m=50.0, occlusion=False, dropout_prob=0.0), np.random.default_rng(0))
    ahead, side, behind, far = agent("a", 20, 0), agent("s", 5, 10), agent("b", -10, 0), agent("f", 80, 0)
    dets = cam.sense([ahead, side, behind, far], ego(), 0.0)
    assert {d.truth_id for d in dets} == {"a"}


def test_occlusion_blocks_line_of_sight():
    lid = LidarModel(SensorConfig(fov_deg=360, range_m=100, occlusion=True, dropout_prob=0.0), np.random.default_rng(0))
    near, hidden = agent("near", 15, 0, length=9.0, width=2.5), agent("hidden", 30, 0.2)
    ids = {d.truth_id for d in lid.sense([near, hidden], ego(), 0.0)}
    assert ids == {"near"}
    lid2 = LidarModel(SensorConfig(fov_deg=360, range_m=100, occlusion=False, dropout_prob=0.0), np.random.default_rng(0))
    assert {d.truth_id for d in lid2.sense([near, hidden], ego(), 0.0)} == {"near", "hidden"}


def test_noise_is_seeded_and_covariance_matches_geometry():
    cfg = SensorConfig(fov_deg=120, range_m=100, occlusion=False, dropout_prob=0.0, bearing_std_deg=1.0,
                       range_std_frac=0.02, range_std_min_m=0.2)
    a = agent("a", 40, 0)
    d1 = CameraModel(cfg, np.random.default_rng(3)).sense([a], ego(), 0.0)[0]
    d2 = CameraModel(cfg, np.random.default_rng(3)).sense([a], ego(), 0.0)[0]
    assert (d1.x, d1.y) == (d2.x, d2.y)                      # deterministic under seed
    assert (d1.x, d1.y) != (40.0, 0.0)                       # actually noisy
    # object straight ahead: range error along x, bearing error along y => cov_xx = sr^2, cov_yy = (r sb)^2
    sr = 0.02 * 40 + 0.2
    assert d1.covariance[0, 0] == pytest.approx(sr ** 2, rel=1e-6)
    assert d1.covariance[1, 1] == pytest.approx((40 * math.radians(1.0)) ** 2, rel=1e-6)


def test_radar_radial_speed_relative_to_moving_sensor():
    rad = RadarModel(SensorConfig(fov_deg=60, range_m=120, occlusion=False, dropout_prob=0.0, radial_speed_std_mps=0.0,
                                  range_std_m=0.0, bearing_std_deg=0.0), np.random.default_rng(0))
    a = agent("a", 50, 0, heading=math.pi, speed=8.0)       # oncoming at 8 m/s
    d = rad.sense([a], ego(v=10.0), 0.0)[0]
    assert d.radial_speed == pytest.approx(-18.0)            # closing at 18 m/s
    assert d.object_type is None                             # radar does not classify


def test_camera_classifies_with_confusion_and_lidar_measures_extent():
    cam = CameraModel(SensorConfig(fov_deg=120, range_m=100, occlusion=False, dropout_prob=0.0, class_accuracy=1.0),
                      np.random.default_rng(0))
    cow = agent("c", 20, 0, otype=ObjectType.CATTLE, length=2.0, width=0.7)
    assert cam.sense([cow], ego(), 0.0)[0].object_type == ObjectType.CATTLE
    cam_bad = CameraModel(SensorConfig(fov_deg=120, range_m=100, occlusion=False, dropout_prob=0.0, class_accuracy=0.0),
                          np.random.default_rng(0))
    assert cam_bad.sense([cow], ego(), 0.0)[0].object_type != ObjectType.CATTLE
    lid = LidarModel(SensorConfig(fov_deg=360, range_m=100, occlusion=False, dropout_prob=0.0, extent_std_m=0.0,
                                  heading_std_deg=0.0), np.random.default_rng(0))
    d = lid.sense([cow], ego(), 0.0)[0]
    assert (d.length, d.width) == pytest.approx((2.0, 0.7)) and d.object_type is None


def test_rate_latency_and_dropout_through_the_suite():
    suite = SensorSuite.from_config({
        "camera": {"enabled": True, "rate_hz": 10.0, "latency_s": 0.1, "fov_deg": 120, "range_m": 100,
                   "occlusion": False, "dropout_prob": 0.0},
        "lidar": {"enabled": False}, "radar": {"enabled": False}}, seed=1)
    a = agent("a", 20, 0)
    delivered = []
    t = 0.0
    while t < 1.0 - 1e-9:
        delivered += [(round(t, 2), d.timestamp) for d in suite.sense([a], ego(), t)]
        t = round(t + 0.02, 2)
    assert len(delivered) == 9                                # 10 samples/s, the last one still in flight
    assert all(arr - meas >= 0.1 - 1e-9 for arr, meas in delivered)
    drop = CameraModel(SensorConfig(fov_deg=120, range_m=100, occlusion=False, dropout_prob=1.0), np.random.default_rng(0))
    assert drop.sense([a], ego(), 0.0) == [] and drop.dropped == 1


# ------------------------------------------------------------------ tracker --
def det(sensor, t, x, y, cov=0.05, **kw):
    return Detection(sensor, t, x, y, np.eye(2) * cov ** 2, truth_id="x", **kw)


def make_tracker(**over):
    cfg = TrackerConfig(**{**PerceptionConfig.load().tracker, **over})
    return SensorFusionTracker(cfg, ObjectProfiles.load())


def test_track_confirms_persists_and_estimates_velocity():
    tr = make_tracker()
    for k in range(20):
        t = k * 0.1
        tr.ingest([det(SensorType.LIDAR, t, 20.0 + 2.0 * t, 1.0)], t, ego())
        objs = tr.get_object_states(t)
        if k < 2:
            assert objs == []                                 # not yet confirmed
    objs = tr.get_object_states(1.9)
    assert len(objs) == 1 and objs[0].id == "trk_1"
    assert objs[0].x == pytest.approx(20.0 + 2.0 * 1.9, abs=0.15)
    assert objs[0].vx == pytest.approx(2.0, abs=0.3) and abs(objs[0].vy) < 0.3
    assert np.all(np.linalg.eigvalsh(objs[0].covariance) > 0)


def test_id_persists_through_dropout_and_track_dies_when_object_vanishes():
    tr = make_tracker(max_misses=4, max_age_s=0.6)
    ids = set()
    for k in range(30):
        t = k * 0.1
        dets = [] if k % 3 == 1 else [det(SensorType.LIDAR, t, 10.0 + t, 0.0)]
        tr.ingest(dets, t, ego())
        ids |= {o.id for o in tr.get_object_states(t)}
    assert ids == {"trk_1"}
    for k in range(30, 45):                                   # object disappears
        tr.ingest([], k * 0.1, ego())
    assert tr.get_object_states(4.5) == []


def test_two_objects_are_not_confused():
    tr = make_tracker()
    for k in range(15):
        t = k * 0.1
        tr.ingest([det(SensorType.LIDAR, t, 20.0, 1.5), det(SensorType.LIDAR, t, 20.0, -1.5)], t, ego())
    objs = sorted(tr.get_object_states(1.4), key=lambda o: o.y)
    assert len(objs) == 2 and objs[0].y < -1.0 < 1.0 < objs[1].y


def test_class_fusion_from_camera_votes_and_unknown_without_camera():
    tr = make_tracker()
    for k in range(10):
        t = k * 0.1
        cls = ObjectType.CATTLE if k % 4 else ObjectType.PEDESTRIAN
        tr.ingest([det(SensorType.LIDAR, t, 30.0, 0.0, length=2.0, width=0.7, heading=1.5),
                   det(SensorType.CAMERA, t, 30.0, 0.0, cov=0.6, object_type=cls, class_confidence=0.8)], t, ego())
    o = tr.get_object_states(0.9)[0]
    assert o.object_type == ObjectType.CATTLE
    assert (o.length, o.width) == pytest.approx((2.0, 0.7), abs=1e-6)
    tr2 = make_tracker()
    for k in range(10):
        tr2.ingest([det(SensorType.LIDAR, k * 0.1, 30.0, 0.0)], k * 0.1, ego())
    assert tr2.get_object_states(0.9)[0].object_type == ObjectType.UNKNOWN


def test_radar_improves_velocity_estimate():
    """Ablation: with radar radial speed the velocity covariance shrinks faster."""
    def run(with_radar):
        tr = make_tracker()
        for k in range(6):
            t = k * 0.1
            x = 30.0 - 6.0 * t                                # approaching at 6 m/s
            dets = [det(SensorType.LIDAR, t, x, 0.0, cov=0.1)]
            if with_radar:
                dets.append(det(SensorType.RADAR, t, x, 0.0, cov=0.4, radial_speed=-6.0, radial_speed_std=0.2))
            tr.ingest(dets, t, ego())
        o = tr.get_object_states(0.5)[0]
        return abs(o.vx + 6.0), tr.velocity_covariance(o.id)[0, 0]
    err_no, var_no = run(False)
    err_yes, var_yes = run(True)
    assert var_yes < var_no
    assert err_yes <= err_no + 0.05


def test_lidar_ablation_widens_position_covariance():
    def run(with_lidar):
        tr = make_tracker()
        for k in range(8):
            t = k * 0.1
            dets = [det(SensorType.CAMERA, t, 30.0, 0.0, cov=0.8, object_type=ObjectType.CAR, class_confidence=0.9)]
            if with_lidar:
                dets.append(det(SensorType.LIDAR, t, 30.0, 0.0, cov=0.1, length=4.3, width=1.8, heading=0.0))
            tr.ingest(dets, t, ego())
        return tr.get_object_states(0.7)[0]
    a, b = run(True), run(False)
    assert np.trace(a.covariance) < np.trace(b.covariance)
    assert (a.length, a.width) == pytest.approx((4.3, 1.8)) and b.length == ObjectProfiles.load().get(ObjectType.CAR).length_m


def test_tracker_never_reads_truth_id():
    import inspect
    from autonomy.perception import tracker as mod
    src = inspect.getsource(mod)
    body = src.split("def ingest", 1)[1]
    assert body.count("truth_id") == 1 and "assert d.truth_id" in body


def test_lidar_only_track_survives_between_10hz_frames_and_a_dropped_frame():
    """Ingest runs at 50 Hz; a LiDAR-only object seen at 10 Hz must not be deleted by the empty ingests,
    nor by one dropped frame. Deletion is time-based (max_age_s)."""
    tr = make_tracker()
    t = 0.0
    published = []
    for k in range(150):                      # 3 s at 50 Hz
        t = k * 0.02
        lidar_frame = (k % 5 == 0) and not (60 <= k < 75)      # 10 Hz, one dropped frame at t=1.2..1.5 s
        dets = [det(SensorType.LIDAR, t, 20.0, 1.0)] if lidar_frame else []
        tr.ingest(dets, t, ego())
        published.append(len(tr.get_object_states(t)))
    assert all(p == 1 for p in published[15:]), "track dropped out"
    assert len(tr.tracks) == 1 and tr.tracks[0].id == "trk_1"    # never re-created under a new id
    for k in range(150, 210):                 # object disappears for 1.2 s -> deleted
        tr.ingest([], k * 0.02, ego())
    assert tr.get_object_states(4.2) == []
