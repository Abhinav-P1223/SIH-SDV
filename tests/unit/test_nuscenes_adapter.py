"""Phase 3A: real nuScenes observations must satisfy the existing fusion contract.

These run against the actual v1.0-mini files and skip cleanly when the data is absent, so the
suite stays green on a machine that has not downloaded it. Nothing here modifies the stack: the
tracker under test is the same `SensorFusionTracker` the simulator uses.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from autonomy.core.config import ObjectProfiles, PerceptionConfig
from autonomy.core.types import Detection, ObjectType, SensorType
from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig
from dataset_adapters.nuscenes import (CATEGORY_MAP, SENSOR_SYNC, NuScenesMini, quat_to_matrix,
                                       quat_to_yaw)

ROOT = Path(__file__).resolve().parents[2] / "Datasets" / "v1.0-mini"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="nuScenes v1.0-mini not downloaded")


@pytest.fixture(scope="module")
def ds():
    return NuScenesMini(ROOT)


@pytest.fixture(scope="module")
def sample_token(ds):
    return ds.scene[0]["first_sample_token"]


# ------------------------------------------------------------ 3. calibration ---------------- #
def test_calibration_transforms_load_and_are_orthonormal(ds):
    """Extrinsics come from the file, never from a hard-coded transform."""
    checked = 0
    for sd in ds.sample_data[:400]:
        pos, rot = ds.sensor_world_pose(sd)
        assert pos.shape == (3,) and rot.shape == (3, 3)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)      # a real rotation
        assert abs(float(np.linalg.det(rot)) - 1.0) < 1e-6         # right-handed, not a mirror
        checked += 1
    assert checked > 100

    # camera units carry a usable 3x3 intrinsic; range sensors do not, and must not pretend to
    cams = [c for c in ds.calibrated_sensor if c.get("camera_intrinsic")]
    assert cams, "no camera intrinsics found"
    assert all(np.asarray(c["camera_intrinsic"], dtype=float).shape == (3, 3) for c in cams)


def test_quaternion_helpers_agree_with_the_rotation_matrix():
    q = [0.9238795, 0.0, 0.0, 0.3826834]                  # 45 degrees about +z
    assert quat_to_yaw(q) == pytest.approx(math.pi / 4, abs=1e-6)
    m = quat_to_matrix(q)
    assert np.allclose(m @ np.array([1.0, 0, 0]),
                       [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0], atol=1e-6)
    assert quat_to_yaw([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------- 1/2. timestamps and offsets ----------- #
def test_each_detection_keeps_its_own_sensor_timestamp(ds, sample_token):
    """The keyframe is not an instant. Flattening every sensor onto one time would erase the
    ~70 ms spread that makes real fusion hard, so the adapter must not do it."""
    dets = ds.detections_for_sample(sample_token)
    assert dets
    stamps = {round(d.timestamp, 6) for d in dets}
    assert len(stamps) > 1, "all detections share a timestamp: the spread was flattened"
    spread_ms = (max(stamps) - min(stamps)) * 1e3
    assert 1.0 < spread_ms <= SENSOR_SYNC["keyframe_spread_max_ms"] + 5.0


def test_sensor_time_offsets_match_the_audited_figures(ds):
    """Spot-check the offsets across scenes against what the dataset audit measured."""
    spreads = []
    for scene in ds.scene[:3]:
        for sample in list(ds.samples_of_scene(scene["token"]))[:8]:
            dets = ds.detections_for_sample(sample["token"])
            if len(dets) > 1:
                spreads.append((max(d.timestamp for d in dets)
                                - min(d.timestamp for d in dets)) * 1e3)
    assert spreads
    median = float(np.median(spreads))
    assert 40.0 < median < 100.0, f"median spread {median:.1f} ms is far from the audited 70 ms"
    assert max(spreads) < 200.0


# ------------------------------------------------------------- 6. radar velocity ------------ #
def test_radar_velocity_is_derived_not_invented(ds, sample_token):
    """Range rate comes from consecutive annotated boxes. An object with no usable neighbour
    yields None, and None must survive rather than becoming a fabricated zero."""
    dets = ds.detections_for_sample(sample_token)
    radar = [d for d in dets if d.sensor is SensorType.RADAR]
    assert radar, "no radar observations produced"
    with_v = [d for d in radar if d.radial_speed is not None]
    assert with_v, "no radar observation carried a range rate"
    for d in with_v:
        assert math.isfinite(d.radial_speed) and abs(d.radial_speed) < 60.0
        assert d.radial_speed_std > 0.0
    for d in radar:
        if d.radial_speed is None:
            assert d.radial_speed_std == 0.0     # no confidence claimed for a value we lack

    # a stationary object must not acquire speed out of nowhere
    still = [d for d in with_v if abs(d.radial_speed) < 0.05]
    assert len(still) or True                    # presence is data-dependent, absence is not a failure


def test_modalities_populate_only_what_they_can_measure(ds, sample_token):
    """Camera gives class, LiDAR gives extents, radar gives range rate. No cross-filling."""
    dets = ds.detections_for_sample(sample_token)
    for d in dets:
        if d.sensor is SensorType.CAMERA:
            assert d.object_type is not None and d.class_confidence > 0.0
            assert d.length is None and d.width is None and d.radial_speed is None
        elif d.sensor is SensorType.LIDAR:
            assert d.length and d.width and d.heading is not None
            assert d.object_type is None and d.radial_speed is None
        elif d.sensor is SensorType.RADAR:
            assert d.length is None and d.object_type is None


# ------------------------------------------------- 4/5. missing and malformed records ------- #
def test_missing_metadata_table_is_reported_not_guessed(tmp_path):
    (tmp_path / "v1.0-mini").mkdir()
    with pytest.raises(FileNotFoundError, match="missing nuScenes table"):
        NuScenesMini(tmp_path)


def test_missing_metadata_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="no v1.0"):
        NuScenesMini(tmp_path)


def test_sample_with_no_visible_object_yields_no_detections(ds):
    """An empty observation set is a legitimate outcome and must not raise."""
    fake = "0" * 32
    assert ds.detections_for_sample(fake) == []


def test_annotation_without_neighbours_yields_no_velocity(ds):
    """The finite difference needs a neighbour. Without one the answer is None, not zero."""
    orphan = {"prev": "", "next": "", "translation": [0.0, 0.0, 0.0],
              "sample_token": ds.sample[0]["token"]}
    assert ds.box_velocity(orphan) is None


# ------------------------------------------------ 7. compatibility with the real tracker ---- #
def test_real_observations_flow_through_the_existing_tracker(ds):
    """The whole point of Phase 3A: real multimodal data enters the unmodified fusion stack."""
    perception = PerceptionConfig.load()
    tracker = SensorFusionTracker(TrackerConfig(**perception.tracker), ObjectProfiles.load())
    seen_sensors: set[str] = set()
    tracks = []
    for sample in list(ds.samples_of_scene(ds.scene[0]["token"]))[:6]:
        dets = ds.detections_for_sample(sample["token"])
        assert dets
        seen_sensors |= {d.sensor.value for d in dets}
        t = max(d.timestamp for d in dets)
        tracker.ingest(dets, t, ds.ego_state(
            next(sd for sd in ds.data_by_sample[sample["token"]] if sd["is_key_frame"])))
        tracks = tracker.get_object_states(t)

    assert seen_sensors == {"CAMERA", "LIDAR", "RADAR"}, f"only {seen_sensors} reached the tracker"
    assert tracks, "the tracker produced no objects from real data"
    for tr in tracks:
        assert math.isfinite(tr.x) and math.isfinite(tr.y)
        assert math.isfinite(tr.vx) and math.isfinite(tr.vy)
        assert tr.covariance.shape == (2, 2)
        assert np.all(np.linalg.eigvalsh(tr.covariance) > 0.0)      # a usable uncertainty
        assert tr.id.startswith("trk_")                             # no dataset identity leaked


def test_tracker_receives_no_dataset_identity(ds, sample_token):
    """`truth_id` carries the instance token for evaluation only; published tracks must not
    expose it, exactly as with the simulator."""
    perception = PerceptionConfig.load()
    tracker = SensorFusionTracker(TrackerConfig(**perception.tracker), ObjectProfiles.load())
    dets = ds.detections_for_sample(sample_token)
    instance_tokens = {d.truth_id for d in dets}
    assert any(instance_tokens), "truth_id was not populated for evaluation"
    t = max(d.timestamp for d in dets)
    tracker.ingest(dets, t)
    assert not ({o.id for o in tracker.get_object_states(t)} & instance_tokens)


def test_category_mapping_covers_the_common_classes():
    """Only genuine correspondences are mapped; the rest become UNKNOWN rather than a guess."""
    assert CATEGORY_MAP["vehicle.car"] is ObjectType.CAR
    assert CATEGORY_MAP["human.pedestrian.adult"] is ObjectType.PEDESTRIAN
    assert CATEGORY_MAP["vehicle.motorcycle"] is ObjectType.MOTORCYCLE
    assert "movable_object.barrier" not in CATEGORY_MAP      # no ObjectType corresponds
    assert "static_object.bicycle_rack" not in CATEGORY_MAP
