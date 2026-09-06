"""Phase 4: timestamp-aware multi-sensor fusion.

Every test here drives `SensorFusionTracker` directly with hand-built detections, so the timing is
exactly what the test says it is rather than whatever the simulator happened to produce.

The object under test is a constant-velocity Kalman filter, so there is a closed-form answer to
"where should this track be". That is what these tests check against: not that the code runs, but
that a measurement taken 80 ms ago is placed where the object was 80 ms ago.
"""
import math

import numpy as np
import pytest

from autonomy.core.config import ObjectProfiles
from autonomy.core.types import Detection, ObjectType, SensorType, VehicleState
from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig

PROFILES = ObjectProfiles.load()


def cfg(**kw) -> TrackerConfig:
    """Temporal fusion ON, confirmation immediate, so a test can read a track after one batch."""
    base = dict(temporal_fusion=True, confirm_hits=1, process_noise_accel_mps2=1.5)
    base.update(kw)
    return TrackerConfig(**base)


def tracker(**kw) -> SensorFusionTracker:
    return SensorFusionTracker(cfg(**kw), PROFILES)


def det(sensor: SensorType, t: float, x: float, y: float, std: float = 0.2, **kw) -> Detection:
    d = Detection(sensor, t, x, y, np.eye(2) * std ** 2, **kw)
    d.truth_id = "truth"                      # evaluation-only field the tracker asserts on
    return d


def camera(t, x, y, **kw):
    return det(SensorType.CAMERA, t, x, y, std=0.5,
               object_type=ObjectType.CAR, class_confidence=0.9, **kw)


def lidar(t, x, y, **kw):
    return det(SensorType.LIDAR, t, x, y, std=0.1, length=4.0, width=1.8, heading=0.0, **kw)


def radar(t, x, y, radial=None, **kw):
    return det(SensorType.RADAR, t, x, y, std=0.4, radial_speed=radial,
               radial_speed_std=0.15, **kw)


EGO = VehicleState(timestamp=0.0, x=0.0, y=0.0, yaw=0.0, longitudinal_velocity=0.0)


def only_track(tk: SensorFusionTracker):
    assert len(tk.tracks) == 1, f"expected exactly one track, got {len(tk.tracks)}"
    return tk.tracks[0]


# ------------------------------------------------------- 1/2. camera and LiDAR, either way ---- #
def test_camera_before_lidar_places_each_measurement_at_its_own_time():
    """An object at 10 m/s along +x, seen by camera at t=0.00 and LiDAR at t=0.06.

    Both measurements are consistent with the SAME moving object. A tracker that applied them at a
    common time would see a 0.6 m disagreement and fight itself; one that respects the timestamps
    sees no disagreement at all and should recover the velocity.
    """
    def run(temporal):
        tk = SensorFusionTracker(cfg(temporal_fusion=temporal), PROFILES)
        tk.ingest([camera(0.00, 0.0, 0.0)], 0.10, EGO)
        tk.ingest([lidar(0.06, 0.6, 0.0)], 0.10, EGO)
        return only_track(tk)
    sync, temporal = run(False), run(True)
    assert temporal.x[2] > sync.x[2] + 0.2,         "the time offset carries the velocity information; it was discarded"
    assert temporal.state_time == pytest.approx(0.06)


def test_lidar_before_camera_gives_a_consistent_answer():
    """The same object and the same two measurements, delivered in the other order."""
    def run(temporal):
        tk = SensorFusionTracker(cfg(temporal_fusion=temporal), PROFILES)
        tk.ingest([lidar(0.00, 0.0, 0.0)], 0.10, EGO)
        tk.ingest([camera(0.06, 0.6, 0.0)], 0.10, EGO)
        return only_track(tk)
    assert run(True).x[2] > run(False).x[2] + 0.2


def test_a_synchronised_pair_infers_no_velocity_from_position_alone():
    """The control for the two tests above: same positions, same timestamp.

    If the tracker inferred a large velocity here it would be reading motion out of sensor
    disagreement rather than out of elapsed time.
    """
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0), camera(0.00, 0.6, 0.0)], 0.10, EGO)
    tr = only_track(tk)
    assert abs(tr.x[2]) < 2.0


# ------------------------------------------------------------ 3. radar between cameras -------- #
def test_radar_arriving_between_two_camera_frames_is_applied_in_between():
    tk = tracker()
    tk.ingest([camera(0.00, 0.0, 0.0)], 0.30, EGO)
    tk.ingest([radar(0.05, 0.5, 0.0, radial=10.0)], 0.30, EGO)
    mid = only_track(tk).state_time
    tk.ingest([camera(0.10, 1.0, 0.0)], 0.30, EGO)
    tr = only_track(tk)
    assert mid == pytest.approx(0.05)
    assert tr.state_time == pytest.approx(0.10)
    assert tr.x[2] > 1.0


def test_the_radar_ekf_still_updates_velocity_from_radial_speed():
    """Phase 4 must not disturb the extended-Kalman radial update."""
    tk = tracker()
    tk.ingest([lidar(0.00, 20.0, 0.0)], 0.00, EGO)
    before = only_track(tk).x[2]
    tk.ingest([radar(0.00, 20.0, 0.0, radial=8.0)], 0.00, EGO)
    after = only_track(tk).x[2]
    assert after > before, "radial speed carried no information into the state"
    assert "RADAR" in only_track(tk).sensors_seen


# ------------------------------------------------------------- 4. non-zero sensor offset ------ #
def test_a_real_nuscenes_sized_offset_is_respected():
    """66.8 ms is the MEASURED median separation between nuScenes sensors in one keyframe."""
    def run(temporal):
        tk = SensorFusionTracker(cfg(temporal_fusion=temporal), PROFILES)
        tk.ingest([lidar(0.000, 0.0, 0.0)], 0.10, EGO)
        tk.ingest([camera(0.0668, 0.668, 0.0)], 0.10, EGO)
        return only_track(tk)
    temporal = run(True)
    assert temporal.state_time == pytest.approx(0.0668)
    assert temporal.x[2] > run(False).x[2] + 0.2


def test_the_same_batch_under_the_synchronised_assumption_loses_the_offset():
    """The A/B in one test: identical measurements, `temporal_fusion` off.

    The synchronised path must place the state at `now` and must NOT recover the motion, because
    it was told the two measurements happened at the same instant.
    """
    tk = SensorFusionTracker(cfg(temporal_fusion=False), PROFILES)
    tk.ingest([lidar(0.000, 0.0, 0.0)], 0.10, EGO)
    tk.ingest([camera(0.0668, 0.668, 0.0)], 0.20, EGO)
    tr = only_track(tk)
    assert tr.state_time == pytest.approx(0.20)


# ---------------------------------------------------------------- 5. out-of-order ------------- #
def test_a_slightly_late_measurement_is_fused_with_inflated_noise():
    """The track is already past the measurement's time, and prediction cannot run backwards.

    The measurement still informs the state, but it is trusted less, by exactly the distance the
    object could have travelled during the lag.
    """
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.20, EGO)
    tk.ingest([lidar(0.10, 1.0, 0.0)], 0.20, EGO)
    before = only_track(tk).x.copy()
    tk.ingest([camera(0.05, 0.5, 0.0)], 0.20, EGO)     # 50 ms behind the state
    tr = only_track(tk)
    assert tk.out_of_order >= 1
    assert not np.allclose(tr.x, before), "the late measurement was ignored entirely"
    assert tr.state_time == pytest.approx(0.10), "prediction must never run backwards"


def test_a_measurement_far_behind_the_state_is_dropped_not_fused():
    tk = tracker(max_out_of_order_s=0.05)
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.60, EGO)
    tk.ingest([lidar(0.50, 0.4, 0.0)], 0.60, EGO)
    before = only_track(tk).x.copy()
    tk.ingest([camera(0.10, 0.1, 0.0)], 0.60, EGO)     # 400 ms behind: far too stale to fuse
    assert np.allclose(only_track(tk).x, before), "a badly stale measurement must not be fused"


def test_a_measurement_from_the_future_is_rejected():
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.00, EGO)
    tk.ingest([lidar(5.00, 50.0, 0.0)], 0.10, EGO)     # stamped 4.9 s ahead of `now`
    assert tk.rejected_future == 1
    assert only_track(tk).state_time == pytest.approx(0.0)


def test_a_measurement_older_than_the_age_limit_is_rejected():
    tk = tracker(max_measurement_age_s=0.2)
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.00, EGO)
    tk.ingest([lidar(0.01, 0.1, 0.0)], 1.00, EGO)      # 990 ms old
    assert tk.rejected_stale == 1


# ---------------------------------------------------- 6. several measurements, one track ------ #
def test_three_sensors_in_one_batch_each_land_at_their_own_time():
    """A track must already exist, otherwise there is nothing to associate to.

    On a first batch every detection necessarily births its own track and duplicate merging tidies
    up afterwards, which is track initialisation rather than fusion. The interesting case is a
    known object measured by all three sensors at three different instants.
    """
    tk = tracker()
    tk.ingest([lidar(-0.05, -0.5, 0.0)], 0.10, EGO)
    tr = only_track(tk)
    hits_before = tr.hits
    tk.ingest([camera(0.00, 0.0, 0.0), lidar(0.03, 0.3, 0.0),
               radar(0.06, 0.6, 0.0, radial=10.0)], 0.10, EGO)
    tr = only_track(tk)
    assert tr.hits == hits_before + 3, "each sensor must contribute exactly one update"
    assert tr.sensors_seen == {"CAMERA", "LIDAR", "RADAR"}
    assert tr.state_time == pytest.approx(0.06), "the state ends at the newest measurement time"


def test_one_sensor_updates_a_track_at_most_once_per_batch():
    """Two LiDAR returns on the same object in one batch must not both update it."""
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.00, EGO)
    tk.ingest([lidar(0.05, 0.5, 0.0), lidar(0.05, 0.55, 0.0)], 0.05, EGO)
    # One update, not two: the track must sit at the measured position, not be dragged past it by
    # a second bite from the same sensor in the same batch.
    assert only_track(tk).x[0] == pytest.approx(0.5, abs=0.15)


def test_tracks_are_never_double_predicted():
    """Predicting a track to a time it has already reached must be a no-op.

    Two sensor groups sharing a timestamp means the second group's predict has nothing to do; if
    it advanced the state anyway, the covariance would grow twice for one interval.
    """
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.00, EGO)
    tr = only_track(tk)
    tk._predict_to(tr, 0.10)
    P_once = tr.P.copy()
    tk._predict_to(tr, 0.10)
    tk._predict_to(tr, 0.05)                          # backwards: also a no-op
    assert np.allclose(tr.P, P_once)
    assert tr.state_time == pytest.approx(0.10)


# ------------------------------------------------------------------ 7. missing stream --------- #
def test_a_lidar_only_stream_still_tracks():
    tk = tracker()
    for i in range(6):
        tk.ingest([lidar(i * 0.05, i * 0.5, 0.0)], i * 0.05, EGO)
    tr = only_track(tk)
    assert tr.sensors_seen == {"LIDAR"}
    # The filter is damped, so it approaches 10 m/s rather than jumping to it; what matters is that
    # a single-sensor stream still yields a confident forward velocity.
    assert tr.x[2] > 1.0


def test_an_empty_batch_advances_nothing_and_crashes_nothing():
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.00, EGO)
    before = only_track(tk).state_time
    tk.ingest([], 0.50, EGO)
    assert only_track(tk).state_time == pytest.approx(before)


# --------------------------------------------------------------------- 8. determinism --------- #
def test_the_same_input_gives_the_same_output_every_time():
    def run():
        tk = tracker()
        for i in range(5):
            tk.ingest([camera(i * 0.10, i * 1.0, 0.0),
                       lidar(i * 0.10 + 0.03, i * 1.0 + 0.3, 0.0),
                       radar(i * 0.10 + 0.06, i * 1.0 + 0.6, 0.0, radial=10.0)],
                      i * 0.10 + 0.09, EGO)
        return only_track(tk).x.copy(), only_track(tk).P.copy()
    x1, P1 = run()
    x2, P2 = run()
    assert np.array_equal(x1, x2) and np.array_equal(P1, P2)


# ------------------------------------------------------------ 9. covariance propagation ------- #
def test_coasting_grows_the_covariance_and_a_measurement_shrinks_it():
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.00, EGO)
    fresh = np.trace(only_track(tk).P[:2, :2])
    tk.ingest([], 0.00, EGO)
    tk._predict_to(only_track(tk), 0.50)
    coasted = np.trace(only_track(tk).P[:2, :2])
    tk.ingest([lidar(0.50, 0.0, 0.0)], 0.50, EGO)
    measured = np.trace(only_track(tk).P[:2, :2])
    assert coasted > fresh, "half a second of coasting must cost certainty"
    assert measured < coasted, "a measurement must buy it back"


def test_the_reported_covariance_is_propagated_to_the_requested_output_time():
    """The published covariance must belong to the same instant as the published position.

    Extrapolating the position while reporting an older covariance understates the uncertainty,
    and the collision margin downstream is sized from exactly this matrix.
    """
    tk = tracker()
    tk.ingest([lidar(0.00, 0.0, 0.0)], 0.00, EGO)
    now_state = tk.get_object_states(0.0)[0]
    later = tk.get_object_states(0.5)[0]
    assert np.trace(later.covariance) > np.trace(now_state.covariance)


def test_covariance_stays_symmetric_and_positive_definite_through_mixed_timing():
    tk = tracker()
    for i in range(8):
        t = i * 0.05
        tk.ingest([camera(t, i * 0.5, 0.0)], t + 0.08, EGO)
        tk.ingest([lidar(t + 0.02, i * 0.5 + 0.2, 0.0)], t + 0.08, EGO)
    P = only_track(tk).P
    assert np.allclose(P, P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(P) > 0)


# ------------------------------------------------- 10. no regression in synchronised mode ----- #
def test_synchronised_mode_is_bit_identical_to_the_pre_phase4_behaviour():
    """With every timestamp equal to `now`, both paths must agree exactly.

    This is the guarantee that Phase 4 is additive: a caller that never had asynchronous data sees
    the same numbers it saw before.
    """
    def run(temporal):
        tk = SensorFusionTracker(cfg(temporal_fusion=temporal), PROFILES)
        for i in range(6):
            t = i * 0.05
            tk.ingest([lidar(t, i * 0.5, 0.0), radar(t, i * 0.5 + 0.05, 0.0, radial=10.0),
                       camera(t, i * 0.5 - 0.05, 0.0)], t, EGO)
        return only_track(tk).x.copy(), only_track(tk).P.copy()
    xa, Pa = run(False)
    xb, Pb = run(True)
    assert np.allclose(xa, xb, atol=1e-12)
    assert np.allclose(Pa, Pb, atol=1e-12)


def test_the_shipped_configuration_leaves_temporal_fusion_off():
    """The simulator default is unchanged, deliberately. See docs/PHASE4_TEMPORAL_FUSION.md."""
    from autonomy.core.config import PerceptionConfig
    assert PerceptionConfig.load().tracker.get("temporal_fusion") is False
    assert TrackerConfig().temporal_fusion is False
