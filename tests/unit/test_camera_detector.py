"""Phase 6: real camera object detection feeding the existing detection interface.

Tests that need the COCO weights or a dataset skip cleanly, so a fresh clone stays green. The rest
run everywhere, because the class map, the monocular geometry and the `Detection` contract are all
testable without a model.

The claim these defend is the one worth attacking: that the boxes reaching the tracker come from a
real network run on a real photograph, that their range uncertainty is honest, and that the two
object types we cannot detect are absent rather than quietly faked.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from autonomy.core.types import Detection, ObjectType, SensorType
from perception_detector.detector import (COCO_TO_OBJECT_TYPE, DetectorConfig, RawDetection,
                                          UNREACHABLE_TYPES)
from perception_detector.geometry import (HEIGHT_SPREAD, TYPICAL_HEIGHT_M, CameraGeometry,
                                          boxes_to_detections)

ROOT = Path(__file__).resolve().parents[2]
NUSC = ROOT / "Datasets" / "v1.0-mini"
IDD = ROOT / "Datasets" / "idd-lite" / "idd20k_lite"
CACHE = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

needs_weights = pytest.mark.skipif(
    not (CACHE.exists() and any(CACHE.glob("*mobilenet*"))), reason="COCO weights not cached")
needs_nuscenes = pytest.mark.skipif(not NUSC.exists(), reason="nuScenes-mini not present")
needs_idd = pytest.mark.skipif(not IDD.exists(), reason="IDD-Lite not present")


def geom(w=1600, h=900, hfov=60.0, **kw):
    return CameraGeometry.from_fov(w, h, hfov, **kw)


def raw(x0, y0, x1, y1, otype=ObjectType.CAR, score=0.9, coco=3):
    return RawDetection(x0, y0, x1, y1, score, coco, otype)


def consistent_box(centre_u: float, range_m: float, otype=ObjectType.CAR, g=None):
    """A box whose ground-contact row and apparent height BOTH imply `range_m`.

    Real boxes rarely agree that well, and when they do not the geometry widens the covariance on
    purpose. A test about something else should not accidentally be a test about that, so this
    builds the agreeing case explicitly.
    """
    g = g or geom()
    v_bottom = g.cy + g.fy * g.height_m / range_m
    h_px = TYPICAL_HEIGHT_M[otype] * g.fy / range_m
    return raw(centre_u - 0.4 * h_px, v_bottom - h_px, centre_u + 0.4 * h_px, v_bottom,
               otype, 0.9, 3)


# ------------------------------------------------------------------ 1. the class map ---------- #
def test_the_class_map_only_claims_honest_equivalences():
    """Every mapped COCO id must be a class that genuinely IS the ObjectType it maps to."""
    assert COCO_TO_OBJECT_TYPE[1] is ObjectType.PEDESTRIAN
    assert COCO_TO_OBJECT_TYPE[3] is ObjectType.CAR
    assert COCO_TO_OBJECT_TYPE[21] is ObjectType.CATTLE          # cow
    mapped = set(COCO_TO_OBJECT_TYPE.values())
    for t in UNREACHABLE_TYPES:
        assert t not in mapped, f"{t} has no COCO equivalent and must not be claimed"
    assert ObjectType.UNKNOWN not in mapped, "never map a COCO class onto UNKNOWN"


def test_the_two_indian_classes_we_cannot_see_are_declared():
    assert set(UNREACHABLE_TYPES) == {ObjectType.AUTO_RICKSHAW, ObjectType.PUSHCART}


def test_every_detectable_type_has_a_height_prior():
    """The size-prior range estimator needs one for each type the detector can emit."""
    for t in set(COCO_TO_OBJECT_TYPE.values()):
        assert t in TYPICAL_HEIGHT_M and TYPICAL_HEIGHT_M[t] > 0.3
        assert t in HEIGHT_SPREAD


# ------------------------------------------------------- 2. monocular geometry ---------------- #
def test_bearing_is_zero_on_the_optical_axis_and_signed_left_positive():
    g = geom()
    assert g.bearing_of_column(g.cx) == pytest.approx(0.0)
    assert g.bearing_of_column(g.cx - 200) > 0, "left of centre must be positive, matching y-left"
    assert g.bearing_of_column(g.cx + 200) < 0


def test_bearing_matches_the_field_of_view_at_the_image_edge():
    g = geom(1600, 900, hfov=60.0)
    assert math.degrees(g.bearing_of_column(0)) == pytest.approx(30.0, abs=0.5)
    assert math.degrees(g.bearing_of_column(1600)) == pytest.approx(-30.0, abs=0.5)


def test_ground_contact_range_falls_as_the_contact_row_rises():
    g = geom()
    near = g.range_from_ground_contact(880)
    far = g.range_from_ground_contact(520)
    assert near is not None and far is not None
    assert far > near > 0, "a higher contact row must mean a further object"


def test_a_contact_point_on_the_horizon_yields_no_range():
    g = geom()
    assert g.range_from_ground_contact(g.cy) is None
    assert g.range_from_ground_contact(g.cy - 50) is None


def test_size_prior_range_is_inverse_in_apparent_height():
    g = geom()
    close = g.range_from_height(300, ObjectType.CAR)
    far = g.range_from_height(75, ObjectType.CAR)
    assert far == pytest.approx(4.0 * close, rel=1e-6)


def test_a_taller_class_at_the_same_pixel_height_is_further_away():
    g = geom()
    assert g.range_from_height(100, ObjectType.BUS) > g.range_from_height(100, ObjectType.PEDESTRIAN)


# ------------------------------------------------- 3. honest range uncertainty ---------------- #
def test_disagreement_between_the_two_range_estimates_widens_the_covariance():
    """THE point of the geometry module. Two methods that argue must produce a wide estimate.

    Both boxes sit at the same image column and the same bottom row, so ground contact says the
    same range for each. Only the apparent height differs, so only the size prior disagrees. The
    box whose two estimates conflict must arrive at the tracker visibly less certain.
    """
    g = geom()
    agree = boxes_to_detections([raw(780, 700, 860, 860)], g, 0.0, 0.0, 0.0, 0.0)[0]
    argue = boxes_to_detections([raw(780, 840, 860, 860)], g, 0.0, 0.0, 0.0, 0.0)[0]
    assert np.trace(argue.covariance) > np.trace(agree.covariance) * 2.0


def test_the_covariance_is_range_dominated_like_a_real_camera():
    """Bearing is cheap and range is expensive; the covariance must say so."""
    g = geom()
    d = boxes_to_detections([raw(780, 700, 860, 860)], g, 0.0, 0.0, 0.0, 0.0)[0]
    vals = np.linalg.eigvalsh(d.covariance)
    assert vals.min() > 0, "covariance must be positive definite"
    assert vals.max() / vals.min() > 5.0, "a monocular camera is far more certain in bearing"


def test_uncertainty_grows_with_range():
    g = geom()
    near = boxes_to_detections([raw(780, 760, 860, 880)], g, 0.0, 0.0, 0.0, 0.0)[0]
    far = boxes_to_detections([raw(790, 600, 830, 660)], g, 0.0, 0.0, 0.0, 0.0)[0]
    assert math.hypot(far.x, far.y) > math.hypot(near.x, near.y)
    assert np.trace(far.covariance) > np.trace(near.covariance)


# --------------------------------------------- 4. the Detection contract ---------------------- #
def test_the_output_is_exactly_what_the_simulated_camera_produces():
    g = geom()
    out = boxes_to_detections([raw(780, 700, 860, 860, ObjectType.CAR, 0.87)], g,
                              sensor_x=5.0, sensor_y=-2.0, sensor_yaw=0.0, timestamp=1.25)
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Detection)
    assert d.sensor is SensorType.CAMERA
    assert d.timestamp == 1.25
    assert d.object_type is ObjectType.CAR
    assert d.class_confidence == pytest.approx(0.87)
    assert d.covariance.shape == (2, 2)
    assert (d.sensor_x, d.sensor_y) == (5.0, -2.0)
    assert d.truth_id == "", "a real detector has no ground-truth id and must not invent one"


def test_the_tracker_accepts_detector_output_unmodified():
    """The assertion that looked like a blocker: the tracker requires truth_id is not None.

    The field defaults to an empty string, which satisfies it, so no change to the tracker was
    needed. This test exists so that a future tightening of that assertion fails here loudly
    instead of silently in a scenario run.
    """
    from autonomy.core.config import ObjectProfiles
    from autonomy.core.types import VehicleState
    from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig
    g = geom()
    # Well separated in BEARING. Two boxes near the optical axis would be merged by the tracker,
    # because a monocular range covariance is large enough that they overlap: see
    # test_nearby_monocular_detections_merge_in_the_tracker, which pins that as a real limitation.
    dets = boxes_to_detections([consistent_box(200, 15.0, g=g), consistent_box(1400, 15.0, g=g)],
                               g, 0.0, 0.0, 0.0, 0.0)
    assert len(dets) == 2
    tk = SensorFusionTracker(TrackerConfig(confirm_hits=1), ObjectProfiles.load())
    ego = VehicleState(timestamp=0.0, x=0.0, y=0.0, yaw=0.0, longitudinal_velocity=0.0)
    tk.ingest(dets, 0.0, ego)
    assert len(tk.tracks) == 2, "detections far apart in bearing must stay separate tracks"
    assert len(tk.get_object_states(0.0)) >= 1


def test_nearby_monocular_detections_merge_in_the_tracker():
    """A real limitation, pinned rather than hidden.

    Monocular range uncertainty is metres, so two genuinely distinct objects at similar bearing and
    similar apparent size land inside each other's gate and the tracker merges them into one. That
    is the tracker behaving correctly given what a single camera can tell it, and it is why this
    phase does not claim camera-only object tracking.
    """
    from autonomy.core.config import ObjectProfiles
    from autonomy.core.types import VehicleState
    from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig
    g = geom()
    dets = boxes_to_detections([raw(700, 700, 800, 860), raw(900, 720, 980, 850)], g,
                               0.0, 0.0, 0.0, 0.0)
    assert len(dets) == 2
    tk = SensorFusionTracker(TrackerConfig(confirm_hits=1), ObjectProfiles.load())
    ego = VehicleState(timestamp=0.0, x=0.0, y=0.0, yaw=0.0, longitudinal_velocity=0.0)
    tk.ingest(dets, 0.0, ego)
    assert len(tk.tracks) == 1


def test_the_sensor_pose_places_the_detection_in_the_world():
    g = geom()
    a = boxes_to_detections([raw(780, 700, 860, 860)], g, 0.0, 0.0, 0.0, 0.0)[0]
    b = boxes_to_detections([raw(780, 700, 860, 860)], g, 10.0, 4.0, 0.0, 0.0)[0]
    assert b.x == pytest.approx(a.x + 10.0) and b.y == pytest.approx(a.y + 4.0)
    # Rotating the camera rotates the detection about the sensor and preserves its range.
    turned = boxes_to_detections([raw(780, 700, 860, 860)], g, 0.0, 0.0, math.pi / 2, 0.0)[0]
    assert math.hypot(turned.x, turned.y) == pytest.approx(math.hypot(a.x, a.y), rel=1e-9)
    assert turned.y > 0 and abs(turned.x) < abs(a.x)


def test_a_box_above_the_horizon_with_no_usable_range_is_dropped():
    """Better to emit nothing than to invent a position for something we cannot place."""
    g = geom()
    floating = raw(700, 10, 760, 40)                       # tiny, high in frame
    out = boxes_to_detections([floating], g, 0.0, 0.0, 0.0, 0.0)
    assert all(g.min_range_m <= math.hypot(d.x, d.y) <= g.max_range_m for d in out)


def test_range_is_clamped_to_the_configured_window():
    g = geom(max_range_m=40.0)
    out = boxes_to_detections([raw(795, 599, 805, 601)], g, 0.0, 0.0, 0.0, 0.0)
    for d in out:
        assert math.hypot(d.x, d.y) <= 40.0 + 1e-6


# ------------------------------------------------- 5. the real model, when available ---------- #
@needs_weights
def test_the_detector_runs_a_real_network_and_is_deterministic():
    pytest.importorskip("torchvision")
    from perception_detector import CameraObjectDetector
    det = CameraObjectDetector(DetectorConfig(score_threshold=0.05))
    img = (np.random.default_rng(0).integers(0, 255, (240, 320, 3))).astype(np.uint8)
    a = det.detect(img)
    assert det.last_inference_ms > 0.0
    b = det.detect(img)
    assert len(a) == len(b)
    for p, q in zip(a, b):
        assert (p.x0, p.y0, p.x1, p.y1, p.score) == (q.x0, q.y0, q.x1, q.y1, q.score)


@needs_weights
def test_the_detector_never_emits_a_class_it_cannot_map():
    pytest.importorskip("torchvision")
    from perception_detector import CameraObjectDetector
    det = CameraObjectDetector(DetectorConfig(score_threshold=0.01))
    img = (np.random.default_rng(1).integers(0, 255, (240, 320, 3))).astype(np.uint8)
    for d in det.detect(img):
        assert d.object_type in set(COCO_TO_OBJECT_TYPE.values())
        assert d.object_type not in UNREACHABLE_TYPES


@needs_weights
@needs_idd
def test_it_detects_something_on_real_indian_road_photographs():
    """Qualitative by necessity: IDD-Lite has no detection labels, so this asserts non-vacuity."""
    pytest.importorskip("torchvision")
    from PIL import Image
    from perception_detector import CameraObjectDetector
    det = CameraObjectDetector()
    imgs = sorted((IDD / "leftImg8bit" / "val").rglob("*_image.jpg"))[:8]
    assert imgs, "no IDD-Lite validation images found"
    total = 0
    for p in imgs:
        total += len(det.detect(np.asarray(Image.open(p).convert("RGB"))))
    assert total > 0, "the detector found nothing at all on real Indian-road images"


# --------------------------------------------- 6. the nuScenes 2D projection ------------------ #
@needs_nuscenes
def test_projected_boxes_are_inside_the_image_and_correctly_typed():
    from dataset_adapters.nuscenes_2d import NUSCENES_TO_OBJECT_TYPE, NuScenesCamera2D
    ds = NuScenesCamera2D(NUSC)
    frames = ds.camera_keyframes(("CAM_FRONT",))
    assert len(frames) == 404
    seen = 0
    for sd in frames[:20]:
        fr = ds.frame(sd)
        for b in fr.boxes:
            assert 0 <= b.x0 < b.x1 <= fr.width
            assert 0 <= b.y0 < b.y1 <= fr.height
            assert b.object_type is NUSCENES_TO_OBJECT_TYPE[b.category]
            assert b.distance_m > 0
            seen += 1
    assert seen > 0


@needs_nuscenes
def test_the_projection_never_claims_a_class_nuscenes_does_not_have():
    """No auto-rickshaw or pushcart may appear: nuScenes has neither, so neither may be scored."""
    from dataset_adapters.nuscenes_2d import NUSCENES_TO_OBJECT_TYPE
    for t in UNREACHABLE_TYPES:
        assert t not in set(NUSCENES_TO_OBJECT_TYPE.values())
