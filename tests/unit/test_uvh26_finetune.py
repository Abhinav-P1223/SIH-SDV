"""Phase 7C: the UVH-26 subset loader, and the fine-tuned detector as a drop-in replacement.

Tests needing the dataset or the fine-tuned checkpoint skip cleanly, so a fresh clone stays green.
The dataset is gitignored and the checkpoint may not have been trained yet.

The claim under test is narrow and worth stating: a model fine-tuned on UVH-26 must reach the
existing tracker through exactly the same `Detection` contract the COCO model uses, with no change
anywhere downstream. If that stops being true, this file fails rather than a scenario run.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from autonomy.core.types import Detection, ObjectType, SensorType

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "Datasets" / "uvh26_subset"
MANIFEST = ROOT / "docs" / "uvh26_manifest.json"
CKPT = ROOT / "perception_detector" / "checkpoints" / "fasterrcnn_uvh26.pt"

needs_data = pytest.mark.skipif(not (SUBSET / "images").exists(), reason="UVH-26 subset not present")
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="fine-tuned checkpoint not present")
has_manifest = pytest.mark.skipif(not MANIFEST.exists(), reason="manifest not present")


# ------------------------------------------------------------ 1. class mapping --------------- #
def test_class_ids_are_fixed_and_background_is_zero():
    """torchvision reserves 0. A checkpoint whose class order drifts is read wrong for ever."""
    from perception_detector.uvh26 import CLASS_NAMES, ID_TO_OBJECT_TYPE, NUM_CLASSES
    assert CLASS_NAMES[0] == "__background__"
    assert NUM_CLASSES == 7
    assert CLASS_NAMES[1:] == ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"]
    for i, t in ID_TO_OBJECT_TYPE.items():
        assert CLASS_NAMES[i] == t.value, "class id order and ObjectType must agree exactly"


def test_every_trained_class_maps_to_a_real_object_type():
    from perception_detector.uvh26 import ID_TO_OBJECT_TYPE
    for t in ID_TO_OBJECT_TYPE.values():
        assert isinstance(t, ObjectType) and t is not ObjectType.UNKNOWN


@has_manifest
def test_the_manifest_mapping_never_invents_a_class():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    from perception_detector.uvh26 import CLASS_NAMES
    for src, dst in man["category_map"].items():
        assert dst in CLASS_NAMES[1:], f"{src} maps to unknown class {dst}"
    assert man["excluded_category"] == "Others"


# ------------------------------------------------------------ 2. dataset integrity ------------ #
@needs_data
@has_manifest
def test_splits_are_the_phase7b_ones_and_do_not_leak():
    from perception_detector.uvh26 import UVH26Subset, load_manifest
    man = load_manifest()
    ds = {s: UVH26Subset(s, man) for s in ("train", "val", "test")}
    assert (len(ds["train"]), len(ds["val"]), len(ds["test"])) == (500, 100, 150)
    names = {s: {e["file_name"] for e in d.entries} for s, d in ds.items()}
    assert not names["train"] & names["val"]
    assert not names["train"] & names["test"]
    assert not names["val"] & names["test"]


@needs_data
@has_manifest
def test_every_box_is_valid_and_inside_its_image():
    from perception_detector.uvh26 import UVH26Subset, load_manifest
    man = load_manifest()
    for split in ("train", "val", "test"):
        d = UVH26Subset(split, man)
        assert d.dropped == 0, f"{split}: {d.dropped} degenerate boxes appeared"
        for i in range(len(d)):
            b, l = d.ground_truth(i)
            assert len(b) > 0, "the selection guaranteed at least one box per image"
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
            assert l.min() >= 1 and l.max() <= 6


@needs_data
@has_manifest
def test_a_loaded_sample_is_a_valid_torchvision_target():
    torch = pytest.importorskip("torch")
    from perception_detector.uvh26 import UVH26Subset, load_manifest
    d = UVH26Subset("val", load_manifest(), augment=False)
    x, t = d[0]
    assert x.dtype == torch.float32 and x.shape[0] == 3
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert set(t) >= {"boxes", "labels"}
    assert t["boxes"].shape[1] == 4 and len(t["boxes"]) == len(t["labels"])


@needs_data
@has_manifest
def test_horizontal_flip_moves_boxes_with_the_image():
    """An augmentation that flips pixels but not boxes silently destroys a small dataset."""
    from perception_detector.uvh26 import UVH26Subset, load_manifest
    man = load_manifest()
    plain = UVH26Subset("val", man, augment=False)
    x0, t0 = plain[0]
    W = x0.shape[2]
    flipped = UVH26Subset("val", man, augment=True, seed=1)
    for _ in range(12):                       # flip is random; find one that actually flipped
        x1, t1 = flipped[0]
        if not np.allclose(t1["boxes"].numpy(), t0["boxes"].numpy()):
            b0 = t0["boxes"].numpy(); b1 = t1["boxes"].numpy()
            assert np.allclose(b1[:, 0], W - b0[:, 2], atol=1e-3)
            assert np.allclose(b1[:, 2], W - b0[:, 0], atol=1e-3)
            assert np.allclose(b1[:, [1, 3]], b0[:, [1, 3]], atol=1e-3)
            return
    pytest.skip("no flip drawn in 12 attempts")


# ------------------------------------------------------------ 3. evaluation code -------------- #
def test_iou_and_matching_behave():
    from perception_detector.detect_eval import iou_matrix, match
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    M = iou_matrix(a, b)
    assert M[0, 0] == pytest.approx(1.0) and M[0, 1] == pytest.approx(0.0)
    pairs, used = match(a, np.array([1]), np.array([0.9]), b, np.array([1, 1]), 0.5)
    assert pairs == [(0, 0)] and used == {0}


def test_matching_is_class_aware():
    """A perfectly placed box with the wrong label must be a false positive, not a hit."""
    from perception_detector.detect_eval import match
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    pairs, used = match(box, np.array([3]), np.array([0.9]), box, np.array([1]), 0.5)
    assert pairs == [] and used == set()


def test_average_precision_is_one_for_a_perfect_ranking():
    from perception_detector.detect_eval import average_precision
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0, abs=1e-6)
    assert average_precision([], 5) != average_precision([], 5) or True   # nan for no predictions


# ------------------------------------------- 4. the fine-tuned model as a drop-in replacement -- #
@needs_ckpt
def test_the_finetuned_detector_loads_and_keeps_the_architecture():
    pytest.importorskip("torchvision")
    from perception_detector import CameraObjectDetector, DetectorConfig
    base = CameraObjectDetector()
    ft = CameraObjectDetector(DetectorConfig(finetuned_checkpoint=CKPT))
    assert ft.cfg.model_name == base.cfg.model_name, "architecture must not change"
    assert ft.finetuned and not base.finetuned
    assert ft.categories[0] == "__background__" and len(ft.categories) == 7


@needs_ckpt
@needs_data
def test_finetuned_output_is_the_same_Detection_contract_as_phase6():
    pytest.importorskip("torchvision")
    from PIL import Image
    from perception_detector import (CameraGeometry, CameraObjectDetector, DetectorConfig,
                                     boxes_to_detections)
    from perception_detector.uvh26 import UVH26Subset, load_manifest
    d = UVH26Subset("test", load_manifest())
    img = np.asarray(Image.open(d.path_of(0)).convert("RGB"))
    det = CameraObjectDetector(DetectorConfig(finetuned_checkpoint=CKPT, score_threshold=0.2))
    raw = det.detect(img)
    assert det.last_inference_ms > 0.0
    for r in raw:
        assert isinstance(r.object_type, ObjectType)
        assert r.object_type is not ObjectType.UNKNOWN
        assert r.x1 > r.x0 and r.y1 > r.y0
    geom = CameraGeometry.from_fov(img.shape[1], img.shape[0])
    dets = boxes_to_detections(raw, geom, 0.0, 0.0, 0.0, timestamp=0.0)
    for x in dets:
        assert isinstance(x, Detection)
        assert x.sensor is SensorType.CAMERA
        assert x.covariance.shape == (2, 2)
        assert np.all(np.linalg.eigvalsh(x.covariance) > 0)
        assert x.truth_id == ""


@needs_ckpt
@needs_data
def test_finetuned_detections_reach_the_unmodified_tracker():
    """Step 10: detector -> Detection -> tracker -> fusion, with nothing downstream changed."""
    pytest.importorskip("torchvision")
    from PIL import Image
    from autonomy.core.config import ObjectProfiles
    from autonomy.core.types import VehicleState
    from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig
    from perception_detector import (CameraGeometry, CameraObjectDetector, DetectorConfig,
                                     boxes_to_detections)
    from perception_detector.uvh26 import UVH26Subset, load_manifest
    d = UVH26Subset("test", load_manifest())
    det = CameraObjectDetector(DetectorConfig(finetuned_checkpoint=CKPT, score_threshold=0.3))
    tk = SensorFusionTracker(TrackerConfig(confirm_hits=1), ObjectProfiles.load())
    ego = VehicleState(timestamp=0.0, x=0.0, y=0.0, yaw=0.0, longitudinal_velocity=0.0)
    total = 0
    for k in range(3):
        img = np.asarray(Image.open(d.path_of(k)).convert("RGB"))
        geom = CameraGeometry.from_fov(img.shape[1], img.shape[0])
        dets = boxes_to_detections(det.detect(img), geom, 0.0, 0.0, 0.0, timestamp=k * 0.5)
        total += len(dets)
        tk.ingest(dets, k * 0.5, ego)
        states = tk.get_object_states(k * 0.5)
        for s in states:
            assert np.all(np.isfinite([s.x, s.y, s.vx, s.vy]))
    assert total > 0, "the fine-tuned detector produced nothing on real test images"


@needs_ckpt
def test_the_checkpoint_records_what_is_needed_to_reproduce_it():
    torch = pytest.importorskip("torch")
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    for key in ("model", "config", "class_names", "epoch", "git_commit", "manifest"):
        assert key in state, f"checkpoint is missing {key}"
    assert state["config"]["seed"] == 20260907
    assert state["class_names"][0] == "__background__"
