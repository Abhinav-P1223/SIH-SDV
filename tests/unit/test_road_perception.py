"""Phase 3B: learned drivable-space perception from IDD-Lite.

Tests that need the dataset skip cleanly when it is absent, so the suite stays green on a machine
that has not downloaded it. Tests that need a trained checkpoint skip when there is none. The
rest run everywhere, because the loader, loss, metrics and corridor extraction are all testable
on synthetic inputs.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from road_perception.dataset import (CLASS_NAMES, DRIVABLE, IGNORE_INDEX, INPUT_H, INPUT_W,
                                     NON_DRIVABLE, NUM_CLASSES, IDDLite, class_weights,
                                     find_pairs)
from road_perception.drivable import (Corridor, GroundProjection, clean_mask,
                                      corridor_width_profile, drivable_mask, extract_corridor)
from road_perception.evaluate import (SegmentationMetrics, binary_boundary, boundary_f1,
                                      confusion_matrix, iou_from_confusion)
from road_perception.integration import DrivableSpacePerception, corridor_agreement
from road_perception.losses import DrivableSegmentationLoss, drivable_boundary
from road_perception.model import FastSCNN, parameter_count

ROOT = Path(__file__).resolve().parents[2] / "Datasets" / "idd-lite" / "idd20k_lite"
CKPT = Path(__file__).resolve().parents[2] / "road_perception" / "checkpoints" / "fastscnn_iddlite.pt"
needs_data = pytest.mark.skipif(not ROOT.exists(), reason="IDD-Lite not downloaded")
needs_model = pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")


# ------------------------------------------------------------- 1/2. loader ------------------ #
@needs_data
def test_loader_splits_match_the_audited_counts():
    assert len(find_pairs(ROOT, "train")) == 1403
    assert len(find_pairs(ROOT, "val")) == 204
    assert len(find_pairs(ROOT, "test")) == 404


@needs_data
def test_every_train_and_val_image_has_a_mask_and_test_has_none():
    for split, labelled in (("train", True), ("val", True), ("test", False)):
        pairs = find_pairs(ROOT, split)
        assert all((p.mask is not None) == labelled for p in pairs), split
        for p in pairs[:20]:
            assert p.image.exists()
            if p.mask:
                assert p.mask.exists()
                # the mask must belong to THIS image, not merely exist
                assert p.mask.stem.replace("_label", "") == p.image.stem.replace("_image", "")


@needs_data
def test_items_have_the_right_shapes_dtypes_and_labels():
    ds = IDDLite(ROOT, "train")
    x, y = ds[0]
    assert x.shape == (3, INPUT_H, INPUT_W) and x.dtype == torch.float32
    assert y.shape == (INPUT_H, INPUT_W) and y.dtype == torch.int64
    labels = set(np.unique(y.numpy()).tolist())
    assert labels <= set(range(NUM_CLASSES)) | {IGNORE_INDEX}, f"unexpected label ids: {labels}"


@needs_data
def test_masks_are_resized_without_inventing_class_ids():
    """NEAREST resampling only: bilinear on a label map would create ids that do not exist."""
    ds = IDDLite(ROOT, "train")
    for i in (0, 5, 50):
        _, y = ds[i]
        assert set(np.unique(y.numpy()).tolist()) <= set(range(NUM_CLASSES)) | {IGNORE_INDEX}


@needs_data
def test_test_split_is_never_used_for_supervision():
    """Test masks do not exist; the loader must surface ignore, not a fabricated label."""
    ds = IDDLite(ROOT, "test")
    _, y = ds[0]
    assert (y.numpy() == IGNORE_INDEX).all()


# ------------------------------------------------------- 3/5. class weights ----------------- #
def test_class_weights_lift_rare_classes_and_are_capped():
    counts = np.array([3_200_000, 240_000, 210_000, 1_140_000, 1_250_000, 2_390_000, 1_480_000])
    w = class_weights(counts, cap=12.0)
    assert w.shape == (NUM_CLASSES,)
    assert w[NON_DRIVABLE] > w[DRIVABLE] * 5, "the rare class must carry far more weight"
    assert w.max() <= 12.0 + 1e-6
    assert np.all(np.isfinite(w)) and np.all(w > 0)


def test_class_weights_survive_an_absent_class():
    counts = np.array([1000, 0, 500, 500, 500, 500, 500])
    w = class_weights(counts)
    assert np.all(np.isfinite(w)) and w[1] == 0.0        # absent -> no weight, never inf


# ------------------------------------------------------------- 4. ignore label -------------- #
def test_ignore_pixels_contribute_nothing_to_the_loss():
    loss = DrivableSegmentationLoss(torch.ones(NUM_CLASSES), boundary_weight=1.0)
    logits = torch.randn(2, NUM_CLASSES, 16, 16)
    target = torch.full((2, 16, 16), IGNORE_INDEX, dtype=torch.long)
    assert float(loss(logits, target)) == pytest.approx(0.0, abs=1e-6)

    target[:, :8, :] = DRIVABLE                          # half real, half ignored
    assert float(loss(logits, target)) > 0.0


def test_ignore_pixels_are_excluded_from_the_confusion_matrix():
    pred = np.zeros((4, 4), dtype=np.int64)
    target = np.full((4, 4), IGNORE_INDEX, dtype=np.int64)
    target[0, :] = DRIVABLE
    cm = confusion_matrix(pred, target)
    assert cm.sum() == 4                                 # only the one real row counted


def test_boundary_weighting_targets_the_drivable_edge():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[:, :, 4:] = NON_DRIVABLE                      # a vertical edge at column 4
    b = drivable_boundary(target)
    assert b.any() and b[0, :, 3:5].all()                # the edge columns are boundary
    assert not b[0, :, 0].any()                          # deep interior is not

    # The property that matters: an error AT the edge must cost more than the same error deep
    # inside a region. A uniform-error probe cannot show this, because the loss is a weighted
    # MEAN and uniform error cancels in both numerator and denominator.
    plain = DrivableSegmentationLoss(torch.ones(NUM_CLASSES), boundary_weight=1.0)
    weighted = DrivableSegmentationLoss(torch.ones(NUM_CLASSES), boundary_weight=5.0)

    def logits_wrong_at(col: int) -> torch.Tensor:
        lg = torch.zeros(1, NUM_CLASSES, 8, 8)
        lg.scatter_(1, target.unsqueeze(1), 5.0)         # correct everywhere ...
        lg[0, :, :, col] = 0.0
        lg[0, (DRIVABLE if target[0, 0, col] != DRIVABLE else NON_DRIVABLE), :, col] = 5.0
        return lg                                        # ... except one wrong column

    at_edge = logits_wrong_at(4)                         # first non-drivable column: on the edge
    interior = logits_wrong_at(7)                        # far from the edge
    assert float(plain(at_edge, target)) == pytest.approx(float(plain(interior, target)), rel=1e-6)
    assert float(weighted(at_edge, target)) > float(weighted(interior, target))


# ------------------------------------------------------------ 6/7. model output ------------- #
def test_model_output_shape_matches_the_input():
    m = FastSCNN(NUM_CLASSES)
    for h, w in ((INPUT_H, INPUT_W), (128, 160)):
        out = m(torch.randn(1, 3, h, w))
        assert out.shape == (1, NUM_CLASSES, h, w)
    assert 0.3e6 < parameter_count(m) < 2.0e6            # stays a lightweight model


def test_model_runs_in_train_mode_at_batch_size_one():
    """The pyramid's 1x1 bin makes BatchNorm fail here; the architecture must not regress."""
    m = FastSCNN(NUM_CLASSES).train()
    assert m(torch.randn(1, 3, 64, 64)).shape == (1, NUM_CLASSES, 64, 64)


def test_metrics_reject_a_degenerate_all_drivable_prediction():
    """The whole point of the weighted loss: a mask that says 'drivable everywhere' must score
    badly on the metrics we actually rely on, even though pixel accuracy looks respectable."""
    target = np.zeros((100, 100), dtype=np.int64)
    target[:, 80:] = NON_DRIVABLE                        # 20% non-drivable
    pred = np.zeros_like(target)                         # predict drivable everywhere

    m = SegmentationMetrics()
    m.update(pred, target)
    s = m.summary()
    assert s["pixel_accuracy"] == pytest.approx(0.8)     # flattering
    assert s["non_drivable_iou"] == pytest.approx(0.0)   # and useless
    assert s["boundary_f1"] == pytest.approx(0.0)


def test_iou_is_nan_for_a_class_absent_from_both_sides():
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm[DRIVABLE, DRIVABLE] = 10
    iou = iou_from_confusion(cm)
    assert iou[DRIVABLE] == pytest.approx(1.0)
    assert np.isnan(iou[NON_DRIVABLE]), "an absent class must not score a free 1.0"


def test_boundary_f1_is_exact_on_a_known_edge():
    target = np.zeros((20, 20), dtype=np.int64)
    target[:, 10:] = NON_DRIVABLE
    assert boundary_f1(target, target)["f1"] == pytest.approx(1.0)

    shifted = np.zeros((20, 20), dtype=np.int64)
    shifted[:, 12:] = NON_DRIVABLE                       # edge moved 2 px
    assert boundary_f1(shifted, target, tolerance_px=2)["f1"] == pytest.approx(1.0)
    assert boundary_f1(shifted, target, tolerance_px=0)["f1"] == pytest.approx(0.0)

    empty = np.zeros((20, 20), dtype=np.int64)           # no edge at all
    assert boundary_f1(empty, target)["f1"] == pytest.approx(0.0)
    assert boundary_f1(empty, empty)["f1"] == pytest.approx(1.0)


def test_binary_boundary_finds_only_transitions():
    m = np.zeros((6, 6), dtype=bool)
    m[:, 3:] = True
    b = binary_boundary(m)
    assert b[:, 2].all() and b[:, 3].all()
    assert not b[:, 0].any() and not b[:, 5].any()


# --------------------------------------------------------- 8/9. mask and corridor ----------- #
def test_drivable_mask_selects_only_class_zero():
    pred = np.array([[DRIVABLE, NON_DRIVABLE], [3, DRIVABLE]], dtype=np.int64)
    assert drivable_mask(pred).tolist() == [[True, False], [False, True]]


def test_cleaning_drops_speckle_and_fills_holes():
    m = np.zeros((60, 60), dtype=bool)
    m[20:60, 10:50] = True                               # the road
    m[35:38, 25:28] = False                              # a car on it: a hole to close
    m[2:4, 2:4] = True                                   # detached speckle
    c = clean_mask(m)
    assert c[36, 26], "hole over the road was not filled"
    assert not c[2:4, 2:4].any(), "detached speckle survived"


def test_cleaning_an_empty_mask_is_safe():
    assert not clean_mask(np.zeros((20, 20), dtype=bool)).any()


def test_corridor_follows_a_straight_road_and_reports_width():
    m = np.zeros((INPUT_H, INPUT_W), dtype=bool)
    m[120:, 120:200] = True                              # a band up the middle
    c = extract_corridor(m)
    assert c.valid_rows >= 3
    assert c.reference.shape[1] == 2 and c.left.shape == c.right.shape
    assert np.all(c.reference[:, 0] > 0.0)               # ranges are ahead of the vehicle
    assert np.all(c.left[:, 1] > c.right[:, 1]), "left of centre must be the positive side"
    assert (corridor_width_profile(c) > 0).all()


def test_corridor_is_empty_when_nothing_is_drivable():
    c = extract_corridor(np.zeros((INPUT_H, INPUT_W), dtype=bool))
    assert c.valid_rows == 0 and c.reference.shape == (0, 2)
    assert corridor_width_profile(c).size == 0


def test_ground_projection_is_monotonic_and_bounded():
    p = GroundProjection(max_range_m=40.0)
    near = p.row_to_range(INPUT_H - 1, INPUT_H)
    far = p.row_to_range(INPUT_H // 2 + 5, INPUT_H)
    assert 0.0 < near < far <= 40.0, "rows higher in the image must be further away"
    assert p.row_to_range(0, INPUT_H) == pytest.approx(40.0)     # at/above horizon -> clamped
    # left of image centre projects to positive (left) lateral offset
    assert p.col_to_lateral(0, INPUT_W, 10.0) > 0 > p.col_to_lateral(INPUT_W - 1, INPUT_W, 10.0)


# ------------------------------------------------------- 10/11. planner interface ----------- #
def test_corridor_becomes_the_planner_s_own_road_model():
    m = np.zeros((INPUT_H, INPUT_W), dtype=bool)
    m[100:, 110:210] = True
    road = DrivableSpacePerception.to_road_model(extract_corridor(m))
    assert road is not None
    from simulation.world.road import DrivableSpace
    assert isinstance(road, DrivableSpace)               # the contract the planner already takes
    s, d, _ = road.project(float(road.reference[0][0]), float(road.reference[0][1]))
    assert math.isfinite(s) and math.isfinite(d)


def test_too_little_evidence_yields_no_road_rather_than_a_degenerate_one():
    """Handing the planner a corridor built from two pixels would be worse than admitting we
    cannot see."""
    assert DrivableSpacePerception.to_road_model(extract_corridor(
        np.zeros((INPUT_H, INPUT_W), dtype=bool))) is None
    tiny = Corridor(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), 2)
    assert DrivableSpacePerception.to_road_model(tiny) is None


def test_corridor_agreement_is_zero_error_against_itself():
    m = np.zeros((INPUT_H, INPUT_W), dtype=bool)
    m[110:, 120:200] = True
    c = extract_corridor(m)
    a = corridor_agreement(c, c)
    assert a["left_rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert a["right_rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert a["overlap_m"] > 0.0


def test_corridor_agreement_handles_empty_input():
    empty = Corridor(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2)), 0)
    a = corridor_agreement(empty, empty)
    assert a["overlap_m"] == 0.0 and a["left_rmse_m"] is None


# ------------------------------------------------ 11. deterministic inference path ---------- #
@needs_model
@needs_data
def test_inference_is_deterministic_and_produces_a_corridor():
    from PIL import Image
    per = DrivableSpacePerception(CKPT)
    img = np.asarray(Image.open(find_pairs(ROOT, "val")[0].image).convert("RGB"))
    a = per.perceive(img)
    b = per.perceive(img)
    assert np.array_equal(a.prediction, b.prediction), "same image gave a different prediction"
    assert a.prediction.shape == (INPUT_H, INPUT_W)
    assert set(np.unique(a.prediction).tolist()) <= set(range(NUM_CLASSES))
    assert a.inference_ms > 0.0


@needs_model
@needs_data
def test_preprocessing_is_size_agnostic():
    per = DrivableSpacePerception(CKPT)
    for shape in ((227, 320, 3), (480, 640, 3)):
        x = per.preprocess(np.zeros(shape, dtype=np.uint8))
        assert x.shape == (1, 3, INPUT_H, INPUT_W)


# ------------------------------------- 12. regressions found by running the evaluation ------- #
def test_corridor_survives_an_unlabelled_bottom_row():
    """IDD-Lite masks stop two rows short of the image bottom.

    The original scan treated the first empty row as the END of the road, so it broke on its very
    first iteration and returned an empty corridor for EVERY image, ground truth included. Rows
    below the first evidence must be skipped, not treated as the end.
    """
    mask = np.zeros((224, 320), dtype=bool)
    mask[40:222, 120:200] = True          # road present, but rows 222-223 unlabelled
    assert not mask[223].any()
    c = extract_corridor(mask)
    assert c.valid_rows >= 10
    assert DrivableSpacePerception.to_road_model(c) is not None


def test_corridor_still_ends_where_the_evidence_ends():
    """The skip must not become extrapolation: a gap ABOVE the start still terminates."""
    mask = np.zeros((224, 320), dtype=bool)
    mask[180:224, 120:200] = True         # near band
    mask[100:140, 120:200] = True         # detached far band, must not be joined
    c = extract_corridor(mask)
    assert c.valid_rows > 0
    # every sampled range must come from the near band, which is closer than the far band
    near_max = GroundProjection().row_to_range(180, 224)
    assert c.reference[:, 0].max() <= near_max + 1e-6


def test_boundary_score_never_leaks_across_a_batch():
    """A batched update must equal scoring each image on its own.

    `distance_transform_edt` over a (B, H, W) array treats the batch axis as spatial, letting a
    boundary pixel in one image satisfy a boundary pixel in the next. That silently inflated the
    training-log boundary F1. IoU was unaffected, so this is a reporting bug, not a model bug.
    """
    rng = np.random.default_rng(7)
    pred = rng.integers(0, NUM_CLASSES, size=(4, 40, 60)).astype(np.uint8)
    target = rng.integers(0, NUM_CLASSES, size=(4, 40, 60)).astype(np.uint8)

    batched = SegmentationMetrics()
    batched.update(pred, target)

    single = SegmentationMetrics()
    for i in range(4):
        single.update(pred[i], target[i])

    assert len(batched.boundary) == 4
    assert batched.summary()["boundary_f1"] == pytest.approx(single.summary()["boundary_f1"])
    assert np.array_equal(batched.cm, single.cm)


def test_binary_boundary_refuses_a_batched_array():
    with pytest.raises(ValueError):
        binary_boundary(np.zeros((2, 8, 8), dtype=bool))
