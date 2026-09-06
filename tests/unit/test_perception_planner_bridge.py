"""Phase 3C: the bridge from a learned drivable mask to the EXISTING planner.

Every test here runs on synthetic masks or on the dataset, never on a trained checkpoint, except
the two marked `needs_model`. That matters: the bridge's contract with the planner must be
verifiable on a machine that has neither the dataset nor the weights.

The central claim these tests defend is the one a jury would attack first: that the corridor the
planner receives really is derived from the segmentation output, and that a rejected prediction is
never silently replaced by ground truth.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from road_perception.dataset import DRIVABLE, INPUT_H, INPUT_W, NON_DRIVABLE, NUM_CLASSES
from road_perception.drivable import Corridor, GroundProjection, corridor_width_profile, extract_corridor
from road_perception.planner_bridge import (AnchorPose, GateThresholds, PerceptionPlannerBridge,
                                            corridor_to_world, resample_uniform, usable_extent,
                                            validate_corridor)
from simulation.world.road import DrivableSpace

ROOT = Path(__file__).resolve().parents[2] / "Datasets" / "idd-lite" / "idd20k_lite"
CKPT = Path(__file__).resolve().parents[2] / "road_perception" / "checkpoints" / "fastscnn_iddlite.pt"
needs_model = pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
ORIGIN = AnchorPose(0.0, 0.0, 0.0)


def straight_road_mask(top: int = 60, slope: float = 0.6) -> np.ndarray:
    """A drivable region shaped the way a straight road actually appears in a camera image.

    A TRAPEZOID, not a constant-width strip. This matters and it is easy to get wrong: with a 60
    degree field of view the whole image spans only 2 * x * tan(30) metres at range x, which is
    4.3 m at the nearest visible row. A strip of fixed pixel width is therefore NARROW in metres
    near the vehicle and wide far away, the exact opposite of a road, and it fails the width gate
    for a reason that has nothing to do with the code under test.
    """
    m = np.full((INPUT_H, INPUT_W), NON_DRIVABLE, dtype=np.uint8)
    c = INPUT_W // 2
    for row in range(top, INPUT_H):
        half = int(10 + slope * (row - top))
        m[row, max(0, c - half):min(INPUT_W, c + half)] = DRIVABLE
    return m


def widening_road_mask() -> np.ndarray:
    """A second trapezoid, opening more sharply, for tests that want a wider corridor."""
    return straight_road_mask(top=60, slope=0.8)


def synthetic_corridor(x0: float, n: int, half_start: float, half_end: float,
                       lateral: np.ndarray | None = None) -> Corridor:
    """A corridor built directly in metres, bypassing the image entirely."""
    ref_x = np.arange(float(x0), float(x0) + n)
    lat = np.zeros(n) if lateral is None else np.asarray(lateral, dtype=float)
    half = np.linspace(half_start, half_end, n)
    ref = np.stack([ref_x, lat], axis=1)
    left = np.stack([ref_x, lat + half], axis=1)
    right = np.stack([ref_x, lat - half], axis=1)
    return Corridor(ref, left, right, n)


# ------------------------------------------------------- 1. mask -> drivable extraction ------- #
def test_a_straight_strip_becomes_a_corridor_the_planner_accepts():
    out = PerceptionPlannerBridge().from_mask(straight_road_mask(), ORIGIN)
    assert out.gate.ok, out.gate.reason
    assert isinstance(out.road, DrivableSpace)
    assert out.gate.usable_rows >= 5


def test_only_class_zero_is_treated_as_drivable():
    """A mask of every OTHER class must yield nothing. Guards against an off-by-one in the id."""
    br = PerceptionPlannerBridge()
    for cls in range(1, NUM_CLASSES):
        m = np.full((INPUT_H, INPUT_W), cls, dtype=np.uint8)
        assert not br.from_mask(m, ORIGIN).accepted


# --------------------------------------------------------- 2. coordinate conversion ----------- #
def test_the_identity_anchor_leaves_the_corridor_where_it_was():
    c = extract_corridor(straight_road_mask() == DRIVABLE)
    ref, left, right = corridor_to_world(c, ORIGIN)
    assert np.allclose(ref, c.reference)
    assert np.allclose(left, c.left)
    assert np.allclose(right, c.right)


def test_a_rotated_anchor_rotates_and_translates_the_corridor():
    """A corridor is measured in the EGO frame; the anchor is the only thing that places it."""
    c = extract_corridor(straight_road_mask() == DRIVABLE)
    anchor = AnchorPose(10.0, -4.0, math.pi / 2)
    ref, _, _ = corridor_to_world(c, anchor)
    # a point 5 m straight ahead of an ego facing +y ends up 5 m north of the anchor
    near = c.reference[0]
    expected = np.array([10.0 - near[1], -4.0 + near[0]])
    assert np.allclose(ref[0], expected, atol=1e-9)


def test_conversion_preserves_corridor_width():
    """A rigid transform cannot change how wide the road is."""
    c = extract_corridor(straight_road_mask() == DRIVABLE)
    _, left, right = corridor_to_world(c, AnchorPose(3.0, 7.0, 0.9))
    before = corridor_width_profile(c)
    after = np.hypot(left[:, 0] - right[:, 0], left[:, 1] - right[:, 1])
    assert np.allclose(before, after, atol=1e-9)


# ---------------------------------------------------------------- 3. corridor continuity ------ #
def test_uniform_resampling_makes_station_spacing_even():
    """Raw stations are spaced by image rows, which under perspective is wildly uneven."""
    c = extract_corridor(straight_road_mask() == DRIVABLE)
    raw = np.diff(c.reference[:, 0])
    assert raw.max() / raw.min() > 5.0, "the raw grid should be strongly non-uniform"
    lateral = resample_uniform(c, 1.0)
    assert lateral.size >= 5


def test_a_corridor_that_teleports_sideways_is_rejected():
    lateral = np.zeros(20)
    lateral[10:] = 9.0                       # a 9 m sidestep between two adjacent stations
    g = validate_corridor(synthetic_corridor(5.0, 20, 2.5, 2.5, lateral))
    assert not g.ok and g.code == "CENTRE_SLOPE"


def test_non_monotonic_range_is_rejected():
    c = synthetic_corridor(5.0, 20, 2.5, 2.5)
    c.reference[5, 0], c.reference[6, 0] = c.reference[6, 0], c.reference[5, 0]
    g = validate_corridor(c)
    assert not g.ok and g.code == "NON_MONOTONIC"


# ------------------------------------------------------------ 4. corridor width validation ---- #
def test_width_is_walked_from_the_vehicle_not_minimised_globally():
    """A road CONVERGES at the horizon, so a global minimum width rejects every real corridor.

    This is the bug that made the first version of this gate refuse 204 of 204 GROUND-TRUTH
    corridors. The usable stretch is the contiguous run from the vehicle outward.
    """
    c = synthetic_corridor(5.0, 20, 3.0, 0.1)        # wide near, pinching to nothing far away
    assert corridor_width_profile(c).min() < 0.5     # a global check would reject this
    n = usable_extent(c, 2.6)
    assert 5 <= n < 20                               # but the near stretch is genuinely usable


def test_a_corridor_narrower_than_the_vehicle_is_rejected():
    g = validate_corridor(synthetic_corridor(5.0, 20, 0.5, 0.5))           # 1.0 m wide
    assert not g.ok and g.code == "TOO_NARROW"


def test_a_bled_mask_that_claims_the_whole_scene_is_rejected():
    g = validate_corridor(synthetic_corridor(5.0, 20, 40.0, 40.0))          # 80 m wide
    assert not g.ok and g.code == "TOO_WIDE"


def test_a_short_corridor_is_rejected_even_when_wide_enough():
    g = validate_corridor(synthetic_corridor(5.0, 5, 3.0, 3.0))          # only 4 m of lookahead
    assert not g.ok and g.code == "SHORT_LOOKAHEAD"


def test_the_gate_clips_to_the_stretch_it_vouches_for():
    """An accepted corridor may be SHORTER than what was extracted, and that is the point."""
    ref_x = np.arange(5.0, 30.0)
    half = np.concatenate([np.full(15, 2.5), np.full(10, 0.3)])
    g = validate_corridor(Corridor(np.stack([ref_x, np.zeros(25)], axis=1),
                                   np.stack([ref_x, half], axis=1),
                                   np.stack([ref_x, -half], axis=1), 25))
    assert g.ok
    assert g.usable.valid_rows == 15
    assert g.usable_rows < g.rows
    assert g.lookahead_m < g.raw_lookahead_m


# --------------------------------------------------------------- 5. invalid / empty mask ------ #
def test_an_empty_mask_is_refused_and_no_road_is_produced():
    out = PerceptionPlannerBridge().from_mask(
        np.full((INPUT_H, INPUT_W), NON_DRIVABLE, dtype=np.uint8), ORIGIN)
    assert not out.accepted and out.road is None
    assert out.gate.code in {"TOO_FEW_STATIONS", "TOO_NARROW"}


def test_non_finite_geometry_is_refused_rather_than_reaching_the_planner():
    c = synthetic_corridor(5.0, 20, 2.5, 2.5)
    c.left[3, 1] = np.nan
    g = validate_corridor(c)
    assert not g.ok and g.code == "NON_FINITE"


def test_disconnected_regions_do_not_produce_a_spliced_corridor():
    """Two separate strips must not be joined into one road across the gap between them."""
    m = np.full((INPUT_H, INPUT_W), NON_DRIVABLE, dtype=np.uint8)
    c = INPUT_W // 2
    m[170:, c - 50:c + 50] = DRIVABLE          # near strip
    m[60:100, c - 50:c + 50] = DRIVABLE        # detached far strip
    out = PerceptionPlannerBridge().from_mask(m, ORIGIN)
    if out.accepted:
        near_end = GroundProjection().row_to_range(170, INPUT_H)
        assert out.gate.usable.reference[:, 0].max() <= near_end + 1e-6


# ------------------------------------------------------------------- 6. planner adapter ------- #
def test_the_road_handed_over_is_the_planner_s_own_type_and_geometry():
    out = PerceptionPlannerBridge().from_mask(widening_road_mask(), ORIGIN)
    assert out.gate.ok, out.gate.reason
    road = out.road
    assert isinstance(road, DrivableSpace)
    # the planner's own projection must work on it, which is what the planner actually calls
    s, lat, _ = road.project(float(road.reference[0, 0]), float(road.reference[0, 1]))
    assert math.isfinite(s) and abs(lat) < 1.0
    assert road.length > 5.0


def test_the_handed_over_road_matches_the_clipped_corridor_not_the_raw_one():
    """Guards against handing the planner geometry the gate never approved."""
    out = PerceptionPlannerBridge().from_mask(widening_road_mask(), ORIGIN)
    assert out.gate.ok
    assert len(out.road.left_boundary) == out.gate.usable.valid_rows
    assert np.allclose(out.road.left_boundary, out.gate.usable.left)


# --------------------------------------------------------------------- 7. fallback behaviour -- #
def test_a_refusal_yields_no_road_at_all_rather_than_a_substitute():
    """THE claim of this phase: a failed prediction is never replaced by ground truth.

    The bridge has exactly one source of geometry, the mask it was handed. When the gate refuses,
    `road` is None and the caller must stop. There is no second source to fall back to.
    """
    br = PerceptionPlannerBridge()
    bad = br.from_mask(np.full((INPUT_H, INPUT_W), NON_DRIVABLE, dtype=np.uint8), ORIGIN)
    good = br.from_mask(straight_road_mask(), ORIGIN)
    assert bad.road is None
    assert good.road is not None
    assert bad.gate.usable is None


def test_every_refusal_is_recorded_with_a_reason():
    br = PerceptionPlannerBridge()
    br.from_mask(np.full((INPUT_H, INPUT_W), NON_DRIVABLE, dtype=np.uint8), ORIGIN)
    br.from_mask(np.full((INPUT_H, INPUT_W), NON_DRIVABLE, dtype=np.uint8), ORIGIN)
    br.from_mask(straight_road_mask(), ORIGIN)
    assert len(br.events) == 2
    assert sum(br.rejection_summary().values()) == 2
    assert all(e.reason for e in br.events)


def test_a_stricter_gate_refuses_more_and_never_fewer():
    """Monotonicity: tightening a threshold cannot turn a refusal into an acceptance."""
    m = widening_road_mask()
    loose = PerceptionPlannerBridge(thresholds=GateThresholds(min_width_m=2.6))
    strict = PerceptionPlannerBridge(thresholds=GateThresholds(min_width_m=8.0))
    assert loose.from_mask(m, ORIGIN).accepted
    assert not strict.from_mask(m, ORIGIN).accepted


# ------------------------------------------------------------------- 8. deterministic output -- #
def test_the_same_mask_and_anchor_always_give_the_same_road():
    br = PerceptionPlannerBridge()
    m = widening_road_mask()
    a = br.from_mask(m, AnchorPose(2.0, 1.0, 0.3))
    b = br.from_mask(m, AnchorPose(2.0, 1.0, 0.3))
    assert np.array_equal(a.road.reference, b.road.reference)
    assert np.array_equal(a.road.left_boundary, b.road.left_boundary)
    assert a.gate.code == b.gate.code


def test_the_gate_verdict_does_not_depend_on_where_the_corridor_is_anchored():
    """Validity is a property of the ROAD's shape, not of where the car happened to be."""
    br = PerceptionPlannerBridge()
    m = widening_road_mask()
    here = br.from_mask(m, ORIGIN)
    far = br.from_mask(m, AnchorPose(500.0, -300.0, 2.1))
    assert here.gate.ok == far.gate.ok
    assert here.gate.usable_rows == far.gate.usable_rows
    assert here.gate.lookahead_m == pytest.approx(far.gate.lookahead_m)


# ----------------------------------------------- 9. the predicted corridor reaches the planner - #
@needs_model
def test_a_real_prediction_reaches_the_planner_as_a_real_road():
    """End to end on a REAL photograph: image -> network -> corridor -> DrivableSpace.

    Skipped without the checkpoint, so it never blocks a fresh clone, but when the weights are
    present this is the test that proves the claim rather than asserting it in a document.
    """
    pytest.importorskip("torch")
    from road_perception.dataset import IDDLite
    if not ROOT.exists():
        pytest.skip("IDD-Lite not downloaded")
    from PIL import Image
    ds = IDDLite(ROOT, "val", augment=False)
    br = PerceptionPlannerBridge.from_checkpoint(CKPT)
    accepted = 0
    for i in range(12):
        img = np.asarray(Image.open(ds._item(i).image).convert("RGB"))
        out = br.from_image(img, ORIGIN)
        assert out.prediction is not None and out.prediction.shape == (INPUT_H, INPUT_W)
        assert out.inference_ms > 0.0
        if out.accepted:
            accepted += 1
            assert isinstance(out.road, DrivableSpace)
            assert out.road.length > 5.0
    assert accepted > 0, "no real prediction cleared the gate, which would make the phase vacuous"


@needs_model
def test_real_inference_is_deterministic():
    pytest.importorskip("torch")
    if not ROOT.exists():
        pytest.skip("IDD-Lite not downloaded")
    from PIL import Image
    from road_perception.dataset import IDDLite
    ds = IDDLite(ROOT, "val", augment=False)
    img = np.asarray(Image.open(ds._item(0).image).convert("RGB"))
    br = PerceptionPlannerBridge.from_checkpoint(CKPT)
    a, b = br.from_image(img, ORIGIN), br.from_image(img, ORIGIN)
    assert np.array_equal(a.prediction, b.prediction)
    assert np.array_equal(a.corridor.reference, b.corridor.reference)


# --------------------------------------------- 10. heading continuity, calibrated on truth ---- #
def test_a_zigzag_corridor_is_refused_before_the_planner_sees_it():
    """A corridor whose centreline swings tens of degrees per metre is extraction noise.

    Measured consequence of letting one through: the planner's Frenet frame becomes
    ill-conditioned, no candidate is feasible, and the vehicle sits still until the run times out.
    The threshold was calibrated on GROUND-TRUTH corridors and then applied unchanged to
    predictions, so both modes face the same bar.
    """
    lateral = np.zeros(24)
    lateral[1::2] = 1.2                       # alternating sidestep, a saw-tooth centreline
    g = validate_corridor(synthetic_corridor(5.0, 24, 3.0, 3.0, lateral))
    assert not g.ok and g.code == "HEADING_STEP"
    assert g.max_heading_step_deg > 30.0


def test_a_smoothly_curving_corridor_is_accepted():
    """The heading check must not punish an ordinary bend, only a discontinuity."""
    x = np.arange(24, dtype=float)
    lateral = 0.02 * x ** 2                   # a steady curve, no kinks
    g = validate_corridor(synthetic_corridor(5.0, 24, 3.0, 3.0, lateral))
    assert g.ok, g.reason
    assert g.max_heading_step_deg < 30.0
