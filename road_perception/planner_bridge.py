"""Phase 3C: the adapter that lets a LEARNED drivable corridor drive the EXISTING planner.

WHAT THIS MODULE IS
    A one-way bridge. It takes a predicted segmentation mask and produces the planner's own
    `DrivableSpace`, or refuses. Nothing under `autonomy/` or `simulation/` is modified, imported
    into, or aware of this file. The planner cannot tell the difference between a corridor that
    came from scenario YAML and one that came from a camera, which is the whole point.

        predicted mask -> drivable mask -> cleanup -> corridor (ego frame)
                       -> VALIDATION GATE -> anchor to a pose -> DrivableSpace (world frame)

THE GATE IS THE POINT
    Phase 3B measured the model's weakness precisely: 46 of 204 validation images have a corridor
    boundary off by more than 3 m on one side. A corridor that wrong must not reach a planner that
    trusts its road model. So every corridor passes an explicit, inspectable gate first, and a
    rejected corridor is REJECTED: it is not repaired, not smoothed into plausibility, and above
    all not quietly replaced by the simulator's ground-truth road. `GateResult.reason` says which
    check failed, and the caller is expected to record it and stop the vehicle.

FRAME CONVENTION, STATED ONCE
    `extract_corridor` returns metres in the EGO frame at the moment of capture: x forward, y
    left. `DrivableSpace` is in the world frame. `AnchorPose` is the only thing that relates them,
    and it is a required argument with no default, so a caller cannot forget that a corridor
    without a pose is meaningless. This module never invents a pose and never integrates ego
    motion; a still photograph carries neither.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulation.world.road import DrivableSpace

from .drivable import (Corridor, GroundProjection, clean_mask, corridor_width_profile,
                       extract_corridor)
from .integration import DrivableSpacePerception


# --------------------------------------------------------------------------------------------- #
# 1. The validation gate
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateThresholds:
    """Every number a corridor must clear, in one place, so a reviewer can argue with them.

    Defaults come from the vehicle and from Phase 3B's measured error, not from whatever made the
    pass rate look good. Two of them exist because of how a road behaves under perspective, and
    both were added after the first version of this gate rejected 204 of 204 GROUND-TRUTH
    corridors, which is how you find out the gate is wrong rather than the data.
    """
    min_rows: int = 5                       # fewer stations than this is not a road
    min_lookahead_m: float = 8.0            # usable distance the car can plan a stop inside
    min_width_m: float = 2.6                # vehicle width 1.8 m plus 0.4 m each side
    max_width_m: float = 30.0               # wider than any real corridor: the mask has bled
    max_centre_slope: float = 3.0           # centreline drift per metre of range, on a 1 m grid
    max_heading_step_deg: float = 30.0      # centreline heading change between 1 m stations
    max_range_gap_m: float = 6.0            # absolute floor for the station-spacing check
    range_gap_fraction: float = 0.45        # spacing allowance that grows with range, see below
    resample_step_m: float = 1.0            # uniform grid the continuity check is measured on

    # THERE IS DELIBERATELY NO WIDTH-SLOPE CHECK.
    # Under a fixed field of view the corridor cannot be wider than the view cone, which spans
    # 2 * x * tan(hfov / 2) at range x. At 60 degrees that is 1.16 m of width per metre of range,
    # so a well-formed corridor's width MUST climb at roughly that rate through the near field.
    # Measured on ground-truth corridors the width slope runs 1.19 m/m median and 6.1 m/m at p99,
    # so any threshold either admits everything or rejects valid ground truth. Width is guarded
    # instead by a floor (`min_width_m`, walked contiguously from the vehicle) and a ceiling
    # (`max_width_m`), both of which measure the quantity that actually matters.


@dataclass(frozen=True)
class GateResult:
    """Why a corridor was accepted or refused, and what part of it survived.

    `usable` is the corridor actually handed onward: the near portion that cleared every check,
    which may be shorter than what was extracted. Clipping is deliberate. Handing the planner only
    the stretch we can vouch for is honest; handing it the whole thing and hoping is not.
    """
    ok: bool
    reason: str = ""                # human-readable, carries the measured value
    code: str = ""                  # stable machine-readable cause, for grouping and for tests
    rows: int = 0                   # stations extracted
    usable_rows: int = 0            # stations that survived the width walk
    lookahead_m: float = 0.0        # usable lookahead, not raw extent
    raw_lookahead_m: float = 0.0
    min_width_m: float = 0.0        # narrowest station WITHIN the usable stretch
    median_width_m: float = 0.0
    max_centre_slope: float = 0.0   # steepest centreline drift per metre, on the uniform grid
    max_heading_step_deg: float = 0.0
    usable: Corridor | None = None

    def __str__(self) -> str:
        return "accepted" if self.ok else f"rejected [{self.code}]: {self.reason}"


def _slice(c: Corridor, n: int) -> Corridor:
    return Corridor(c.reference[:n], c.left[:n], c.right[:n], n)


def resample_uniform(c: Corridor, step_m: float = 1.0) -> np.ndarray:
    """Centreline lateral offsets on an evenly spaced range grid.

    The raw stations are spaced by IMAGE ROWS, which under perspective means 0.12 m apart at the
    bumper and 9.6 m apart near the horizon. A finite difference on that grid is meaningless: the
    same physical kink reads as an enormous slope near the vehicle and a negligible one far away.
    Resampling onto an even grid first is what makes a continuity threshold mean one thing.
    """
    x = c.reference[:, 0]
    grid = np.arange(float(x.min()), float(x.max()) + 1e-9, step_m)
    if grid.size < 2:
        return np.zeros(0)
    return np.interp(grid, x, c.reference[:, 1])


def usable_extent(corridor: Corridor, min_width_m: float) -> int:
    """How many stations, from the vehicle outward, are continuously wide enough to drive.

    A road CONVERGES under perspective: its farthest station sits near the vanishing point by
    construction, so the narrowest width across a whole corridor is always tiny and testing it
    rejects everything, ground truth included. What matters is the contiguous stretch in front of
    the car that is genuinely wide enough. This walks outward from the near end and stops at the
    first station that pinches below `min_width_m`.
    """
    n = 0
    for w in corridor_width_profile(corridor):
        if w < min_width_m:
            break
        n += 1
    return n


def validate_corridor(corridor: Corridor,
                      thresholds: GateThresholds | None = None) -> GateResult:
    """Decide whether a predicted corridor is safe to hand to the planner, and clip it if so.

    Ordered cheapest-first, returning on the FIRST failure so the reason is unambiguous.

      * stations / finiteness  -- the model saw almost nothing, or a NaN would enter the planner's
                                  arc-length maths and propagate silently
      * monotonic range        -- stations must march away from the vehicle
      * usable width walk      -- the contiguous near stretch the car actually fits inside
      * usable lookahead       -- that stretch must be long enough to plan a stop within
      * station spacing        -- a genuine hole in coverage, allowing for the fact that equal
                                  steps in image rows map to growing steps in ground range
      * width ceiling          -- the drivable mask has bled into pavement or sky
      * width / centre jumps   -- a boundary that steps sideways between adjacent stations is
                                  noise, and a corridor that teleports laterally would whip the
                                  controller
    """
    t = thresholds or GateThresholds()
    if corridor.valid_rows < t.min_rows:
        return GateResult(False, f"too few stations ({corridor.valid_rows} < {t.min_rows})",
                          "TOO_FEW_STATIONS", rows=corridor.valid_rows)

    ref, left, right = corridor.reference, corridor.left, corridor.right
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return GateResult(False, "non-finite corridor coordinates", "NON_FINITE",
                          rows=corridor.valid_rows)
    if np.any(np.diff(ref[:, 0]) <= 0.0):
        return GateResult(False, "range stations are not strictly increasing", "NON_MONOTONIC",
                          rows=corridor.valid_rows)

    raw_extent = float(ref[:, 0].max() - ref[:, 0].min())
    n = usable_extent(corridor, t.min_width_m)
    base = dict(rows=corridor.valid_rows, usable_rows=n, raw_lookahead_m=raw_extent)
    if n < t.min_rows:
        w0 = float(corridor_width_profile(corridor)[0])
        return GateResult(False, f"only {n} stations reach {t.min_width_m:.1f} m wide "
                                 f"(nearest station is {w0:.2f} m)", "TOO_NARROW", **base)

    u = _slice(corridor, n)
    ux = u.reference[:, 0]
    widths = corridor_width_profile(u)
    lookahead = float(ux.max() - ux.min())
    steps = np.diff(ux)
    centre = resample_uniform(u, t.resample_step_m)
    c_slope = (float(np.abs(np.diff(centre)).max() / t.resample_step_m)
               if centre.size > 1 else 0.0)
    heading = np.degrees(np.arctan2(np.diff(centre), t.resample_step_m)) if centre.size > 1         else np.zeros(0)
    h_step = float(np.abs(np.diff(heading)).max()) if heading.size > 1 else 0.0
    common = dict(lookahead_m=lookahead, min_width_m=float(widths.min()),
                  median_width_m=float(np.median(widths)), max_centre_slope=c_slope,
                  max_heading_step_deg=h_step, **base)

    if lookahead < t.min_lookahead_m:
        return GateResult(False, f"usable lookahead {lookahead:.1f} m below "
                                 f"{t.min_lookahead_m:.1f} m", "SHORT_LOOKAHEAD", **common)
    # Station spacing grows with range because equal steps in IMAGE ROWS are not equal steps on
    # the ground: near the horizon a few rows span many metres. A fixed threshold therefore
    # rejects every well-formed corridor at its far end. The allowance grows with range, so a
    # genuine hole near the vehicle is still caught.
    allowed = np.maximum(t.max_range_gap_m, t.range_gap_fraction * ux[:-1])
    if steps.size and np.any(steps > allowed):
        i = int(np.argmax(steps - allowed))
        return GateResult(False, f"station gap {steps[i]:.1f} m at {ux[i]:.1f} m exceeds "
                                 f"{allowed[i]:.1f} m", "STATION_GAP", **common)
    if float(np.median(widths)) > t.max_width_m:
        return GateResult(False, f"median width {np.median(widths):.1f} m exceeds "
                                 f"{t.max_width_m:.1f} m", "TOO_WIDE", **common)
    if c_slope > t.max_centre_slope:
        return GateResult(False, f"centreline slope {c_slope:.2f} m/m exceeds "
                                 f"{t.max_centre_slope:.2f} m/m", "CENTRE_SLOPE", **common)
    # A drivable road does not change heading by tens of degrees per metre. A corridor that does
    # is extraction noise, and the planner's Frenet frame becomes ill-conditioned inside it: the
    # measured consequence is that no candidate is feasible and the vehicle sits still until the
    # run times out. Calibrated on GROUND-TRUTH corridors, where a 30 degree limit separates the
    # corridors the planner drives from the ones it gets stuck in, then applied unchanged to
    # predictions so the two modes face the same bar.
    if h_step > t.max_heading_step_deg:
        return GateResult(False, f"heading step {h_step:.0f} deg exceeds "
                                 f"{t.max_heading_step_deg:.0f} deg", "HEADING_STEP", **common)
    return GateResult(True, "", "OK", usable=u, **common)


# --------------------------------------------------------------------------------------------- #
# 2. Ego frame -> world frame
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnchorPose:
    """Where the camera was, in world coordinates, when the image was taken.

    There is no default. A corridor measured in the ego frame is meaningless without one, and a
    default would let a caller silently anchor a photograph to the origin without deciding to.
    """
    x: float
    y: float
    yaw: float


def corridor_to_world(corridor: Corridor, anchor: AnchorPose) -> tuple[np.ndarray, ...]:
    """Rotate and translate the three polylines from the ego frame into the world frame."""
    c, s = math.cos(anchor.yaw), math.sin(anchor.yaw)
    rot = np.array([[c, -s], [s, c]])
    offset = np.array([anchor.x, anchor.y])
    return tuple((np.asarray(poly, dtype=float) @ rot.T) + offset
                 for poly in (corridor.reference, corridor.left, corridor.right))


# --------------------------------------------------------------------------------------------- #
# 3. The bridge
# --------------------------------------------------------------------------------------------- #
@dataclass
class BridgeOutput:
    """One image's journey to the planner, with every stage kept for evidence."""
    road: DrivableSpace | None          # None whenever `gate.ok` is False
    gate: GateResult
    corridor: Corridor                  # ego frame, pre-anchor, exactly as extracted
    prediction: np.ndarray | None       # (H, W) class ids, None when a mask was supplied directly
    drivable: np.ndarray                # (H, W) bool, after cleanup
    inference_ms: float
    adapt_ms: float

    @property
    def accepted(self) -> bool:
        return self.road is not None


class PerceptionPlannerBridge:
    """Segmentation mask -> validated `DrivableSpace`, or an explicit refusal.

    Deterministic by construction: eval-mode inference, fixed preprocessing, no randomness in the
    cleanup or the row scan, and a gate made of fixed thresholds. The same image and the same
    anchor always produce the same road, which the tests assert directly.

    Two entry points, both running the IDENTICAL downstream path so that an A/B comparison is a
    comparison of masks and nothing else:
      * `from_image`  -- runs the real network, for Mode B
      * `from_mask`   -- takes a class map already in hand, for Mode A's ground-truth masks and
                         for tests that must not depend on a trained checkpoint
    """

    def __init__(self, perception: DrivableSpacePerception | None = None,
                 projection: GroundProjection | None = None,
                 thresholds: GateThresholds | None = None,
                 row_step: int = 4, min_run_px: int = 8):
        self.perception = perception
        self.projection = projection or (perception.projection if perception else GroundProjection())
        self.thresholds = thresholds or GateThresholds()
        self.row_step = row_step
        self.min_run_px = min_run_px
        self.events: list[GateResult] = []          # every refusal, in order, for reporting

    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoint(cls, checkpoint: Path | str, **kw) -> "PerceptionPlannerBridge":
        return cls(DrivableSpacePerception(checkpoint), **kw)

    def _finish(self, drivable_bool: np.ndarray, anchor: AnchorPose,
                prediction: np.ndarray | None, inference_ms: float) -> BridgeOutput:
        """The single shared path: cleanup -> corridor -> gate -> anchor. Used by BOTH modes."""
        t0 = time.perf_counter()
        mask = clean_mask(drivable_bool)
        corridor = extract_corridor(mask, self.projection, row_step=self.row_step,
                                    min_run_px=self.min_run_px)
        gate = validate_corridor(corridor, self.thresholds)
        road = None
        if gate.ok:
            ref, left, right = corridor_to_world(gate.usable, anchor)
            try:
                road = DrivableSpace(reference=ref, left_boundary=left, right_boundary=right)
            except Exception as exc:                    # a geometry the road model itself refuses
                gate = GateResult(False, f"road model rejected the geometry: {exc}",
                                  "ROAD_MODEL_REJECTED",
                                  rows=gate.rows, usable_rows=gate.usable_rows,
                                  lookahead_m=gate.lookahead_m,
                                  raw_lookahead_m=gate.raw_lookahead_m,
                                  min_width_m=gate.min_width_m, median_width_m=gate.median_width_m,
                                  max_centre_slope=gate.max_centre_slope,
                                  max_heading_step_deg=gate.max_heading_step_deg)
        if not gate.ok:
            self.events.append(gate)
        return BridgeOutput(road, gate, corridor, prediction, mask, inference_ms,
                            (time.perf_counter() - t0) * 1e3)

    def from_image(self, image: np.ndarray, anchor: AnchorPose) -> BridgeOutput:
        """Mode B. Runs the trained network on a real photograph."""
        if self.perception is None:
            raise RuntimeError("no perception model: construct with from_checkpoint() for Mode B")
        from .dataset import DRIVABLE
        pred, infer_ms = self.perception.segment(image)
        return self._finish(pred == DRIVABLE, anchor, pred, infer_ms)

    def from_mask(self, class_map: np.ndarray, anchor: AnchorPose) -> BridgeOutput:
        """Mode A, and tests. Takes a class map that is already known, so no model is needed."""
        from .dataset import DRIVABLE
        return self._finish(np.asarray(class_map) == DRIVABLE, anchor, None, 0.0)

    # ------------------------------------------------------------------ #
    def rejection_summary(self) -> dict[str, int]:
        """How many corridors were refused, grouped by the check that refused them.

        Reported rather than hidden. A phase that silently dropped its failures would be measuring
        only the images it happened to find easy. Grouping uses `GateResult.code`, a fixed cause
        label, so the counts do not fragment on the measured value that happened to trip a check.
        """
        out: dict[str, int] = {}
        for e in self.events:
            out[e.code] = out.get(e.code, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
