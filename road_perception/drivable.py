"""Predicted segmentation -> a corridor the EXISTING planner can consume.

The planner never sees a mask. It consumes a `RoadModel`: a reference polyline with a left and a
right boundary. This module builds one from a predicted drivable mask, deterministically, in four
steps that are each easy to inspect and to disable:

    predicted 7-class mask
        -> binary drivable mask (class 0)
        -> denoise: keep the largest connected component, close small holes
        -> per-row left/right extent of the drivable region, in image space
        -> project to a ground plane and emit reference + boundaries

WHAT IS HONEST HERE
    The image-to-ground step needs a camera pose and intrinsics. IDD-Lite ships neither, so this
    module takes them as explicit arguments with documented defaults rather than pretending to
    recover them. The resulting corridor is metrically plausible, not metrically calibrated: the
    SHAPE of the drivable region is learned, the SCALE comes from the assumed geometry. That
    limitation is the reason Mode B in the comparison is reported as a perception demonstration
    rather than as a drop-in replacement for the simulator's corridor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .dataset import DRIVABLE


@dataclass
class GroundProjection:
    """Flat-ground pinhole projection from image row/column to vehicle-frame metres.

    Defaults describe a forward-facing camera roughly where a dashcam sits. They are ASSUMPTIONS,
    not measurements: IDD-Lite carries no calibration.
    """
    height_m: float = 1.4           # camera height above the road
    hfov_deg: float = 60.0          # horizontal field of view
    vfov_deg: float = 42.0          # vertical field of view
    pitch_deg: float = 0.0          # downward tilt, positive looks down
    max_range_m: float = 40.0       # beyond this the flat-ground assumption is worthless

    def row_to_range(self, row: float, height_px: int) -> float:
        """Ground distance for an image row, under a flat-road assumption.

        A row below the horizon subtends a depression angle; distance is height / tan(angle).
        Rows at or above the horizon return `max_range_m`, since a flat road never meets them.
        """
        v = (row + 0.5) / height_px                       # 0 at the top, 1 at the bottom
        half = np.radians(self.vfov_deg) / 2.0
        angle = np.radians(self.pitch_deg) + (v - 0.5) * 2.0 * half
        if angle <= 1e-3:
            return self.max_range_m
        return float(min(self.height_m / np.tan(angle), self.max_range_m))

    def col_to_lateral(self, col: float, width_px: int, range_m: float) -> float:
        """Lateral offset in metres at a given ground distance. Left of centre is positive."""
        u = (col + 0.5) / width_px - 0.5
        return float(-2.0 * u * range_m * np.tan(np.radians(self.hfov_deg) / 2.0))


def drivable_mask(pred: np.ndarray) -> np.ndarray:
    """Binary drivable map from a predicted class map."""
    return pred == DRIVABLE


def clean_mask(mask: np.ndarray, min_area_frac: float = 0.02,
               close_iterations: int = 2) -> np.ndarray:
    """Largest connected drivable region, with small holes closed.

    Justification for each operation, since post-processing can otherwise hide a bad model:
      * keeping the largest component removes speckle and detached patches of road that the
        vehicle cannot reach anyway without crossing non-drivable ground
      * binary closing fills holes left by objects ON the road (a car, a pedestrian). Those are
        obstacles for the PLANNER to avoid, not evidence that the road is missing there; the
        planner already receives them as tracked objects
    Both are deterministic and neither invents drivable area outside the predicted region's hull.
    """
    if not mask.any():
        return mask
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        if sizes.max() >= min_area_frac * mask.size:
            mask = labels == keep
        else:
            return np.zeros_like(mask)
    return ndimage.binary_closing(mask, structure=np.ones((3, 3)),
                                  iterations=close_iterations)


@dataclass
class Corridor:
    """A drivable corridor in the vehicle frame, ready to become a `RoadModel`.

    reference / left / right are (N, 2) arrays of (x forward, y left) in metres.
    `valid_rows` is how many image rows contributed, a direct confidence signal.
    """
    reference: np.ndarray
    left: np.ndarray
    right: np.ndarray
    valid_rows: int

    def as_road_dict(self) -> dict:
        """The same shape `DrivableSpace` is built from elsewhere in the project."""
        return {"reference": self.reference.tolist(),
                "left_boundary": self.left.tolist(),
                "right_boundary": self.right.tolist(),
                "lane_markings": "NONE"}


def extract_corridor(mask: np.ndarray, projection: GroundProjection | None = None,
                     row_step: int = 4, min_run_px: int = 8) -> Corridor:
    """Left and right edges of the drivable region, row by row, projected to the ground.

    Scans from the bottom of the image upward. For each sampled row it takes the widest
    horizontal run of drivable pixels that contains the column nearest the vehicle's own lateral
    position, so a side road does not capture the corridor. Once the corridor has started, the
    first row with no adequate run ends it: the estimate stops where the evidence stops rather
    than extrapolating.

    Rows BELOW the first evidence are skipped, not treated as the end. The bottom rows of a
    dashcam frame are frequently bonnet, ignore-label or unannotated padding -- IDD-Lite's masks
    stop two rows short of the bottom -- and treating an empty bottom row as the end of the road
    yields an empty corridor for every single image. The skip is naturally bounded because the
    scan stops once the flat-ground range exceeds `max_range_m`.
    """
    proj = projection or GroundProjection()
    h, w = mask.shape
    ref, left, right = [], [], []
    anchor = w // 2
    started = False

    for row in range(h - 1, -1, -row_step):
        rng = proj.row_to_range(row, h)
        if rng >= proj.max_range_m:
            break

        cols = np.flatnonzero(mask[row])
        run = None
        if cols.size >= min_run_px:
            # split into contiguous runs, choose the one nearest the anchor column
            breaks = np.flatnonzero(np.diff(cols) > 1)
            runs = [r for r in np.split(cols, breaks + 1) if r.size >= min_run_px]
            if runs:
                run = min(runs, key=lambda r: abs((r[0] + r[-1]) / 2.0 - anchor))
        if run is None:
            if started:
                break
            continue
        started = True
        centre = (run[0] + run[-1]) / 2.0
        ref.append([rng, proj.col_to_lateral(centre, w, rng)])
        left.append([rng, proj.col_to_lateral(run[0], w, rng)])       # small col = left in image
        right.append([rng, proj.col_to_lateral(run[-1], w, rng)])
        anchor = centre                                               # follow the road as it bends

    if len(ref) < 2:
        empty = np.zeros((0, 2))
        return Corridor(empty, empty, empty, 0)
    return Corridor(np.asarray(ref), np.asarray(left), np.asarray(right), len(ref))


def corridor_width_profile(c: Corridor) -> np.ndarray:
    """Corridor width in metres at each sampled range. Empty when the corridor is empty."""
    if c.valid_rows == 0:
        return np.zeros(0)
    return np.abs(c.left[:, 1] - c.right[:, 1])
