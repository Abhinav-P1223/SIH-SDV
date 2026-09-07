"""Image-space boxes -> world-frame `Detection`s, with a covariance that admits what it does not know.

THE HARD PART OF THIS PHASE IS RANGE, NOT CLASSIFICATION
    A monocular box gives BEARING almost for free: the horizontal position of the box centre maps
    through the camera intrinsics to an angle, and that angle is good to a fraction of a degree.

    Range is the opposite. A single image contains no depth. Two estimators are available and both
    are assumptions rather than measurements:

      * GROUND CONTACT. Assume the bottom edge of the box is where the object meets a flat road.
        The depression angle of that image row then gives range as height / tan(angle). Accurate
        when the assumption holds, and badly wrong when it does not: an occluded object whose feet
        are hidden reads as nearer than it is, and a road that slopes breaks it outright.
      * SIZE PRIOR. Assume the object is the typical size for its class and invert the projection.
        Robust to occlusion of the base, but it inherits the spread of real object sizes, and a
        misclassification becomes a range error.

    Both are used. The two estimates are combined, and crucially the DISAGREEMENT between them is
    folded into the reported range variance, so a box the two methods argue about arrives at the
    tracker already marked as uncertain. That matters more than the estimate itself: the Kalman
    filter is more sensitive to a confidently wrong covariance than to a noisy measurement.

    The resulting covariance is deliberately range-dominated and large, which is exactly the shape
    the simulated camera already produces. Nothing downstream needed changing to accept it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autonomy.core.types import Detection, ObjectType, SensorType

# Typical real-world height in metres for each type we can detect, used by the size-prior estimator.
# These are physical facts about the objects, not tuned parameters.
TYPICAL_HEIGHT_M: dict[ObjectType, float] = {
    ObjectType.PEDESTRIAN: 1.70,
    ObjectType.BICYCLE: 1.10,
    ObjectType.MOTORCYCLE: 1.30,
    ObjectType.CAR: 1.50,
    ObjectType.BUS: 3.20,
    ObjectType.TRUCK: 3.00,
    ObjectType.CATTLE: 1.40,
    ObjectType.AUTO_RICKSHAW: 1.80,
    ObjectType.PUSHCART: 1.20,
    ObjectType.UNKNOWN: 1.60,
}
# Coefficient of variation of real object heights within a class. A car is 1.5 m give or take 15%;
# a truck varies far more. This is what makes the size-prior estimate uncertain.
HEIGHT_SPREAD: dict[ObjectType, float] = {
    ObjectType.PEDESTRIAN: 0.09,
    ObjectType.BICYCLE: 0.15,
    ObjectType.MOTORCYCLE: 0.15,
    ObjectType.CAR: 0.15,
    ObjectType.BUS: 0.25,
    ObjectType.TRUCK: 0.30,
    ObjectType.CATTLE: 0.20,
}


@dataclass(frozen=True)
class CameraGeometry:
    """Intrinsics and mounting pose needed to turn a pixel into a bearing and a range.

    `fx`, `fy`, `cx`, `cy` are real intrinsics when the dataset supplies them (nuScenes does) and
    documented assumptions when it does not (IDD-Lite ships none, exactly as in Phase 3B).
    """
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = 1.4          # camera height above the road
    pitch_deg: float = 0.0         # downward tilt, positive looks down
    max_range_m: float = 80.0
    min_range_m: float = 1.0

    @staticmethod
    def from_intrinsic(K, **kw) -> "CameraGeometry":
        K = np.asarray(K, dtype=float)
        return CameraGeometry(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), **kw)

    @staticmethod
    def from_fov(width_px: int, height_px: int, hfov_deg: float = 60.0, **kw) -> "CameraGeometry":
        """For a camera with no published intrinsics. The field of view is then an ASSUMPTION."""
        fx = 0.5 * width_px / math.tan(math.radians(hfov_deg) / 2.0)
        return CameraGeometry(fx, fx, 0.5 * width_px, 0.5 * height_px, **kw)

    # ------------------------------------------------------------------ #
    def bearing_of_column(self, u: float) -> float:
        """Angle left of the optical axis for an image column. Positive is left, matching y-left."""
        return -math.atan2(u - self.cx, self.fx)

    def range_from_ground_contact(self, v: float) -> float | None:
        """Range implied by the image row where the object meets a flat road."""
        angle = math.atan2(v - self.cy, self.fy) + math.radians(self.pitch_deg)
        if angle <= 1e-4:
            return None                      # at or above the horizon: a flat road never meets it
        return self.height_m / math.tan(angle)

    def range_from_height(self, box_h_px: float, otype: ObjectType) -> float | None:
        """Range implied by how tall the object appears against its typical real height."""
        if box_h_px <= 1e-6:
            return None
        return TYPICAL_HEIGHT_M.get(otype, TYPICAL_HEIGHT_M[ObjectType.UNKNOWN]) * self.fy / box_h_px


def _fuse_range(r_ground: float | None, r_size: float | None, otype: ObjectType,
                geom: CameraGeometry) -> tuple[float, float] | None:
    """Combine the two range estimates and report a std-dev that includes their disagreement.

    When both estimators are available the mean is their inverse-variance weighted combination and
    the variance carries an extra term proportional to how far apart they were. Two methods that
    agree produce a tighter estimate; two that argue produce a wide one, which is the behaviour a
    filter downstream actually needs.
    """
    spread = HEIGHT_SPREAD.get(otype, 0.25)
    cands: list[tuple[float, float]] = []
    if r_ground is not None and r_ground > 0:
        # Ground contact degrades with range: one pixel of row error is worth more metres far away.
        cands.append((r_ground, max(0.10 * r_ground, 0.5)))
    if r_size is not None and r_size > 0:
        cands.append((r_size, max(spread * r_size, 0.5)))
    if not cands:
        return None
    w = np.array([1.0 / s ** 2 for _, s in cands])
    mu = float(np.sum(w * np.array([r for r, _ in cands])) / np.sum(w))
    var = float(1.0 / np.sum(w))
    if len(cands) == 2:
        var += 0.25 * (cands[0][0] - cands[1][0]) ** 2      # disagreement is real uncertainty
    r = min(max(mu, geom.min_range_m), geom.max_range_m)
    return r, math.sqrt(var)


def boxes_to_detections(dets, geom: CameraGeometry, sensor_x: float, sensor_y: float,
                        sensor_yaw: float, timestamp: float,
                        bearing_std_deg: float = 0.5) -> list[Detection]:
    """`RawDetection`s -> world-frame `Detection`s the existing tracker already accepts.

    `sensor_x/y/yaw` place the camera in the world. The emitted objects are identical in type and
    field usage to what `simulation.sensors.models.CameraModel` produces, which is why no change
    to the tracker, fusion or anything downstream was required.
    """
    out: list[Detection] = []
    sb = math.radians(bearing_std_deg)
    for d in dets:
        u, v = d.bottom_centre
        fused = _fuse_range(geom.range_from_ground_contact(v),
                            geom.range_from_height(d.height, d.object_type),
                            d.object_type, geom)
        if fused is None:
            continue
        rng, sr = fused
        bearing = sensor_yaw + geom.bearing_of_column(u)
        x = sensor_x + rng * math.cos(bearing)
        y = sensor_y + rng * math.sin(bearing)
        c, s = math.cos(bearing), math.sin(bearing)
        R = np.array([[c, -s], [s, c]])
        cov = R @ np.diag([sr ** 2, (rng * sb) ** 2]) @ R.T
        out.append(Detection(
            SensorType.CAMERA, timestamp, x, y, cov,
            object_type=d.object_type, class_confidence=d.score,
            sensor_x=sensor_x, sensor_y=sensor_y))
    return out
