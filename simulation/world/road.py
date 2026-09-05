"""Drivable-space corridor for unstructured roads.

A `DrivableSpace` is defined by three polylines:
    reference       : the direction of travel toward the goal (NOT a lane line)
    left_boundary   : left road edge (positive lateral offset side)
    right_boundary  : right road edge

The corridor polygon is left_boundary + reversed(right_boundary). Width may
vary along s and the corridor need not be symmetric about the reference.
Lane markings are an optional annotation (`lane_markings`) that nothing in
the planner depends on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autonomy.core.geometry import (points_in_polygon, points_to_polyline_distance,
                                    polyline_arclength)
from autonomy.core.interfaces import RoadModel


@dataclass
class DrivableSpace(RoadModel):
    reference: np.ndarray            # (M,2)
    left_boundary: np.ndarray        # (K,2)
    right_boundary: np.ndarray       # (J,2)
    lane_markings: str = "NONE"      # informational only
    resample_step_m: float = 1.0
    _s: np.ndarray = field(init=False, repr=False)
    _heading: np.ndarray = field(init=False, repr=False)
    _polygon: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reference = self._resample(np.asarray(self.reference, dtype=float))
        self.left_boundary = np.asarray(self.left_boundary, dtype=float)
        self.right_boundary = np.asarray(self.right_boundary, dtype=float)
        self._s = polyline_arclength(self.reference)
        d = np.diff(self.reference, axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        self._heading = np.concatenate([h, h[-1:]])
        self._polygon = np.vstack([self.left_boundary, self.right_boundary[::-1]])

    def _resample(self, poly: np.ndarray) -> np.ndarray:
        if poly.shape[0] < 2:
            raise ValueError("reference polyline needs at least two points")
        s = polyline_arclength(poly)
        n = max(2, int(math.ceil(s[-1] / self.resample_step_m)) + 1)
        ss = np.linspace(0.0, s[-1], n)
        return np.stack([np.interp(ss, s, poly[:, 0]), np.interp(ss, s, poly[:, 1])], axis=1)

    # -- RoadModel -------------------------------------------------------- #
    @property
    def length(self) -> float:
        return float(self._s[-1])

    def heading_at(self, s: np.ndarray | float) -> np.ndarray:
        idx = np.clip(np.searchsorted(self._s, s, side="right") - 1, 0, len(self._s) - 2)
        return self._heading[idx]

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        p = np.array([x, y])
        a = self.reference[:-1]
        b = self.reference[1:]
        ab = b - a
        len2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
        t = np.clip(np.sum((p - a) * ab, axis=1) / len2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist2 = np.sum((p - proj) ** 2, axis=1)
        i = int(np.argmin(dist2))
        seg_len = math.sqrt(len2[i])
        s = float(self._s[i] + t[i] * seg_len)
        heading = float(self._heading[i])
        # signed lateral offset: positive to the left of the reference direction
        dx, dy = p - proj[i]
        d = float(-math.sin(heading) * dx + math.cos(heading) * dy)
        return s, d, heading

    def to_cartesian(self, s: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        d = np.asarray(d, dtype=float)
        s_c = np.clip(s, 0.0, self.length)
        rx = np.interp(s_c, self._s, self.reference[:, 0])
        ry = np.interp(s_c, self._s, self.reference[:, 1])
        h = self.heading_at(s_c)
        # extrapolate beyond the end along the final heading
        over = s - s_c
        rx = rx + over * np.cos(h)
        ry = ry + over * np.sin(h)
        x = rx - d * np.sin(h)
        y = ry + d * np.cos(h)
        return x, y, h

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        return points_in_polygon(points, self._polygon)

    def boundary_clearance(self, corners: np.ndarray) -> np.ndarray:
        corners = np.asarray(corners, dtype=float)
        n = corners.shape[0]
        flat = corners.reshape(-1, 2)
        d_left = points_to_polyline_distance(flat, self.left_boundary)
        d_right = points_to_polyline_distance(flat, self.right_boundary)
        d = np.minimum(d_left, d_right).reshape(n, 4).min(axis=1)
        inside = self.contains_points(flat).reshape(n, 4).all(axis=1)
        return np.where(inside, d, 0.0)

    def footprint_inside(self, corners: np.ndarray, margin: float) -> np.ndarray:
        corners = np.asarray(corners, dtype=float)
        n = corners.shape[0]
        inside = self.contains_points(corners.reshape(-1, 2)).reshape(n, 4).all(axis=1)
        if margin <= 0:
            return inside
        return inside & (self.boundary_clearance(corners) >= margin)

    def lateral_bounds_at(self, s: float) -> tuple[float, float]:
        """(d_right, d_left) offsets of the boundaries at arc length s (right < 0 < left)."""
        x, y, h = self.to_cartesian(np.array([s]), np.array([0.0]))
        p = np.array([x[0], y[0]])
        normal = np.array([-math.sin(float(h[0])), math.cos(float(h[0]))])
        d_left = self._ray_hit(p, normal, self.left_boundary)
        d_right = -self._ray_hit(p, -normal, self.right_boundary)
        return d_right, d_left

    @staticmethod
    def _ray_hit(p: np.ndarray, n: np.ndarray, poly: np.ndarray) -> float:
        """Distance along ray p + t n to the nearest polyline crossing (inf if none)."""
        best = math.inf
        for a, b in zip(poly[:-1], poly[1:]):
            ab = b - a
            denom = n[0] * ab[1] - n[1] * ab[0]
            if abs(denom) < 1e-12:
                continue
            ap = a - p
            t = (ap[0] * ab[1] - ap[1] * ab[0]) / denom
            u = (ap[0] * n[1] - ap[1] * n[0]) / denom
            if t >= 0 and 0.0 <= u <= 1.0:
                best = min(best, t)
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.tolist(),
            "left_boundary": self.left_boundary.tolist(),
            "right_boundary": self.right_boundary.tolist(),
            "lane_markings": self.lane_markings,
            "length": self.length,
        }

    @staticmethod
    def straight(length: float, half_width_left: float, half_width_right: float,
                 x0: float = 0.0, y0: float = 0.0, heading: float = 0.0) -> "DrivableSpace":
        c, s = math.cos(heading), math.sin(heading)
        ref = np.array([[x0, y0], [x0 + length * c, y0 + length * s]])
        nl = np.array([-s, c])
        return DrivableSpace(ref, ref + half_width_left * nl, ref - half_width_right * nl)
