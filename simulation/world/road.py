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

from autonomy.core.geometry import (points_in_polygon, points_to_segments_distance,
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
    _d_left: np.ndarray = field(init=False, repr=False)
    _d_right: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reference = self._resample(np.asarray(self.reference, dtype=float))
        self.left_boundary = np.asarray(self.left_boundary, dtype=float)
        self.right_boundary = np.asarray(self.right_boundary, dtype=float)
        self._s = polyline_arclength(self.reference)
        d = np.diff(self.reference, axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        self._heading = np.concatenate([h, h[-1:]])
        self._polygon = np.vstack([self.left_boundary, self.right_boundary[::-1]])
        # corridor edges at every reference station, for the vectorised lateral_bounds()
        dl, dr = [], []
        for i in range(len(self._s)):
            p = self.reference[i]
            hh = float(self._heading[i])
            normal = np.array([-math.sin(hh), math.cos(hh)])
            dl.append(self._ray_hit(p, normal, self.left_boundary))
            dr.append(-self._ray_hit(p, -normal, self.right_boundary))
        self._d_left = np.array(dl)
        self._d_right = np.array(dr)
        # stations without a hit (open ends) inherit their neighbour
        for arr in (self._d_left, self._d_right):
            bad = ~np.isfinite(arr)
            if bad.any() and (~bad).any():
                arr[bad] = np.interp(self._s[bad], self._s[~bad], arr[~bad])

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

    # -- boundary queries --------------------------------------------------- #
    # Both boundaries are polylines running in the direction of travel. For a
    # batch of query points only the boundary segments near the points' bounding
    # box are considered (window), which keeps the cost proportional to the
    # planning horizon rather than the road length. A point is inside the
    # corridor when it lies to the RIGHT of its nearest left-boundary segment and
    # to the LEFT of its nearest right-boundary segment (cross-product sign).
    _WINDOW_PAD_M = 12.0

    @staticmethod
    def _windowed_segments(points: np.ndarray, poly: np.ndarray, pad: float) -> tuple[np.ndarray, np.ndarray]:
        a, b = poly[:-1], poly[1:]
        lo = points.min(axis=0) - pad
        hi = points.max(axis=0) + pad
        seg_lo = np.minimum(a, b)
        seg_hi = np.maximum(a, b)
        keep = np.all(seg_hi >= lo, axis=1) & np.all(seg_lo <= hi, axis=1)
        if not np.any(keep):
            return a, b
        return a[keep], b[keep]

    def _nearest_side_and_distance(self, points: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(signed side, distance) of points (P,2) to polyline; side>0 means left of the polyline direction."""
        a, b = self._windowed_segments(points, poly, self._WINDOW_PAD_M)
        d = points_to_segments_distance(points[None], a[None], b[None])[0]      # (P,S)
        k = np.argmin(d, axis=1)
        dist = d[np.arange(points.shape[0]), k]
        seg_a, seg_b = a[k], b[k]
        e = seg_b - seg_a
        r = points - seg_a
        side = e[:, 0] * r[:, 1] - e[:, 1] * r[:, 0]
        return side, dist

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        side_l, _ = self._nearest_side_and_distance(points, self.left_boundary)
        side_r, _ = self._nearest_side_and_distance(points, self.right_boundary)
        return (side_l <= 0.0) & (side_r >= 0.0)

    def contains_points_polygon(self, points: np.ndarray) -> np.ndarray:
        """Reference implementation (even-odd ray casting over the full polygon)."""
        return points_in_polygon(points, self._polygon)

    def boundary_clearance(self, corners: np.ndarray) -> np.ndarray:
        corners = np.asarray(corners, dtype=float)
        n = corners.shape[0]
        flat = corners.reshape(-1, 2)
        side_l, d_left = self._nearest_side_and_distance(flat, self.left_boundary)
        side_r, d_right = self._nearest_side_and_distance(flat, self.right_boundary)
        inside = ((side_l <= 0.0) & (side_r >= 0.0)).reshape(n, 4).all(axis=1)
        d = np.minimum(d_left, d_right).reshape(n, 4).min(axis=1)
        return np.where(inside, d, 0.0)

    def footprint_inside(self, corners: np.ndarray, margin: float) -> np.ndarray:
        corners = np.asarray(corners, dtype=float)
        clearance = self.boundary_clearance(corners)
        if margin <= 0:
            return clearance > 0.0
        return clearance >= margin

    def lateral_bounds(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        return np.interp(s, self._s, self._d_right), np.interp(s, self._s, self._d_left)

    def lateral_bounds_at(self, s: float) -> tuple[float, float]:
        """(d_right, d_left) offsets of the boundaries at arc length s (right < 0 < left)."""
        dr, dl = self.lateral_bounds(np.array([s]))
        return float(dr[0]), float(dl[0])

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

    @staticmethod
    def from_segments(segments: list[dict], half_width_left: float, half_width_right: float,
                      x0: float = 0.0, y0: float = 0.0, heading: float = 0.0, step_m: float = 1.0,
                      lane_markings: str = "NONE") -> "DrivableSpace":
        """Corridor from a list of {straight: L} / {arc: {radius: R, angle_deg: A}} segments.

        Positive angle turns left. Boundaries are the reference offset along its
        local normal, so the corridor keeps a constant width through curves.
        """
        pts = [np.array([x0, y0])]
        h = heading
        for seg in segments:
            if "straight" in seg:
                L = float(seg["straight"])
                n = max(1, int(math.ceil(L / step_m)))
                base = pts[-1].copy()
                for i in range(1, n + 1):
                    d = L * i / n
                    pts.append(base + d * np.array([math.cos(h), math.sin(h)]))
            elif "arc" in seg:
                R = float(seg["arc"]["radius"])
                A = math.radians(float(seg["arc"]["angle_deg"]))
                n = max(2, int(math.ceil(abs(A) * R / step_m)))
                sign = 1.0 if A >= 0 else -1.0
                cx = pts[-1][0] - sign * R * math.sin(h)
                cy = pts[-1][1] + sign * R * math.cos(h)
                for i in range(1, n + 1):
                    hi = h + A * i / n
                    pts.append(np.array([cx + sign * R * math.sin(hi), cy - sign * R * math.cos(hi)]))
                h = h + A
            else:
                raise ValueError(f"unknown segment {seg}")
        ref = np.array(pts)
        d = np.gradient(ref, axis=0)
        hd = np.arctan2(d[:, 1], d[:, 0])
        normal = np.stack([-np.sin(hd), np.cos(hd)], axis=1)
        return DrivableSpace(ref, ref + half_width_left * normal, ref - half_width_right * normal,
                             lane_markings=lane_markings)
