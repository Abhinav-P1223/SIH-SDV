"""Planar geometry for footprints and drivable space.

Everything here is written with plain numpy so it ports to MATLAB without a
geometry library. The hot path for the planner is `box_sequence_distance`,
which compares N ego footprints against N object footprints in one call.

Conventions: right-handed world frame, yaw counter-clockwise from +x, radians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrientedBox:
    cx: float
    cy: float
    yaw: float
    length: float   # along heading
    width: float    # across heading

    def corners(self) -> np.ndarray:
        """(4, 2) corners counter-clockwise starting front-left."""
        return box_corners(
            np.array([self.cx]), np.array([self.cy]), np.array([self.yaw]),
            self.length, self.width,
        )[0]

    def inflate(self, margin: float) -> "OrientedBox":
        return OrientedBox(self.cx, self.cy, self.yaw,
                           self.length + 2 * margin, self.width + 2 * margin)

    @property
    def half_diagonal(self) -> float:
        return 0.5 * math.hypot(self.length, self.width)


# --------------------------------------------------------------------------- #
# Vectorised footprint helpers
# --------------------------------------------------------------------------- #
def box_corners(cx: np.ndarray, cy: np.ndarray, yaw: np.ndarray,
                length: float | np.ndarray, width: float | np.ndarray) -> np.ndarray:
    """Corners of N oriented boxes -> (N, 4, 2), counter-clockwise."""
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)
    yaw = np.asarray(yaw, dtype=float)
    hl = 0.5 * np.asarray(length, dtype=float)
    hw = 0.5 * np.asarray(width, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    # local corners: front-left, rear-left, rear-right, front-right
    lx = np.stack([hl, -hl, -hl, hl], axis=-1) * np.ones_like(cx)[..., None]
    ly = np.stack([hw, hw, -hw, -hw], axis=-1) * np.ones_like(cx)[..., None]
    wx = cx[..., None] + lx * c[..., None] - ly * s[..., None]
    wy = cy[..., None] + lx * s[..., None] + ly * c[..., None]
    return np.stack([wx, wy], axis=-1)


def _project_interval(corners: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """corners (N,4,2), axis (N,2) -> (min (N,), max (N,)) projections."""
    p = np.einsum("nkj,nj->nk", corners, axis)
    return p.min(axis=1), p.max(axis=1)


def boxes_overlap(corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
    """Separating-axis test for N pairs of convex quads -> bool (N,)."""
    n = corners_a.shape[0]
    overlap = np.ones(n, dtype=bool)
    for corners in (corners_a, corners_b):
        for k in range(2):  # two unique edge directions per rectangle
            edge = corners[:, k + 1, :] - corners[:, k, :]
            axis = np.stack([-edge[:, 1], edge[:, 0]], axis=1)
            norm = np.linalg.norm(axis, axis=1, keepdims=True)
            axis = axis / np.where(norm > 0, norm, 1.0)
            amin, amax = _project_interval(corners_a, axis)
            bmin, bmax = _project_interval(corners_b, axis)
            overlap &= (amax >= bmin) & (bmax >= amin)
    return overlap


def points_to_segments_distance(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Distance from points (N,P,2) to segments (N,S,2)->(N,S,2) : result (N,P,S)."""
    d = seg_b - seg_a                                  # (N,S,2)
    len2 = np.sum(d * d, axis=-1)                      # (N,S)
    len2 = np.where(len2 > 0, len2, 1.0)
    ap = points[:, :, None, :] - seg_a[:, None, :, :]  # (N,P,S,2)
    t = np.sum(ap * d[:, None, :, :], axis=-1) / len2[:, None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = seg_a[:, None, :, :] + t[..., None] * d[:, None, :, :]
    return np.linalg.norm(points[:, :, None, :] - proj, axis=-1)


def polygon_distance(corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
    """Minimum distance between N pairs of convex quads (0 if overlapping) -> (N,)."""
    ea = np.roll(corners_a, -1, axis=1)
    eb = np.roll(corners_b, -1, axis=1)
    d_ab = points_to_segments_distance(corners_a, corners_b, eb).min(axis=(1, 2))
    d_ba = points_to_segments_distance(corners_b, corners_a, ea).min(axis=(1, 2))
    dist = np.minimum(d_ab, d_ba)
    dist[boxes_overlap(corners_a, corners_b)] = 0.0
    return dist


def box_sequence_distance(ax: np.ndarray, ay: np.ndarray, ayaw: np.ndarray, alen: float, awid: float,
                          bx: np.ndarray, by: np.ndarray, byaw: np.ndarray, blen: float, bwid: float,
                          exact_within: float = 3.0, a_corners: np.ndarray | None = None) -> np.ndarray:
    """Distance between paired boxes A_i and B_i for i in range(N) -> (N,).

    Exact (0 when overlapping) whenever the pair could be within `exact_within`
    metres of contact, i.e. centre distance <= r_a + r_b + exact_within with r the
    half-diagonals. Farther pairs return the conservative lower bound
    centre_distance - r_a - r_b (never above the true distance), which is all the
    margin tests and the saturating clearance cost need. Fully vectorised.
    """
    ax, ay, ayaw = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (ax, ay, ayaw))
    bx, by, byaw = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (bx, by, byaw))
    ra = 0.5 * math.hypot(alen, awid)
    rb = 0.5 * math.hypot(blen, bwid)
    centre = np.hypot(ax - bx, ay - by)
    out = centre - ra - rb
    near = out <= exact_within
    if np.any(near):
        # `a_corners` lets a caller that compares ONE sequence of A-boxes against several different
        # B-objects build the A corners once instead of once per object.
        ca = box_corners(ax[near], ay[near], ayaw[near], alen, awid) if a_corners is None             else a_corners[near]
        cb = box_corners(bx[near], by[near], byaw[near], blen, bwid)
        out[near] = polygon_distance(ca, cb)
    return out


def box_distance(a: OrientedBox, b: OrientedBox) -> float:
    """Scalar convenience wrapper (always exact)."""
    return float(box_sequence_distance(
        np.array([a.cx]), np.array([a.cy]), np.array([a.yaw]), a.length, a.width,
        np.array([b.cx]), np.array([b.cy]), np.array([b.yaw]), b.length, b.width, exact_within=math.inf,
    )[0])


def box_overlap(a: OrientedBox, b: OrientedBox) -> bool:
    return bool(boxes_overlap(a.corners()[None], b.corners()[None])[0])


# --------------------------------------------------------------------------- #
# Polylines and polygons (drivable space)
# --------------------------------------------------------------------------- #
def points_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point (P,2) to a polyline (M,2) -> (P,)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    seg_a = polyline[:-1][None]
    seg_b = polyline[1:][None]
    d = points_to_segments_distance(points[None], seg_a, seg_b)  # (1,P,S)
    return d[0].min(axis=1)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for points (P,2) against polygon (M,2) -> bool (P,)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    px, py = polygon[:, 0], polygon[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    inside = np.zeros(points.shape[0], dtype=bool)
    for i in range(polygon.shape[0]):
        cond = (py[i] > y) != (qy[i] > y)
        denom = (qy[i] - py[i])
        denom = denom if denom != 0 else 1e-12
        xint = (qx[i] - px[i]) * (y - py[i]) / denom + px[i]
        inside ^= cond & (x < xint)
    return inside


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi if isinstance(a, np.ndarray) \
        else (a + math.pi) % (2 * math.pi) - math.pi


def polyline_arclength(polyline: np.ndarray) -> np.ndarray:
    seg = np.hypot(np.diff(polyline[:, 0]), np.diff(polyline[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])
