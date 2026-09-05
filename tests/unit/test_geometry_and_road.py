import math

import numpy as np
import pytest

from autonomy.core.geometry import (OrientedBox, box_corners, box_distance, box_overlap,
                                    box_sequence_distance, points_in_polygon)
from simulation.world.road import DrivableSpace


# ----------------------------------------------------------------- geometry --
def test_box_corners_axis_aligned():
    c = box_corners(np.array([0.0]), np.array([0.0]), np.array([0.0]), 4.0, 2.0)[0]
    assert set(map(tuple, np.round(c, 6))) == {(2, 1), (-2, 1), (-2, -1), (2, -1)}


def test_box_distance_axis_aligned_gap():
    assert box_distance(OrientedBox(0, 0, 0, 4, 2), OrientedBox(5, 0, 0, 2, 1)) == pytest.approx(2.0)


def test_box_distance_rotated_neighbour():
    # box rotated 90 deg sits above: its half-width along y is length/2 = 2 -> gap 4-1-2 = 1
    assert box_distance(OrientedBox(0, 0, 0, 4, 2), OrientedBox(0, 4, math.pi / 2, 4, 2)) == pytest.approx(1.0)


def test_box_overlap_and_zero_distance():
    a, b = OrientedBox(0, 0, 0, 4, 2), OrientedBox(2.5, 0.5, 0.3, 2, 1)
    assert box_overlap(a, b)
    assert box_distance(a, b) == 0.0


def test_box_no_overlap_diagonal_corner_case():
    # touching only near a corner but separated
    a, b = OrientedBox(0, 0, 0, 2, 2), OrientedBox(2.2, 2.2, math.pi / 4, 2, 2)
    assert not box_overlap(a, b)
    assert box_distance(a, b) > 0.0


def test_box_sequence_distance_vectorised_matches_scalar():
    n = 25
    ax = np.linspace(0, 20, n); ay = np.zeros(n); ayaw = np.linspace(0, 0.3, n)
    bx = np.full(n, 10.0); by = np.linspace(-3, 3, n); byaw = np.full(n, 1.2)
    exact = box_sequence_distance(ax, ay, ayaw, 4.2, 1.8, bx, by, byaw, 2.0, 0.7, exact_within=math.inf)
    fast = box_sequence_distance(ax, ay, ayaw, 4.2, 1.8, bx, by, byaw, 2.0, 0.7)          # default shortcut
    for i in range(n):
        s = box_distance(OrientedBox(ax[i], ay[i], ayaw[i], 4.2, 1.8),
                         OrientedBox(bx[i], by[i], byaw[i], 2.0, 0.7))
        assert exact[i] == pytest.approx(s, abs=1e-9)
        assert fast[i] <= s + 1e-9                          # the shortcut never overestimates
        if s <= 3.0:
            assert fast[i] == pytest.approx(s, abs=1e-9)    # and is exact where it matters


def test_points_in_polygon():
    poly = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    inside = points_in_polygon(np.array([[5, 2.5], [11, 2.5], [5, -1], [0.1, 0.1]]), poly)
    assert inside.tolist() == [True, False, False, True]


# --------------------------------------------------------------------- road --
@pytest.fixture
def road():
    return DrivableSpace.straight(100.0, 3.5, 3.5)


def test_projection_roundtrip(road):
    s, d, h = road.project(40.0, 1.5)
    assert (s, d, h) == pytest.approx((40.0, 1.5, 0.0))
    x, y, _ = road.to_cartesian(np.array([40.0]), np.array([1.5]))
    assert (x[0], y[0]) == pytest.approx((40.0, 1.5))


def test_projection_sign_convention_left_positive():
    _, d, _ = road.project(10.0, 2.0) if False else DrivableSpace.straight(50, 3, 3).project(10.0, 2.0)
    assert d > 0


def test_rotated_corridor_projection():
    r = DrivableSpace.straight(50.0, 3.0, 3.0, heading=math.pi / 2)  # travelling +y
    s, d, h = r.project(-1.0, 20.0)  # left of +y travel is -x
    assert s == pytest.approx(20.0) and d == pytest.approx(1.0) and h == pytest.approx(math.pi / 2)


def test_footprint_inside_and_clearance(road):
    inside_corners = box_corners(np.array([50.0]), np.array([0.0]), np.array([0.0]), 4.2, 1.8)
    edge_corners = box_corners(np.array([50.0]), np.array([2.8]), np.array([0.0]), 4.2, 1.8)   # top at 3.7 > 3.5
    assert road.footprint_inside(inside_corners, 0.25).tolist() == [True]
    assert road.footprint_inside(edge_corners, 0.0).tolist() == [False]
    assert road.boundary_clearance(inside_corners)[0] == pytest.approx(3.5 - 0.9)
    assert road.boundary_clearance(edge_corners)[0] == 0.0


def test_footprint_margin_enforced(road):
    # top edge at 3.4 -> clearance 0.1 < margin 0.25
    corners = box_corners(np.array([50.0]), np.array([2.5]), np.array([0.0]), 4.2, 1.8)
    assert road.footprint_inside(corners, 0.0).tolist() == [True]
    assert road.footprint_inside(corners, 0.25).tolist() == [False]


def test_lateral_bounds(road):
    d_right, d_left = road.lateral_bounds_at(30.0)
    assert (d_right, d_left) == pytest.approx((-3.5, 3.5))


def test_extrapolation_beyond_end(road):
    x, y, _ = road.to_cartesian(np.array([120.0]), np.array([0.0]))
    assert x[0] == pytest.approx(120.0)


def test_segment_corridor_and_bounds():
    r = DrivableSpace.from_segments([{"straight": 30}, {"arc": {"radius": 40, "angle_deg": 60}}, {"straight": 20}], 3.0, 4.0)
    assert r.length == pytest.approx(30 + math.radians(60) * 40 + 20, abs=0.05)
    dr, dl = r.lateral_bounds(np.array([10.0, 50.0, 85.0]))
    assert np.allclose(dr, -4.0, atol=0.05) and np.allclose(dl, 3.0, atol=0.05)
    # projection round trip inside the curve
    x, y, _ = r.to_cartesian(np.array([55.0]), np.array([1.2]))
    s, d, _ = r.project(float(x[0]), float(y[0]))
    assert (s, d) == pytest.approx((55.0, 1.2), abs=0.05)


def test_windowed_inside_matches_polygon_reference():
    r = DrivableSpace.from_segments([{"straight": 30}, {"arc": {"radius": 40, "angle_deg": 60}}, {"straight": 20}], 3.5, 3.5)
    rng = np.random.default_rng(3)
    s = rng.uniform(2, r.length - 2, 400)
    d = rng.uniform(-5, 5, 400)
    x, y, _ = r.to_cartesian(s, d)
    pts = np.stack([x, y], axis=1)
    fast = r.contains_points(pts)
    ref = r.contains_points_polygon(pts)
    # disagreement only possible within a few cm of the edge
    edge = np.abs(np.abs(d) - 3.5) < 0.1
    assert np.array_equal(fast[~edge], ref[~edge])
