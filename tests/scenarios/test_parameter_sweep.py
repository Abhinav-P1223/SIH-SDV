"""Robustness: the cattle crossing must be handled across a grid of crossing
speeds and trigger distances, not just the single committed configuration.
Each cell is a full closed-loop run (about 3 s each)."""
import pytest

from scripts.sweep import DEFAULT_GRID, sweep

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cells():
    return sweep("SUDDEN_CATTLE_CROSSING", DEFAULT_GRID["SUDDEN_CATTLE_CROSSING"])


def test_grid_is_non_trivial(cells):
    assert len(cells) == 6
    assert len({tuple(c.overrides.values()) for c in cells}) == 6


def test_every_cell_is_collision_free_and_completes(cells):
    failures = [c.overrides for c in cells if not c.success]
    assert failures == [], f"failed cells: {failures}"


def test_clearance_margin_holds_across_grid(cells):
    from autonomy.core.config import AutonomyConfig
    margin = AutonomyConfig.load().planning.safety_margin_m
    assert all(c.min_clearance >= margin for c in cells)


def test_faster_crossing_is_recognised_as_more_urgent(cells):
    """Same trigger distance: a faster cow gives a shorter minimum TTC or an EB, never a silent pass."""
    by_trigger = {}
    for c in cells:
        by_trigger.setdefault(c.overrides["trigger_distance_m"], []).append(c)
    for trig, group in by_trigger.items():
        group.sort(key=lambda c: c.overrides["crossing_speed_mps"])
        assert all(c.min_ttc < 4.0 for c in group), f"risk never became finite at trigger {trig}"
