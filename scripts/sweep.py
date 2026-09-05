"""Parameter sweep over a scenario: measures scenario success rate honestly.

    python scripts/sweep.py                      # default grid on SUDDEN_CATTLE_CROSSING
    python scripts/sweep.py --scenario SUDDEN_PEDESTRIAN_DART --param trigger_distance_m --values 14 18 22 26

Each cell is a full closed-loop run with the committed configuration. The
result table (collision-free, completed, min clearance, min TTC, EB count) is
printed and returned; `tests/scenarios/test_parameter_sweep.py` asserts on a
small default grid, and `scripts/generate_results.py` embeds the table.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autonomy.core.config import AutonomyConfig, ObjectProfiles, load_vehicle_parameters, load_yaml  # noqa: E402
from autonomy.telemetry.telemetry import TelemetryPublisher  # noqa: E402
from simulation.runner import Simulation  # noqa: E402
from simulation.scenarios.loader import SCENARIO_DIR, build_scenario  # noqa: E402


@dataclass
class SweepCell:
    overrides: dict
    completed: bool
    collisions: int
    min_clearance: float
    min_ttc: float
    eb_activations: int
    time_to_completion: float | None
    planner_mean_ms: float

    @property
    def success(self) -> bool:
        return self.completed and self.collisions == 0


DEFAULT_GRID = {
    "SUDDEN_CATTLE_CROSSING": {"crossing_speed_mps": [0.6, 1.0, 1.4], "trigger_distance_m": [25.0, 35.0]},
    "SUDDEN_PEDESTRIAN_DART": {"trigger_distance_m": [18.0, 22.0, 26.0]},
}


def run_cell(name: str, overrides: dict, cfg: AutonomyConfig, params, profiles) -> SweepCell:
    data = load_yaml(SCENARIO_DIR / f"{name.lower()}.yaml")
    for a in data["agents"]:
        a["params"].update(overrides)
    sc = build_scenario(data, profiles)
    sim = Simulation(sc, cfg, params, profiles, TelemetryPublisher([]))
    m = sim.run()
    return SweepCell(dict(overrides), m.scenario_completed, m.collision_count, m.minimum_obstacle_clearance,
                     m.minimum_ttc, m.emergency_brake_activations, m.time_to_completion, m.planning_latency_mean_ms)


def sweep(name: str, grid: dict[str, list]) -> list[SweepCell]:
    cfg = AutonomyConfig.load()
    params = load_vehicle_parameters()
    profiles = ObjectProfiles.load()
    keys = list(grid)
    cells = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cells.append(run_cell(name, dict(zip(keys, combo)), cfg, params, profiles))
    return cells


def table(cells: list[SweepCell]) -> str:
    keys = list(cells[0].overrides) if cells else []
    head = "| " + " | ".join(keys) + " | completed | collisions | min clearance [m] | min TTC [s] | EB | t_done [s] | planner [ms] |"
    sep = "|" + "---|" * (len(keys) + 7)
    rows = [head, sep]
    for c in cells:
        ttc = "inf" if math.isinf(c.min_ttc) else f"{c.min_ttc:.2f}"
        clr = "inf" if math.isinf(c.min_clearance) else f"{c.min_clearance:.2f}"
        done = "n/a" if c.time_to_completion is None else f"{c.time_to_completion:.1f}"
        rows.append("| " + " | ".join(str(c.overrides[k]) for k in keys)
                    + f" | {c.completed} | {c.collisions} | {clr} | {ttc} | {c.eb_activations} | {done} | {c.planner_mean_ms:.0f} |")
    n_ok = sum(c.success for c in cells)
    rows.append(f"\nScenario success rate: {n_ok}/{len(cells)} = {n_ok / len(cells):.2f}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="SUDDEN_CATTLE_CROSSING")
    ap.add_argument("--param", default=None)
    ap.add_argument("--values", nargs="*", type=float, default=None)
    args = ap.parse_args()
    grid = {args.param: args.values} if args.param else DEFAULT_GRID[args.scenario]
    cells = sweep(args.scenario, grid)
    print(f"## {args.scenario} sweep\n")
    print(table(cells))
    return 0 if all(c.success for c in cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
