"""Jury audit: adaptivity. Perturb the world and show that the SELECTED trajectory
and the outcome change. If they did not, the behaviour would be scripted.

    python scripts/audit_adaptivity.py
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autonomy.core.config import AutonomyConfig, ObjectProfiles, PerceptionConfig, load_vehicle_parameters, load_yaml  # noqa: E402
from autonomy.telemetry.telemetry import InMemorySink, TelemetryPublisher  # noqa: E402
from simulation.runner import Simulation  # noqa: E402
from simulation.scenarios.loader import SCENARIO_DIR, build_scenario  # noqa: E402


PERCEPTION_MODE = "ground_truth"      # overridden by --perception


def run(data, cfg=None):
    cfg = cfg or AutonomyConfig.load()
    sc = build_scenario(data)
    mem = InMemorySink()
    perception = PerceptionConfig.load().with_mode(PERCEPTION_MODE)
    sim = Simulation(sc, cfg, load_vehicle_parameters(), ObjectProfiles.load(), TelemetryPublisher([mem]),
                     perception=perception)
    m = sim.run()
    return m, mem.frames


def selected_at_range(frames, obj_id, rng_m=25.0):
    """Selected candidate at the first planning cycle where the hazard is within rng_m ahead."""
    for f in frames:
        if not f.planning_cycle:
            continue
        for o in f.objects:
            if o.id == obj_id and 0 < o.x - f.ego.x <= rng_m:
                return f.plan.selected.id, f.decision.state.value, f.plan.feasible_count, f.plan.rejected_count
    return "n/a", "n/a", 0, 0


def summarise(name, m, frames, obj_id="cattle_1"):
    ys = [f.ego.y for f in frames]
    vs = [f.ego.longitudinal_velocity for f in frames]
    states = []
    for f in frames:
        if f.decision and (not states or states[-1] != f.decision.state.value):
            states.append(f.decision.state.value)
    sel, st, feas, rej = selected_at_range(frames, obj_id)
    ttc = "inf" if math.isinf(m.minimum_ttc) else f"{m.minimum_ttc:.2f}"
    clr = "inf" if math.isinf(m.minimum_obstacle_clearance) else f"{m.minimum_obstacle_clearance:.2f}"
    done = "n/a" if m.time_to_completion is None else f"{m.time_to_completion:.1f}"
    return (f"| {name} | {m.termination_reason} | {m.collision_count} | {clr} | {ttc} | {min(vs):.1f} | "
            f"{min(ys):.2f}..{max(ys):.2f} | {sel} ({st}, {feas}/{feas + rej}) | {' > '.join(states)} | {done} |")


def main() -> int:
    global PERCEPTION_MODE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", choices=["ground_truth", "sensors"], default="ground_truth")
    PERCEPTION_MODE = ap.parse_args().perception
    base = load_yaml(SCENARIO_DIR / "sudden_cattle_crossing.yaml")
    variants = []

    def variant(name, fn):
        d = copy.deepcopy(base)
        fn(d)
        variants.append((name, d))

    variant("baseline", lambda d: None)
    variant("cattle starts 0.6 m further in (y=2.5)", lambda d: d["agents"][0].__setitem__("y", 2.5))
    variant("cattle 2x faster (1.6 m/s)", lambda d: d["agents"][0]["params"].__setitem__("crossing_speed_mps", 1.6))
    variant("cattle crosses from the RIGHT edge", lambda d: (d["agents"][0].__setitem__("y", -3.1),
                                                              d["agents"][0].__setitem__("heading_deg", 90.0)))
    variant("cattle never crosses (static at edge)", lambda d: d["agents"][0].__setitem__("behavior", "STATIC"))
    variant("cattle 20 m closer (x=40)", lambda d: d["agents"][0].__setitem__("x", 40.0))
    variant("ego slower (7 m/s desired)", lambda d: (d["ego"].__setitem__("speed_mps", 7.0),
                                                     d["ego"].__setitem__("desired_speed_mps", 7.0)))
    variant("ego faster (13 m/s desired)", lambda d: (d["ego"].__setitem__("speed_mps", 13.0),
                                                      d["ego"].__setitem__("desired_speed_mps", 13.0)))
    variant("road narrower (5 m: +-2.5)", lambda d: (d["road"].__setitem__("left_boundary", [[-20, 2.5], [160, 2.5]]),
                                                     d["road"].__setitem__("right_boundary", [[-20, -2.5], [160, -2.5]]),
                                                     d["ego"].__setitem__("y", 1.0),
                                                     d["ego"].__setitem__("desired_lateral_offset_m", 1.0),
                                                     d["agents"][0].__setitem__("y", 2.1)))
    variant("ego starts 15 m further on (x=15)", lambda d: d["ego"].__setitem__("x", 15.0))
    variant("ego on the right half (d=-1.75)", lambda d: (d["ego"].__setitem__("y", -1.75),
                                                          d["ego"].__setitem__("desired_lateral_offset_m", -1.75)))

    print("## Adaptivity: SUDDEN_CATTLE_CROSSING perturbations\n")
    print("| variant | termination | collisions | min clearance [m] | min TTC [s] | min speed [m/s] | ego y range [m] | "
          "selected @25 m (state, feasible/total) | behaviour states | t_done [s] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, d in variants:
        m, frames = run(d)
        print(summarise(name, m, frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
