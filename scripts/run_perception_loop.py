"""Phase 3C: drive the EXISTING autonomy stack on a corridor that came from a camera.

    python scripts/run_perception_loop.py --root Datasets/idd-lite/idd20k_lite

WHAT THIS PROVES, AND WHAT IT DOES NOT
    Each IDD-Lite validation image becomes one independent driving episode. The image is
    segmented by the trained Phase-3B network, the prediction becomes a corridor, the corridor
    passes the safety gate, and the surviving geometry is handed to the unmodified `Simulation`
    class as its road. Planner, risk engine, behaviour FSM, controller, safety supervisor and
    vehicle model are the shipped ones, constructed exactly as `run_scenario` constructs them.

    MODE A uses a corridor built from the ground-truth mask.
    MODE B uses a corridor built from the network's prediction.
    Everything downstream of the mask is byte-identical between the two. The comparison is
    therefore a comparison of PERCEPTION, which is the only thing Phase 3C changes.

    What this is NOT: a temporally continuous camera-only drive. IDD-Lite is 204 unrelated
    photographs with no ego motion and no calibration, so an episode perceives ONCE and then
    drives the resulting geometry. No ego motion is invented, because the ego starts at the
    origin facing along the corridor, which makes the ego frame and the world frame the same
    frame and the anchor an identity. That is the honest interface boundary and it is drawn here
    rather than hidden.

    Nothing in `autonomy/` or `simulation/` is imported-into, patched or subclassed by this file.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image                                                       # noqa: E402

from autonomy.core.config import (AutonomyConfig, ObjectProfiles, PerceptionConfig,  # noqa: E402
                                  load_vehicle_parameters)
from autonomy.core.types import VehicleState                                # noqa: E402
from road_perception.dataset import IDDLite, INPUT_H, INPUT_W               # noqa: E402
from road_perception.drivable import corridor_width_profile                 # noqa: E402
from road_perception.integration import corridor_agreement                  # noqa: E402
from road_perception.planner_bridge import (AnchorPose, GateThresholds,     # noqa: E402
                                            PerceptionPlannerBridge)
from simulation.runner import Simulation                                    # noqa: E402
from simulation.scenarios.loader import EgoSetup, Scenario                  # noqa: E402
from autonomy.telemetry.telemetry import TelemetryPublisher                 # noqa: E402
from simulation.world.road import DrivableSpace                             # noqa: E402
from simulation.world.world import World                                    # noqa: E402

CHECKPOINT = Path("road_perception/checkpoints/fastscnn_iddlite.pt")
# The camera sits at the origin looking down +x, so the ego frame IS the world frame and the
# anchor is an identity. Stated as a constant rather than buried, because this is exactly the
# place where a dishonest implementation would invent a pose.
IDENTITY_ANCHOR = AnchorPose(0.0, 0.0, 0.0)


@dataclass
class Episode:
    """One image driven end to end, or the explicit refusal that stopped it."""
    image: str
    mode: str
    accepted: bool
    gate_code: str
    gate_reason: str
    lookahead_m: float
    median_width_m: float
    completed: bool = False
    termination: str = "GATE_REJECTED"
    collisions: int = 0
    distance_m: float = 0.0
    duration_s: float = 0.0
    plan_cycles: int = 0
    infeasible_cycles: int = 0
    plan_latency_ms: float = 0.0
    max_cross_track_m: float = 0.0
    min_clearance_m: float = float("nan")
    emergency_brakes: int = 0
    perception_ms: float = 0.0


def build_scenario(road: DrivableSpace, name: str, desired_speed: float) -> Scenario:
    """A one-road, zero-agent scenario. Constructed, not loaded from YAML.

    Zero agents is not a shortcut, it is the scope: Phase 3C is about the ROAD coming from a
    camera. IDD-Lite carries no tracked objects, and inventing some would be exactly the fake
    integration this phase forbids.
    """
    world = World(road=road, agents=[], goal_s=max(float(road.length) - 1.0, 1.0))
    ego = EgoSetup(
        initial_state=VehicleState(timestamp=0.0, x=float(road.reference[0, 0]),
                                   y=float(road.reference[0, 1]), yaw=0.0,
                                   longitudinal_velocity=0.0),
        desired_speed=desired_speed, desired_lateral_offset=0.0)
    return Scenario(name=name, description="Phase 3C perception-driven corridor", seed=0,
                    world=world, ego=ego, raw={})


def drive(road: DrivableSpace, name: str, cfg: AutonomyConfig, params, profiles,
          desired_speed: float) -> dict:
    """Run the shipped `Simulation` on this corridor and return what it did."""
    sc = build_scenario(road, name, desired_speed)
    sim = Simulation(sc, cfg, params, profiles, TelemetryPublisher(),
                     perception=PerceptionConfig(mode="ground_truth"))
    # Cross-track error and infeasible-cycle counts are the two things the shipped metrics
    # collector does not expose, so they are sampled here. Everything else comes from
    # `SimulationMetrics` unchanged, rather than being re-derived and risking disagreement.
    cross_track, infeasible = [], 0
    while sim.running:
        before = sim.plan
        sim.step()
        if sim.plan is not before and sim.plan is not None and sim.plan.feasible_count == 0:
            infeasible += 1
        if sim.tracker.last_debug is not None:
            cross_track.append(abs(sim.tracker.last_debug.lateral.cross_track_error))
    m = sim.metrics.finalize(sim.termination_reason == "GOAL_REACHED", sim.time,
                             sim.termination_reason)
    return {
        "completed": bool(m.scenario_completed),
        "termination": sim.termination_reason,
        "collisions": int(m.collision_count),
        "distance_m": float(m.path_length),
        "duration_s": float(sim.time),
        "plan_cycles": int(m.replanning_count),
        "infeasible_cycles": infeasible,
        "plan_latency_ms": float(m.planning_latency_mean_ms),
        "max_cross_track_m": float(max(cross_track)) if cross_track else 0.0,
        "emergency_brakes": int(m.emergency_brake_activations),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("Datasets/idd-lite/idd20k_lite"))
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--limit", type=int, default=0, help="0 = every validation image")
    ap.add_argument("--speed", type=float, default=8.0, help="desired speed, m/s")
    ap.add_argument("--out", type=Path, default=Path("docs/phase3c_results.json"))
    a = ap.parse_args()

    cfg = AutonomyConfig.load()
    params = load_vehicle_parameters()
    profiles = ObjectProfiles.load()
    thresholds = GateThresholds()
    bridge_a = PerceptionPlannerBridge(thresholds=thresholds)
    bridge_b = PerceptionPlannerBridge.from_checkpoint(a.checkpoint, thresholds=thresholds)

    ds = IDDLite(a.root, "val", augment=False)
    n = len(ds) if a.limit <= 0 else min(a.limit, len(ds))
    print("=" * 78)
    print(f"Phase 3C: {n} validation images, one episode per image, two modes")
    print("=" * 78, flush=True)

    episodes: list[Episode] = []
    geometry: list[dict] = []
    for i in range(n):
        p = ds._item(i)
        gt = np.asarray(Image.open(p.mask).resize((INPUT_W, INPUT_H), Image.NEAREST))
        img = np.asarray(Image.open(p.image).convert("RGB"))

        out_a = bridge_a.from_mask(gt, IDENTITY_ANCHOR)
        out_b = bridge_b.from_image(img, IDENTITY_ANCHOR)
        if out_a.gate.ok and out_b.gate.ok:
            ag = corridor_agreement(out_b.gate.usable, out_a.gate.usable)
            geometry.append({"image": p.image.name, **ag,
                             "width_a": float(np.median(corridor_width_profile(out_a.gate.usable))),
                             "width_b": float(np.median(corridor_width_profile(out_b.gate.usable)))})

        for mode, out in (("A", out_a), ("B", out_b)):
            ep = Episode(image=p.image.name, mode=mode, accepted=out.gate.ok,
                         gate_code=out.gate.code, gate_reason=out.gate.reason,
                         lookahead_m=out.gate.lookahead_m,
                         median_width_m=out.gate.median_width_m,
                         perception_ms=out.inference_ms + out.adapt_ms)
            if out.gate.ok:
                # THE ONLY PLACE THE CORRIDOR REACHES THE PLANNER. `out.road` came from the mask
                # named by `mode`. There is no branch here that substitutes ground truth for a
                # failed prediction; a rejected corridor simply never gets driven.
                r = drive(out.road, f"P3C_{mode}_{p.image.stem}", cfg, params, profiles, a.speed)
                for k, v in r.items():
                    if hasattr(ep, k):
                        setattr(ep, k, v)
            episodes.append(ep)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n} images", flush=True)

    report(episodes, geometry, bridge_a, bridge_b, a.out)
    return 0


def report(episodes, geometry, bridge_a, bridge_b, out_path: Path) -> None:
    def sel(mode, pred=lambda e: True):
        return [e for e in episodes if e.mode == mode and pred(e)]

    def med(vals):
        vals = [v for v in vals if v == v]
        return statistics.median(vals) if vals else float("nan")

    total = len(sel("A"))
    print("\n" + "=" * 78)
    print("STEP 5 - SAFETY GATE, before anything reaches the planner")
    print("=" * 78)
    for mode, br in (("A (ground-truth mask)", bridge_a), ("B (predicted mask)", bridge_b)):
        key = mode[0]
        acc = len(sel(key, lambda e: e.accepted))
        print(f"\nMode {mode}: accepted {acc}/{total}, rejected {total - acc}")
        for code, cnt in br.rejection_summary().items():
            print(f"    {code:<20s} {cnt}")

    paired = {e.image for e in sel("A", lambda e: e.accepted)} & \
             {e.image for e in sel("B", lambda e: e.accepted)}
    print(f"\nimages both modes accepted (the paired set)   {len(paired)}")

    print("\n" + "=" * 78)
    print("STEP 4 - A/B, identical planner, risk, controller and vehicle model")
    print("=" * 78)
    rows = []
    for key, label in (("A", "Mode A  ground-truth corridor"), ("B", "Mode B  predicted corridor")):
        eps = sel(key, lambda e: e.accepted and e.image in paired)
        rows.append((label, eps))
    hdr = f"{'':<32s}{'Mode A':>14s}{'Mode B':>14s}"
    print("\n" + hdr)
    print("-" * len(hdr))

    def line(name, fn, fmt="{:>14.2f}"):
        a_v, b_v = fn(rows[0][1]), fn(rows[1][1])
        print(f"{name:<32s}" + fmt.format(a_v) + fmt.format(b_v))

    line("episodes", lambda e: len(e), "{:>14.0f}")
    line("completed", lambda e: sum(x.completed for x in e), "{:>14.0f}")
    line("collisions", lambda e: sum(x.collisions for x in e), "{:>14.0f}")
    line("corridor width, median m", lambda e: med([x.median_width_m for x in e]))
    line("usable lookahead, median m", lambda e: med([x.lookahead_m for x in e]))
    line("distance driven, median m", lambda e: med([x.distance_m for x in e]))
    line("planning cycles, median", lambda e: med([x.plan_cycles for x in e]))
    line("infeasible cycles, total", lambda e: sum(x.infeasible_cycles for x in e), "{:>14.0f}")
    line("plan latency, median ms", lambda e: med([x.plan_latency_ms for x in e]))
    line("max cross-track, median m", lambda e: med([x.max_cross_track_m for x in e]))
    line("safety interventions, total", lambda e: sum(x.emergency_brakes for x in e), "{:>14.0f}")
    line("perception, median ms", lambda e: med([x.perception_ms for x in e]))

    if geometry:
        print("\ncorridor geometry on the paired set (Mode B against Mode A):")
        for name, key in (("left boundary RMSE", "left_rmse_m"),
                          ("right boundary RMSE", "right_rmse_m"),
                          ("width bias", "width_bias_m")):
            v = med([g[key] for g in geometry if g[key] is not None])
            print(f"    {name:<24s} {v:+.2f} m median")

    term_b = {}
    for e in sel("B", lambda e: e.accepted and e.image in paired):
        term_b[e.termination] = term_b.get(e.termination, 0) + 1
    print("\nMode B episode outcomes:")
    for k, v in sorted(term_b.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24s} {v}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "episodes": [e.__dict__ for e in episodes],
        "geometry": geometry,
        "paired_images": sorted(paired),
        "gate_rejections": {"A": bridge_a.rejection_summary(),
                            "B": bridge_b.rejection_summary()},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
