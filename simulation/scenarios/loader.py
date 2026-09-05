"""Scenario definitions (YAML) -> World + initial ego state.

Scenarios are data. Adding a scenario must never require touching the
autonomy stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autonomy.core.config import ObjectProfiles, load_yaml
from autonomy.core.types import AgentBehaviorType, ObjectType, VehicleState
from simulation.agents.agent import Agent
from simulation.world.road import DrivableSpace
from simulation.world.world import World

SCENARIO_DIR = Path(__file__).resolve().parent


@dataclass
class EgoSetup:
    initial_state: VehicleState
    desired_speed: float
    desired_lateral_offset: float


@dataclass
class Scenario:
    name: str
    description: str
    seed: int
    world: World
    ego: EgoSetup
    raw: dict[str, Any]


def build_scenario(data: dict[str, Any], profiles: ObjectProfiles | None = None) -> Scenario:
    profiles = profiles or ObjectProfiles.load()
    seed = int(data.get("seed", 0))
    r = data["road"]
    if "segments" in r:
        road = DrivableSpace.from_segments(
            r["segments"], float(r["half_width_left_m"]), float(r["half_width_right_m"]),
            x0=float(r.get("x0", 0.0)), y0=float(r.get("y0", 0.0)),
            heading=math.radians(float(r.get("heading_deg", 0.0))),
            lane_markings=str(r.get("lane_markings", "NONE")),
        )
    else:
        road = DrivableSpace(
            reference=np.asarray(r["reference"], dtype=float),
            left_boundary=np.asarray(r["left_boundary"], dtype=float),
            right_boundary=np.asarray(r["right_boundary"], dtype=float),
            lane_markings=str(r.get("lane_markings", "NONE")),
        )
    agents: list[Agent] = []
    for i, a in enumerate(data.get("agents", [])):
        otype = ObjectType(a["type"])
        prof = profiles.get(otype)
        # agents may be placed in corridor coordinates (s, d) instead of (x, y)
        if "s" in a:
            ax, ay, h_ref = road.to_cartesian(np.array([float(a["s"])]), np.array([float(a.get("d", 0.0))]))
            a_x, a_y = float(ax[0]), float(ay[0])
            heading = float(h_ref[0]) + math.radians(float(a.get("heading_rel_deg", 0.0)))
        else:
            a_x, a_y = float(a["x"]), float(a["y"])
            heading = math.radians(float(a.get("heading_deg", 0.0)))
        agents.append(Agent(
            id=str(a["id"]), object_type=otype,
            x=a_x, y=a_y,
            heading=heading,
            speed=float(a.get("speed_mps", 0.0)),
            length=float(a.get("length_m", prof.length_m)),
            width=float(a.get("width_m", prof.width_m)),
            behavior_type=AgentBehaviorType(a.get("behavior", "STATIC")),
            behavior_params=dict(a.get("params", {})),
            seed=seed * 1000 + i,
        ))
    e = data["ego"]
    if "s" in e:
        ex, ey, eh = road.to_cartesian(np.array([float(e["s"])]), np.array([float(e.get("d", 0.0))]))
        ego_state = VehicleState(timestamp=0.0, x=float(ex[0]), y=float(ey[0]), yaw=float(eh[0]),
                                 longitudinal_velocity=float(e.get("speed_mps", 0.0)))
    else:
        ego_state = VehicleState(
            timestamp=0.0, x=float(e["x"]), y=float(e["y"]),
            yaw=math.radians(float(e.get("yaw_deg", 0.0))),
            longitudinal_velocity=float(e.get("speed_mps", 0.0)),
        )
    ego = EgoSetup(ego_state, float(e["desired_speed_mps"]),
                   float(e.get("desired_lateral_offset_m", 0.0)))
    world = World(road=road, agents=agents, goal_s=float(data["goal"]["s_m"]),
                  ego_xy=(ego_state.x, ego_state.y))
    return Scenario(str(data["name"]), str(data.get("description", "")), seed, world, ego, data)


def load_scenario(name_or_path: str | Path, profiles: ObjectProfiles | None = None) -> Scenario:
    p = Path(name_or_path)
    if not p.exists():
        p = SCENARIO_DIR / f"{str(name_or_path).lower()}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"scenario not found: {name_or_path}")
    return build_scenario(load_yaml(p), profiles)
