"""Simulated world: drivable space, goal and dynamic agents.

The world never touches the ego vehicle state; it only reads the ego position
to drive reactive agent behaviours (e.g. a crossing triggered by proximity).
Ground-truth objects are exposed through `GroundTruthObjectProvider`, which
implements the same `ObjectStateProvider` interface a sensor-fusion stack will
implement in Stage 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from autonomy.core.interfaces import ObjectStateProvider
from autonomy.core.types import ObjectState
from simulation.agents.agent import Agent
from simulation.agents.behaviors import behavior_for

from .road import DrivableSpace


@dataclass
class World:
    road: DrivableSpace
    agents: list[Agent]
    goal_s: float
    time: float = 0.0
    ego_xy: tuple[float, float] = (0.0, 0.0)
    _provider: "GroundTruthObjectProvider" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._provider = GroundTruthObjectProvider(self)

    @property
    def object_provider(self) -> ObjectStateProvider:
        return self._provider

    def update(self, dt: float, ego_xy: tuple[float, float]) -> None:
        """Advance every agent. `ego_xy` may be a plain (x, y) tuple or an `EgoView` carrying the
        ego's speed and heading as well; EgoView indexes as (x, y), so behaviours that only want
        the position are unaffected. Agents never receive the ego's future control commands."""
        self.ego_xy = ego_xy
        for agent in self.agents:
            behavior_for(agent).update(agent, ego_xy, dt, self.road)
        self.time += dt

    def to_dict(self) -> dict:
        return {
            "road": self.road.to_dict(),
            "goal_s": self.goal_s,
            "agents": [
                {"id": a.id, "type": a.object_type.value, "behavior": a.behavior_type.value,
                 "phase": a.phase, "x": a.x, "y": a.y, "heading": a.heading, "speed": a.speed}
                for a in self.agents
            ],
        }


class GroundTruthObjectProvider(ObjectStateProvider):
    def __init__(self, world: World):
        self.world = world

    def get_object_states(self, timestamp: float) -> list[ObjectState]:
        return [a.to_object_state(timestamp) for a in self.world.agents]
