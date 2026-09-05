"""Generic dynamic agent used by the simulated world.

An Agent has a kinematic state and a pluggable behaviour. Behaviours are
generic (STATIC, CONSTANT_VELOCITY, CROSSING, MERGING, ERRATIC); object
types (CATTLE, PEDESTRIAN, ...) only select dimensions and prediction / risk
profiles. Nothing in the autonomy stack knows which behaviour an agent runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autonomy.core.geometry import OrientedBox
from autonomy.core.types import AgentBehaviorType, ObjectState, ObjectType


@dataclass
class Agent:
    id: str
    object_type: ObjectType
    x: float
    y: float
    heading: float           # rad
    speed: float             # m/s along heading
    length: float
    width: float
    behavior_type: AgentBehaviorType
    behavior_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    # runtime state written by behaviours
    phase: str = "INIT"
    elapsed: float = 0.0
    memory: dict[str, Any] = field(default_factory=dict)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @property
    def vx(self) -> float:
        return self.speed * math.cos(self.heading)

    @property
    def vy(self) -> float:
        return self.speed * math.sin(self.heading)

    def footprint(self) -> OrientedBox:
        return OrientedBox(self.x, self.y, self.heading, self.length, self.width)

    def advance(self, dt: float) -> None:
        """Move along the current heading at the current speed."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.elapsed += dt

    def to_object_state(self, timestamp: float) -> ObjectState:
        return ObjectState(
            id=self.id, object_type=self.object_type, timestamp=timestamp,
            x=self.x, y=self.y, vx=self.vx, vy=self.vy, heading=self.heading,
            length=self.length, width=self.width, confidence=1.0,
            covariance=np.zeros((2, 2)),
        )
