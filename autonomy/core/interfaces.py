"""Abstract interfaces that decouple the autonomy stack from its data sources.

`ObjectStateProvider` is the seam between simulation ground truth (Stage 1)
and sensor fusion (Stage 2). Prediction, risk, behaviour, planning and control
only ever see `ObjectState` lists and a `DrivableSpace`-like road model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import ObjectState


class ObjectStateProvider(ABC):
    @abstractmethod
    def get_object_states(self, timestamp: float) -> list[ObjectState]:
        """Current tracked objects, in the world frame, at `timestamp`."""


class RoadModel(ABC):
    """Minimum road interface the planner needs. Lane markings are NOT part of it."""

    @abstractmethod
    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """World -> corridor frame: (s along reference, d lateral (+left), reference heading)."""

    @abstractmethod
    def to_cartesian(self, s: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Corridor -> world: (x, y, reference heading) for arrays s, d."""

    @abstractmethod
    def footprint_inside(self, corners: np.ndarray, margin: float) -> np.ndarray:
        """corners (N,4,2) -> bool (N,) true if every corner is >= margin inside the corridor."""

    @abstractmethod
    def boundary_clearance(self, corners: np.ndarray) -> np.ndarray:
        """corners (N,4,2) -> (N,) min distance from any corner to the nearest boundary (0 if outside)."""

    @abstractmethod
    def lateral_bounds(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(d_right[N], d_left[N]) corridor edges at arc lengths s (right < 0 < left)."""

    @property
    @abstractmethod
    def length(self) -> float:
        """Arc length of the reference polyline."""
