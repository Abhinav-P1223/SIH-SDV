"""Telemetry: one frame per simulation step, independent of any UI.

`TelemetryFrame.to_dict()` is plain JSON-serialisable data. `TelemetryPublisher`
fans frames out to sinks. Stage 1 sinks: in-memory (for the debug view and
tests), JSON lines file, CSV summary, console. A WebSocket / REST / MATLAB
adapter is just another sink.
"""
from __future__ import annotations

import csv
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from autonomy.core.types import (BehaviorDecision, ControlCommand, ObjectPrediction, ObjectState,
                                 PlannerOutput, RiskSummary, SafetyStatus, SimulationMetrics,
                                 VehicleState)


@dataclass
class TelemetryFrame:
    timestamp: float
    step: int
    planning_cycle: bool
    ego: VehicleState
    control: ControlCommand
    control_nominal: ControlCommand
    objects: list[ObjectState]
    predictions: list[ObjectPrediction]
    risk: RiskSummary
    decision: Optional[BehaviorDecision]
    plan: Optional[PlannerOutput]
    tracker_debug: Optional[dict[str, Any]]
    safety: SafetyStatus
    road: dict[str, Any]
    metrics: SimulationMetrics
    agents: list[dict[str, Any]]

    def to_dict(self, full: bool = True, candidate_stride: int = 2) -> dict[str, Any]:
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "step": self.step,
            "planning_cycle": self.planning_cycle,
            "ego": self.ego.to_dict(),
            "control": self.control.to_dict(),
            "control_nominal": self.control_nominal.to_dict(),
            "objects": [o.to_dict() for o in self.objects],
            "risk": self.risk.to_dict(),
            "behavior": self.decision.to_dict() if self.decision else None,
            "safety": self.safety.to_dict(),
            "tracker": self.tracker_debug,
            "metrics": self.metrics.to_dict(),
            "agents": self.agents,
        }
        if full:
            d["predictions"] = [p.to_dict() for p in self.predictions]
            d["plan"] = self.plan.to_dict(candidate_stride) if self.plan else None
            d["road"] = self.road
        else:
            d["plan"] = {
                "selected_id": self.plan.selected.id, "latency_ms": self.plan.latency_ms,
                "feasible_count": self.plan.feasible_count, "rejected_count": self.plan.rejected_count,
                "total_cost": None if math.isinf(self.plan.selected.total_cost) else self.plan.selected.total_cost,
            } if self.plan else None
        return d


class TelemetrySink(ABC):
    @abstractmethod
    def write(self, frame: TelemetryFrame) -> None: ...

    def close(self) -> None: ...


class InMemorySink(TelemetrySink):
    def __init__(self) -> None:
        self.frames: list[TelemetryFrame] = []

    def write(self, frame: TelemetryFrame) -> None:
        self.frames.append(frame)


class JsonLinesSink(TelemetrySink):
    """Full frames on planning cycles, compact frames otherwise."""

    def __init__(self, path: Path | str, planning_cycles_only: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self.planning_only = planning_cycles_only

    def write(self, frame: TelemetryFrame) -> None:
        if self.planning_only and not frame.planning_cycle:
            return
        self._fh.write(json.dumps(frame.to_dict(full=frame.planning_cycle)) + "\n")

    def close(self) -> None:
        self._fh.close()


class CsvSummarySink(TelemetrySink):
    """One row per planning cycle with the headline numbers."""

    FIELDS = ["timestamp", "state", "x", "y", "yaw_deg", "speed", "steer_deg", "accel_cmd", "brake_cmd",
              "safety_override", "risk_level", "risk_score", "min_ttc", "min_pred_dist",
              "candidates", "feasible", "rejected", "selected", "total_cost", "planner_ms", "reason"]

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._w.writeheader()

    def write(self, f: TelemetryFrame) -> None:
        if not f.planning_cycle or f.plan is None:
            return
        r = f.risk
        self._w.writerow({
            "timestamp": round(f.timestamp, 3),
            "state": f.decision.state.value if f.decision else "",
            "x": round(f.ego.x, 3), "y": round(f.ego.y, 3),
            "yaw_deg": round(math.degrees(f.ego.yaw), 2),
            "speed": round(f.ego.longitudinal_velocity, 3),
            "steer_deg": round(math.degrees(f.control.steering_angle), 2),
            "accel_cmd": round(f.control.acceleration, 3), "brake_cmd": round(f.control.brake, 3),
            "safety_override": f.safety.override_active,
            "risk_level": r.max_level.name, "risk_score": round(r.max_score, 4),
            "min_ttc": "" if math.isinf(r.min_ttc) else round(r.min_ttc, 3),
            "min_pred_dist": "" if math.isinf(r.min_predicted_distance) else round(r.min_predicted_distance, 3),
            "candidates": len(f.plan.candidates), "feasible": f.plan.feasible_count,
            "rejected": f.plan.rejected_count, "selected": f.plan.selected.id,
            "total_cost": "" if math.isinf(f.plan.selected.total_cost) else round(f.plan.selected.total_cost, 4),
            "planner_ms": round(f.plan.latency_ms, 2),
            "reason": f.decision.reason if f.decision else "",
        })

    def close(self) -> None:
        self._fh.close()


class ConsoleSink(TelemetrySink):
    def __init__(self, every_s: float = 0.5):
        self.every = every_s
        self._next = 0.0
        self._last_state = None

    def write(self, f: TelemetryFrame) -> None:
        if not f.planning_cycle or f.plan is None:
            return
        state = f.decision.state.value if f.decision else "-"
        changed = state != self._last_state
        if f.timestamp + 1e-9 < self._next and not changed and not f.safety.override_active:
            return
        self._next = f.timestamp + self.every
        self._last_state = state
        r = f.risk
        ttc = "inf" if math.isinf(r.min_ttc) else f"{r.min_ttc:.2f}s"
        line = (f"t={f.timestamp:6.2f}s | {state:15s} | v={f.ego.longitudinal_velocity:5.2f} m/s "
                f"steer={math.degrees(f.ego.steering_angle):+5.1f}deg a={f.control.net_acceleration:+5.2f} | "
                f"risk {r.max_level.name:8s} {r.max_score:.2f} TTC {ttc:>6s} | "
                f"sel {f.plan.selected.id:16s} J={f.plan.selected.total_cost:6.3f} "
                f"feas {f.plan.feasible_count:2d}/{len(f.plan.candidates):2d} | {f.plan.latency_ms:5.1f} ms"
                + (" | SAFETY OVERRIDE" if f.safety.override_active else ""))
        print(line)
        if changed and f.decision:
            print(f"        -> {f.decision.reason}")


class TelemetryPublisher:
    def __init__(self, sinks: list[TelemetrySink] | None = None):
        self.sinks = sinks or []

    def add(self, sink: TelemetrySink) -> None:
        self.sinks.append(sink)

    def publish(self, frame: TelemetryFrame) -> None:
        for s in self.sinks:
            s.write(frame)

    def close(self) -> None:
        for s in self.sinks:
            s.close()
