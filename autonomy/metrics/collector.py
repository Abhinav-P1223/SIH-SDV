"""Metrics computed from the executed simulation, never from planner intent.

Every value in `SimulationMetrics` is derived from actual ego states, actual
agent footprints, actual planner outputs and actual safety activations.
Stage 2 hooks (jerk, path smoothness, prediction error) are computed where the
data already exists; perception latency stays None until sensors exist.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from autonomy.core.geometry import OrientedBox, box_distance
from autonomy.core.types import (BehaviorDecision, ObjectPrediction, ObjectState, PlannerOutput,
                                 RiskSummary, SafetyStatus, SimulationMetrics, VehicleParameters,
                                 VehicleState)
from autonomy.vehicle.models import footprint_for


class MetricsCollector:
    def __init__(self, params: VehicleParameters, safety_margin: float, prediction_eval_horizon_s: float = 1.0):
        self.params = params
        self.safety_margin = safety_margin
        self.m = SimulationMetrics()
        self._prev_state: Optional[VehicleState] = None
        self._prev_acc: Optional[float] = None
        self._latencies: list[float] = []
        self._curvatures: list[float] = []
        self._speeds: list[float] = []
        self._in_collision = False
        self._last_selection: Optional[tuple[float, float]] = None
        self._last_state_name: Optional[str] = None
        self._last_state_time: float = 0.0
        self._pred_history: deque[tuple[float, list[ObjectPrediction]]] = deque(maxlen=200)
        self._pred_errors: list[float] = []
        self._pred_eval_h = prediction_eval_horizon_s
        self.boundary_violations = 0
        self.collision_events: list[tuple[float, str]] = []
        self.max_jerk = 0.0

    # ------------------------------------------------------------------ #
    def on_plan(self, plan: PlannerOutput, decision: BehaviorDecision, risk: RiskSummary,
                predictions: list[ObjectPrediction], objects: list[ObjectState], now: float) -> None:
        m = self.m
        m.replanning_count += 1
        self._latencies.append(plan.latency_ms)
        m.planning_latency_max_ms = max(m.planning_latency_max_ms, plan.latency_ms)
        m.planning_latency_mean_ms = float(np.mean(self._latencies))
        sel = (round(plan.selected.lateral_offset_end, 2), round(plan.selected.target_speed, 1))
        if self._last_selection is not None and sel != self._last_selection:
            m.plan_change_count += 1
        self._last_selection = sel
        if math.isfinite(risk.min_ttc):
            m.minimum_ttc = min(m.minimum_ttc, risk.min_ttc)
        # behaviour bookkeeping
        name = decision.state.value
        if self._last_state_name is None:
            self._last_state_name, self._last_state_time = name, now
        elif name != self._last_state_name:
            m.behavior_state_durations[self._last_state_name] = \
                m.behavior_state_durations.get(self._last_state_name, 0.0) + (now - self._last_state_time)
            m.behavior_transitions += 1
            self._last_state_name, self._last_state_time = name, now
        # prediction error hook: compare a prediction made ~h seconds ago with what happened
        self._pred_history.append((now, predictions))
        for t_made, preds in self._pred_history:
            if abs((now - t_made) - self._pred_eval_h) <= 0.026:
                actual = {o.id: o for o in objects}
                for p in preds:
                    o = actual.get(p.object_id)
                    if o is None:
                        continue
                    px = float(np.interp(now, p.times, p.x))
                    py = float(np.interp(now, p.times, p.y))
                    self._pred_errors.append(math.hypot(px - o.x, py - o.y))
                break
        if self._pred_errors:
            m.prediction_error_m = float(np.mean(self._pred_errors))

    def update(self, state: VehicleState, objects: list[ObjectState], safety: SafetyStatus,
               inside_corridor: bool, dt: float) -> None:
        m = self.m
        ego_fp = footprint_for(self.params, state.x, state.y, state.yaw)
        colliding = False
        for o in objects:
            d = box_distance(ego_fp, OrientedBox(o.x, o.y, o.heading, o.length, o.width))
            m.minimum_obstacle_clearance = min(m.minimum_obstacle_clearance, d)
            if d <= 0.0:
                colliding = True
                if not self._in_collision:
                    m.collision_count += 1
                    self.collision_events.append((state.timestamp, o.id))
        self._in_collision = colliding

        if not inside_corridor:
            self.boundary_violations += 1

        if self._prev_state is not None:
            prev = self._prev_state
            ds = math.hypot(state.x - prev.x, state.y - prev.y)
            m.path_length += ds
            if state.longitudinal_velocity < -0.05:
                m.reverse_distance_m += ds
                if prev.longitudinal_velocity >= -0.05:
                    m.reverse_manoeuvres += 1
            m.max_acceleration = max(m.max_acceleration, state.longitudinal_acceleration)
            m.max_deceleration = max(m.max_deceleration, -state.longitudinal_acceleration)
            rate = abs(state.steering_angle - prev.steering_angle) / dt
            m.max_steering_rate = max(m.max_steering_rate, rate)
            if state.longitudinal_velocity > 0.5:
                k = abs(state.yaw_rate / state.longitudinal_velocity)
                self._curvatures.append(k)
                m.max_curvature = max(m.max_curvature, k)
            if self._prev_acc is not None:
                jerk = abs(state.longitudinal_acceleration - self._prev_acc) / dt
                self.max_jerk = max(self.max_jerk, jerk)
        self._prev_state = state
        self._prev_acc = state.longitudinal_acceleration
        self._speeds.append(state.longitudinal_velocity)
        m.emergency_brake_activations = safety.activation_count

    def on_perception(self, tracks: list[ObjectState], truth: list[ObjectState], latency_s: Optional[float],
                      n_detections: int) -> None:
        """Sensors mode: compare published tracks with simulator truth (evaluation only)."""
        m = self.m
        m.detections_total += n_detections
        if latency_s is not None:
            m.perception_latency_ms = latency_s * 1000.0
        self._track_counts = getattr(self, "_track_counts", [])
        self._track_counts.append(len(tracks))
        m.track_count_mean = float(np.mean(self._track_counts))
        if not tracks or not truth:
            return
        pe = getattr(self, "_pos_err", [])
        ve = getattr(self, "_vel_err", [])
        for tr in tracks:
            best = min(truth, key=lambda o: math.hypot(o.x - tr.x, o.y - tr.y))
            d = math.hypot(best.x - tr.x, best.y - tr.y)
            if d < 3.0:
                pe.append(d)
                ve.append(math.hypot(best.vx - tr.vx, best.vy - tr.vy))
        self._pos_err, self._vel_err = pe, ve
        if pe:
            m.tracking_position_error_m = float(np.mean(pe))
            m.tracking_velocity_error_mps = float(np.mean(ve))

    def finalize(self, completed: bool, now: float, reason: str) -> SimulationMetrics:
        m = self.m
        m.scenario_completed = completed
        m.time_to_completion = now if completed else None
        m.termination_reason = reason
        if self._speeds:
            m.average_speed = float(np.mean(self._speeds))
        if self._curvatures:
            m.mean_abs_curvature = float(np.mean(self._curvatures))
            k = np.asarray(self._curvatures)
            m.path_smoothness = float(np.mean(np.diff(k) ** 2)) if len(k) > 1 else 0.0
        m.max_jerk = self.max_jerk
        if self._last_state_name is not None:
            m.behavior_state_durations[self._last_state_name] = \
                m.behavior_state_durations.get(self._last_state_name, 0.0) + (now - self._last_state_time)
        m.scenario_success_rate = 1.0 if (completed and m.collision_count == 0) else 0.0
        return m
