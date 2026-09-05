"""Risk assessment: separate module, consumed by behaviour, planner and safety.

For every object the ego footprint along its *reference motion* (last selected
trajectory, or constant velocity along the current heading when no plan
exists yet) is compared with the object's predicted footprint at each
prediction time. From the distance series d(t) and uncertainty sigma(t):

    intersection        = any t : d(t) <= margin
    ttc                 = first t with d(t) <= margin            (inf if none)
    ttc_kinematic       = d(0) / closing_speed                    (inf if not closing)
    gap(t)              = max(d(t) - margin, 0)
    p(t)                = exp(-0.5 * (gap(t) / sigma(t))^2)       (1 when intersecting)
    risk_score          = w_type * max_t [ p(t) * exp(-t / tau) ]
    risk_level          = thresholds on risk_score, escalated by TTC thresholds

A `RiskSummary` aggregates the worst object and also identifies a *lead
object* (slower traffic ahead within a lateral window) for the FOLLOW state.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from autonomy.core.config import BehaviorConfig, RiskConfig
from autonomy.core.geometry import box_sequence_distance
from autonomy.core.interfaces import RoadModel
from autonomy.core.types import (ObjectPrediction, ObjectState, RiskAssessment, RiskLevel,
                                 RiskSummary, Trajectory, VehicleParameters, VehicleState)


def kinematic_ttc(distance: float, closing_speed: float) -> float:
    """Classical TTC: current gap over closing speed. inf when not closing."""
    if closing_speed <= 1e-6:
        return math.inf
    return max(distance, 0.0) / closing_speed


def first_overlap_time(rel_times: np.ndarray, distances: np.ndarray, margin: float) -> float:
    hits = np.nonzero(distances <= margin)[0]
    return float(rel_times[hits[0]]) if hits.size else math.inf


def ego_reference_motion(ego: VehicleState, trajectory: Optional[Trajectory],
                         times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego (x, y, yaw) sampled at absolute `times`.

    Uses the trajectory where it covers `times`; beyond its end (or with no
    trajectory) the ego is propagated at constant velocity along its heading.
    """
    if trajectory is not None and len(trajectory) >= 2:
        t = trajectory.t
        inside = (times >= t[0]) & (times <= t[-1])
        x = np.interp(times, t, trajectory.x)
        y = np.interp(times, t, trajectory.y)
        yaw = np.interp(times, t, np.unwrap(trajectory.yaw))
        if not np.all(inside):
            extra = np.maximum(times - t[-1], 0.0)
            v_end = float(trajectory.velocity[-1])
            x = np.where(inside, x, trajectory.x[-1] + v_end * extra * math.cos(trajectory.yaw[-1]))
            y = np.where(inside, y, trajectory.y[-1] + v_end * extra * math.sin(trajectory.yaw[-1]))
        return x, y, yaw
    rel = times - ego.timestamp
    v = ego.longitudinal_velocity
    return (ego.x + v * rel * math.cos(ego.yaw),
            ego.y + v * rel * math.sin(ego.yaw),
            np.full_like(times, ego.yaw))


class RiskEngine:
    def __init__(self, config: RiskConfig, behavior_cfg: BehaviorConfig, params: VehicleParameters):
        self.cfg = config
        self.bcfg = behavior_cfg
        self.params = params

    # ------------------------------------------------------------------ #
    def evaluate(self, ego: VehicleState, ego_trajectory: Optional[Trajectory],
                 objects: list[ObjectState], predictions: list[ObjectPrediction],
                 road: Optional[RoadModel] = None) -> RiskSummary:
        if not predictions:
            return RiskSummary.empty()
        times = predictions[0].times
        rel = times - times[0]
        ex, ey, eyaw = ego_reference_motion(ego, ego_trajectory, times)
        off = self.params.footprint_center_offset
        fx = ex + off * np.cos(eyaw)
        fy = ey + off * np.sin(eyaw)
        ego_vx = ego.longitudinal_velocity * math.cos(ego.yaw)
        ego_vy = ego.longitudinal_velocity * math.sin(ego.yaw)
        obj_by_id = {o.id: o for o in objects}

        assessments = [self._assess(pred, obj_by_id.get(pred.object_id), rel, fx, fy, eyaw, ego_vx, ego_vy)
                       for pred in predictions]

        worst = max(assessments, key=lambda a: (a.risk_level.value, a.risk_score))
        summary = RiskSummary(
            assessments=assessments,
            max_level=worst.risk_level,
            max_score=worst.risk_score,
            min_ttc=min(a.ttc for a in assessments),
            min_predicted_distance=min(a.min_predicted_distance for a in assessments),
            worst_object_id=worst.object_id,
            any_intersection=any(a.trajectory_intersection for a in assessments),
        )
        if road is not None:
            self._find_lead(summary, ego, objects, road)
        return summary

    # ------------------------------------------------------------------ #
    def _assess(self, pred: ObjectPrediction, obj: Optional[ObjectState], rel: np.ndarray,
                fx: np.ndarray, fy: np.ndarray, eyaw: np.ndarray,
                ego_vx: float, ego_vy: float) -> RiskAssessment:
        p = self.params
        dist = box_sequence_distance(fx, fy, eyaw, p.length, p.width,
                                     pred.x, pred.y, pred.heading, pred.length, pred.width)
        margin = self.cfg.safety_margin_m
        sigma = pred.sigma()

        gap = np.maximum(dist - margin, 0.0)
        prob = np.exp(-0.5 * (gap / np.maximum(sigma, 1e-6)) ** 2)
        discounted = prob * np.exp(-rel / self.cfg.time_constant_s)
        k_min = int(np.argmin(dist))
        ttc = first_overlap_time(rel, dist, margin)
        intersection = math.isfinite(ttc)

        vx = float(pred.vx[0]); vy = float(pred.vy[0])
        rvx, rvy = vx - ego_vx, vy - ego_vy
        rx, ry = float(pred.x[0] - fx[0]), float(pred.y[0] - fy[0])
        rnorm = math.hypot(rx, ry)
        closing = -(rx * rvx + ry * rvy) / rnorm if rnorm > 1e-9 else 0.0

        score = float(pred.risk_weight * np.max(discounted))
        level = self._level(score, ttc)
        return RiskAssessment(
            object_id=pred.object_id, object_type=pred.object_type,
            distance=float(dist[0]),
            relative_speed=math.hypot(rvx, rvy),
            closing_speed=closing,
            ttc=ttc,
            ttc_kinematic=kinematic_ttc(float(dist[0]), closing),
            min_predicted_distance=float(dist[k_min]),
            time_of_min_distance=float(rel[k_min]),
            trajectory_intersection=intersection,
            collision_probability=float(np.max(prob)),
            risk_score=score,
            risk_level=level,
            uncertainty=float(sigma[k_min]),
            risk_weight=pred.risk_weight,
        )

    def _level(self, score: float, ttc: float) -> RiskLevel:
        lv = self.cfg.levels
        if score >= lv.critical:
            level = RiskLevel.CRITICAL
        elif score >= lv.high:
            level = RiskLevel.HIGH
        elif score >= lv.medium:
            level = RiskLevel.MEDIUM
        elif score >= lv.low:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.NONE
        if ttc < self.cfg.ttc_critical_s:
            level = RiskLevel.CRITICAL
        elif ttc < self.cfg.ttc_high_s and level.value < RiskLevel.HIGH.value:
            level = RiskLevel.HIGH
        return level

    def _find_lead(self, summary: RiskSummary, ego: VehicleState,
                   objects: list[ObjectState], road: RoadModel) -> None:
        s_ego, d_ego, h_ref = road.project(ego.x, ego.y)
        best_gap = math.inf
        for o in objects:
            s_o, d_o, h_o = road.project(o.x, o.y)
            along = s_o - s_ego
            if along <= 0 or along > self.bcfg.follow_max_range_m:
                continue
            if abs(d_o - d_ego) > self.bcfg.follow_lateral_window_m:
                continue
            aligned = math.cos(o.heading - h_o) > 0.5 if o.speed > 0.1 else True
            if not aligned:
                continue
            gap = along - 0.5 * (self.params.length + o.length)
            if gap < best_gap:
                best_gap = gap
                summary.lead_object_id = o.id
                summary.lead_gap = gap
                summary.lead_speed = o.speed * math.cos(o.heading - h_o)
