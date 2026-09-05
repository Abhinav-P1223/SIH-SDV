"""Risk assessment: separate module, consumed by behaviour, planner and safety.

Two ego motions are evaluated against every object's predicted footprint:

1. ROUTE motion — "what happens if the ego proceeds along its route at the
   desired speed": follow the corridor at the current lateral offset at the
   scenario's desired speed. This is the reference for behaviour decisions, so
   a threat to the mission is seen even after the planner has slowed down.
   With no road model it degrades to constant heading.

2. PHYSICAL motion — constant velocity along the CURRENT heading and speed. Only its earliest
   overlap time is kept (`min_ttc_current_speed`); it drives the independent
   safety supervisor and the CRITICAL escalation. A stopped ego has infinite
   physical TTC, which is what lets the STOPPED state release cleanly.

3. PLANNED motion — the currently selected trajectory. Its aggregates
   (`plan_*` fields) tell the behaviour layer whether the chosen plan is clear.

Per object, from the distance series d(t) and uncertainty sigma(t):

    intersection   = any t : d(t) <= margin
    ttc            = first t with d(t) <= margin                     (inf if none)
    ttc_kinematic  = d(0) / closing_speed                             (inf if not closing)
    p(t)           = band_collision_probability(d, sigma, margin, W)  (1 when intersecting)
    risk_score     = w_type * max_t [ p(t) * exp(-t / tau) ]
    risk_level     = thresholds on risk_score, escalated to HIGH by the route TTC.
                     CRITICAL is reserved for the PHYSICAL view: it means the
                     vehicle, at its current speed, overlaps the object within
                     ttc_critical_s. A route-view score alone never yields
                     CRITICAL, so a stopped vehicle facing an obstacle is HIGH,
                     not an emergency.

A `RiskSummary` aggregates the worst object and identifies a *lead object*
(traffic ahead in the ego's lateral band) for the FOLLOW state.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from autonomy.core.config import BehaviorConfig, RiskConfig
from autonomy.core.geometry import box_sequence_distance
from autonomy.core.interfaces import RoadModel
from autonomy.core.probability import band_collision_probability
from autonomy.core.types import (ObjectPrediction, ObjectState, RiskAssessment, RiskLevel,
                                 RiskSummary, Trajectory, VehicleParameters, VehicleState)


def kinematic_ttc(distance: float, closing_speed: float) -> float:
    """Classical TTC: current gap over closing speed. inf when not closing."""
    if closing_speed <= 1e-6:
        return math.inf
    return max(distance, 0.0) / closing_speed


def first_overlap_time(rel_times: np.ndarray, distances: np.ndarray, margin: float,
                       grace_s: float = 0.0, closing_eps: float = 0.05) -> float:
    """Earliest time the footprints come within `margin` while CLOSING, or truly overlap.

    Standing (or driving) alongside an object at a constant sub-margin distance is
    not a time-to-collision; only a distance that shrinks below the margin relative
    to now, or an actual overlap, counts. Inside the grace window only overlap counts.
    """
    d0 = float(distances[0])
    closing = distances < d0 - closing_eps
    hit = (distances <= 0.0) | ((distances <= margin) & closing & (rel_times >= grace_s))
    hits = np.nonzero(hit)[0]
    return float(rel_times[hits[0]]) if hits.size else math.inf


def nominal_motion(ego: VehicleState, times: np.ndarray, road: Optional[RoadModel],
                   speed: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego (x, y, yaw) if it follows the corridor at its current offset at `speed` (default: current)."""
    rel = times - times[0]
    v = max(ego.longitudinal_velocity if speed is None else speed, 0.0)
    if road is None:
        return (ego.x + v * rel * math.cos(ego.yaw), ego.y + v * rel * math.sin(ego.yaw),
                np.full_like(times, ego.yaw))
    s0, d0, _ = road.project(ego.x, ego.y)
    x, y, h = road.to_cartesian(s0 + v * rel, np.full_like(times, d0))
    return x, y, h


def trajectory_motion(traj: Trajectory, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego (x, y, yaw) along a trajectory, extrapolated at constant velocity beyond its end."""
    t = traj.t
    x = np.interp(times, t, traj.x)
    y = np.interp(times, t, traj.y)
    yaw = np.interp(times, t, np.unwrap(traj.yaw))
    beyond = times > t[-1]
    if np.any(beyond):
        extra = np.maximum(times - t[-1], 0.0)
        v_end = float(traj.velocity[-1])
        x = np.where(beyond, traj.x[-1] + v_end * extra * math.cos(traj.yaw[-1]), x)
        y = np.where(beyond, traj.y[-1] + v_end * extra * math.sin(traj.yaw[-1]), y)
    return x, y, yaw


class RiskEngine:
    def __init__(self, config: RiskConfig, behavior_cfg: BehaviorConfig, params: VehicleParameters,
                 desired_speed: Optional[float] = None):
        self.cfg = config
        self.bcfg = behavior_cfg
        self.params = params
        self.desired_speed = desired_speed      # None -> use current speed for the route view

    # ------------------------------------------------------------------ #
    def evaluate(self, ego: VehicleState, ego_trajectory: Optional[Trajectory],
                 objects: list[ObjectState], predictions: list[ObjectPrediction],
                 road: Optional[RoadModel] = None) -> RiskSummary:
        if not predictions:
            return RiskSummary.empty()
        times = predictions[0].times
        rel = times - times[0]
        ego_vx = ego.longitudinal_velocity * math.cos(ego.yaw)
        ego_vy = ego.longitudinal_velocity * math.sin(ego.yaw)

        route_speed = self.desired_speed if self.desired_speed is not None else ego.longitudinal_velocity
        route_speed = max(route_speed, ego.longitudinal_velocity)
        nx, ny, nyaw = nominal_motion(ego, times, road, route_speed)
        fx, fy = self._footprint_centres(nx, ny, nyaw)
        assessments = [self._assess(pred, rel, fx, fy, nyaw, ego_vx, ego_vy) for pred in predictions]

        # physical TTC: constant velocity along the CURRENT heading (what happens if control froze now);
        # deliberately not the corridor offset, so a swerve in progress is not mistaken for a head-on threat
        cx, cy, cyaw = nominal_motion(ego, times, None)
        cfx, cfy = self._footprint_centres(cx, cy, cyaw)
        phys = [self._ttc_only(pred, rel, cfx, cfy, cyaw) for pred in predictions]
        physical_ttc = min(phys)
        for a, t_phys in zip(assessments, phys):
            a.ttc_physical = t_phys
            if t_phys < self.cfg.ttc_critical_s:
                a.risk_level = RiskLevel.CRITICAL

        worst = max(assessments, key=lambda a: (a.risk_level.value, a.risk_score))
        summary = RiskSummary(
            assessments=assessments,
            max_level=worst.risk_level,
            max_score=worst.risk_score,
            min_ttc=min(a.ttc for a in assessments),
            min_predicted_distance=min(a.min_predicted_distance for a in assessments),
            worst_object_id=worst.object_id,
            any_intersection=any(a.trajectory_intersection for a in assessments),
            min_ttc_current_speed=physical_ttc,
        )

        if ego_trajectory is not None and len(ego_trajectory) >= 2:
            px, py, pyaw = trajectory_motion(ego_trajectory, times)
            pfx, pfy = self._footprint_centres(px, py, pyaw)
            plan = [self._assess(pred, rel, pfx, pfy, pyaw, ego_vx, ego_vy) for pred in predictions]
            pw = max(plan, key=lambda a: (a.risk_level.value, a.risk_score))
            summary.plan_min_ttc = min(a.ttc for a in plan)
            summary.plan_max_score = pw.risk_score
            summary.plan_max_level = pw.risk_level
            summary.plan_any_intersection = any(a.trajectory_intersection for a in plan)
            summary.plan_min_predicted_distance = min(a.min_predicted_distance for a in plan)
        else:
            summary.plan_min_ttc = summary.min_ttc
            summary.plan_max_score = summary.max_score
            summary.plan_max_level = summary.max_level
            summary.plan_any_intersection = summary.any_intersection
            summary.plan_min_predicted_distance = summary.min_predicted_distance

        if road is not None:
            self._find_lead(summary, ego, objects, road)
        others = [a for a in assessments if a.object_id != summary.lead_object_id]
        if summary.lead_object_id is None or not others:
            summary.non_lead_max_level = summary.max_level if summary.lead_object_id is None else RiskLevel.NONE
            summary.non_lead_max_score = summary.max_score if summary.lead_object_id is None else 0.0
            summary.non_lead_any_intersection = summary.any_intersection if summary.lead_object_id is None else False
        else:
            w = max(others, key=lambda a: (a.risk_level.value, a.risk_score))
            summary.non_lead_max_level = w.risk_level
            summary.non_lead_max_score = w.risk_score
            summary.non_lead_any_intersection = any(a.trajectory_intersection for a in others)
        return summary

    # ------------------------------------------------------------------ #
    def _footprint_centres(self, x: np.ndarray, y: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        off = self.params.footprint_center_offset
        return x + off * np.cos(yaw), y + off * np.sin(yaw)

    def _ttc_only(self, pred: ObjectPrediction, rel: np.ndarray,
                  fx: np.ndarray, fy: np.ndarray, eyaw: np.ndarray) -> float:
        p = self.params
        dist = box_sequence_distance(fx, fy, eyaw, p.length, p.width,
                                     pred.x, pred.y, pred.heading, pred.length, pred.width, exact_within=math.inf)
        margin = self.cfg.safety_margin_m + min(self.cfg.uncertainty_margin_gain * pred.meas_sigma, self.cfg.uncertainty_margin_max_m)
        # Being inside the margin is not by itself an emergency: while squeezing past a parked obstacle the ego
        # sits inside it for the whole pass, and braking does not widen the gap. Only a distance that is still
        # shrinking (or a real overlap) can raise the physical view to CRITICAL, which is the same rule the
        # collision checker uses. Without it the ego panic-brakes repeatedly alongside anything it passes close.
        closing = (dist < dist[0] - self.cfg.closing_epsilon_m) | (dist <= 0.0)
        return first_overlap_time(rel, np.where(closing, dist, math.inf), margin, self.cfg.margin_grace_s)

    def _assess(self, pred: ObjectPrediction, rel: np.ndarray,
                fx: np.ndarray, fy: np.ndarray, eyaw: np.ndarray,
                ego_vx: float, ego_vy: float) -> RiskAssessment:
        p = self.params
        dist = box_sequence_distance(fx, fy, eyaw, p.length, p.width,
                                     pred.x, pred.y, pred.heading, pred.length, pred.width, exact_within=math.inf)
        margin = self.cfg.safety_margin_m + min(self.cfg.uncertainty_margin_gain * pred.meas_sigma, self.cfg.uncertainty_margin_max_m)
        sigma = pred.sigma_toward(fx, fy)
        band = p.width + max(pred.width, pred.length) + 2.0 * margin
        prob = band_collision_probability(dist, sigma, margin, band)
        discounted = prob * np.exp(-rel / self.cfg.time_constant_s)
        k_min = int(np.argmin(dist))
        ttc = first_overlap_time(rel, dist, margin, self.cfg.margin_grace_s)

        vx = float(pred.vx[0]); vy = float(pred.vy[0])
        rvx, rvy = vx - ego_vx, vy - ego_vy
        rx, ry = float(pred.x[0] - fx[0]), float(pred.y[0] - fy[0])
        rnorm = math.hypot(rx, ry)
        closing = -(rx * rvx + ry * rvy) / rnorm if rnorm > 1e-9 else 0.0

        score = float(pred.risk_weight * np.max(discounted))
        return RiskAssessment(
            object_id=pred.object_id, object_type=pred.object_type,
            distance=float(dist[0]),
            relative_speed=math.hypot(rvx, rvy),
            closing_speed=closing,
            ttc=ttc,
            ttc_kinematic=kinematic_ttc(float(dist[0]), closing),
            min_predicted_distance=float(dist[k_min]),
            time_of_min_distance=float(rel[k_min]),
            trajectory_intersection=math.isfinite(ttc),
            collision_probability=float(np.max(prob)),
            risk_score=score,
            risk_level=self._level(score, ttc),
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
        if level == RiskLevel.CRITICAL:
            level = RiskLevel.HIGH                     # CRITICAL is reserved for the physical view
        if ttc < self.cfg.ttc_high_s and level.value < RiskLevel.HIGH.value:
            level = RiskLevel.HIGH
        return level

    def _find_lead(self, summary: RiskSummary, ego: VehicleState,
                   objects: list[ObjectState], road: RoadModel) -> None:
        """A lead object travels along the corridor ahead of the ego, inside its lateral band."""
        s_ego, d_ego, _ = road.project(ego.x, ego.y)
        best_gap = math.inf
        for o in objects:
            s_o, d_o, h_o = road.project(o.x, o.y)
            along = s_o - s_ego
            if along <= 0 or along > self.bcfg.follow_max_range_m:
                continue
            lateral_overlap = abs(d_o - d_ego) < 0.5 * (self.params.width + o.width)
            if not lateral_overlap and abs(d_o - d_ego) > self.bcfg.follow_lateral_window_m:
                continue
            aligned = math.cos(o.heading - h_o) > 0.5
            moving_along = o.speed > 0.5 and aligned
            if not moving_along and not (lateral_overlap and aligned):
                continue
            gap = along - 0.5 * (self.params.length + o.length)
            if gap < best_gap:
                best_gap = gap
                summary.lead_object_id = o.id
                summary.lead_gap = gap
                summary.lead_speed = o.speed * math.cos(o.heading - h_o)
