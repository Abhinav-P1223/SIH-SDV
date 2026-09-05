"""Simulated camera, LiDAR and radar mounted on the ego.

These are *sensor models*, not renderers: each converts ground-truth agents
into noisy `Detection`s in the world frame, subject to field of view, range,
occlusion (line of sight blocked by another agent's footprint), dropout,
update rate and latency. All randomness comes from a seeded generator.

    Camera : bearing (sigma_b) + range (sigma_r = frac*range + min) -> x, y with
             range-dominated covariance; object class with a confusion probability.
    LiDAR  : x, y with small isotropic noise; extent (length, width) and heading.
    Radar  : range + bearing (coarse) -> x, y with bearing-dominated covariance;
             radial speed relative to the sensor. Small objects may be missed.

What the autonomy stack receives is ONLY the Detection list; agent identity,
type and true state are not passed on (truth_id is carried for evaluation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autonomy.core.geometry import box_corners, wrap_angle
from autonomy.core.types import Detection, ObjectType, SensorType, VehicleState
from simulation.agents.agent import Agent


@dataclass
class SensorConfig:
    enabled: bool = True
    mount_x_m: float = 0.0
    rate_hz: float = 10.0
    latency_s: float = 0.0
    fov_deg: float = 360.0
    range_m: float = 80.0
    dropout_prob: float = 0.0
    occlusion: bool = True
    # camera
    bearing_std_deg: float = 1.0
    range_std_frac: float = 0.04
    range_std_min_m: float = 0.3
    class_accuracy: float = 0.9
    # lidar
    position_std_m: float = 0.1
    extent_std_m: float = 0.2
    heading_std_deg: float = 6.0
    # radar
    range_std_m: float = 0.3
    radial_speed_std_mps: float = 0.2
    min_rcs_size_m: float = 0.5
    small_object_dropout_prob: float = 0.35


def _segment_hits_box(px, py, qx, qy, corners: np.ndarray) -> bool:
    """Does segment p->q intersect the convex quad `corners` (4,2)?"""
    def cross(ax, ay, bx, by):
        return ax * by - ay * bx
    for k in range(4):
        ax, ay = corners[k]
        bx, by = corners[(k + 1) % 4]
        d1 = cross(qx - px, qy - py, ax - px, ay - py)
        d2 = cross(qx - px, qy - py, bx - px, by - py)
        d3 = cross(bx - ax, by - ay, px - ax, py - ay)
        d4 = cross(bx - ax, by - ay, qx - ax, qy - ay)
        if (d1 * d2 < 0) and (d3 * d4 < 0):
            return True
    return False


class SensorModel:
    sensor_type: SensorType

    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.next_sample_t = 0.0
        self.dropped = 0
        self.emitted = 0

    def position(self, ego: VehicleState) -> tuple[float, float]:
        return (ego.x + self.cfg.mount_x_m * math.cos(ego.yaw),
                ego.y + self.cfg.mount_x_m * math.sin(ego.yaw))

    def due(self, t: float) -> bool:
        if not self.cfg.enabled or t + 1e-9 < self.next_sample_t:
            return False
        self.next_sample_t = t + 1.0 / self.cfg.rate_hz
        return True

    def visible(self, sx: float, sy: float, ego: VehicleState, agent: Agent, others: list[Agent]) -> bool:
        dx, dy = agent.x - sx, agent.y - sy
        rng = math.hypot(dx, dy)
        if rng > self.cfg.range_m or rng < 0.5:
            return False
        bearing = float(wrap_angle(math.atan2(dy, dx) - ego.yaw))
        if abs(bearing) > math.radians(self.cfg.fov_deg) / 2.0:
            return False
        if self.cfg.occlusion:
            for o in others:
                if o is agent:
                    continue
                if math.hypot(o.x - sx, o.y - sy) >= rng:
                    continue                      # farther than the target cannot occlude it
                corners = box_corners(np.array([o.x]), np.array([o.y]), np.array([o.heading]), o.length, o.width)[0]
                if _segment_hits_box(sx, sy, agent.x, agent.y, corners):
                    return False
        return True

    def sense(self, agents: list[Agent], ego: VehicleState, t: float) -> list[Detection]:
        if not self.due(t):
            return []
        sx, sy = self.position(ego)
        out: list[Detection] = []
        for a in agents:
            if not self.visible(sx, sy, ego, a, agents):
                continue
            if self.rng.random() < self.cfg.dropout_prob:
                self.dropped += 1
                continue
            det = self.measure(a, sx, sy, ego, t)
            if det is not None:
                det.sensor_x, det.sensor_y, det.truth_id = sx, sy, a.id
                out.append(det)
                self.emitted += 1
        return out

    def measure(self, a: Agent, sx: float, sy: float, ego: VehicleState, t: float):  # pragma: no cover
        raise NotImplementedError


def _polar_to_cov(rng: float, bearing_w: float, sr: float, sb: float) -> np.ndarray:
    """Covariance of (x, y) from range/bearing std-devs, rotated into the world frame."""
    c, s = math.cos(bearing_w), math.sin(bearing_w)
    R = np.array([[c, -s], [s, c]])
    local = np.diag([sr ** 2, (rng * sb) ** 2])
    return R @ local @ R.T


class CameraModel(SensorModel):
    sensor_type = SensorType.CAMERA

    def measure(self, a, sx, sy, ego, t):
        dx, dy = a.x - sx, a.y - sy
        rng = math.hypot(dx, dy)
        bearing_w = math.atan2(dy, dx)
        sr = self.cfg.range_std_frac * rng + self.cfg.range_std_min_m
        sb = math.radians(self.cfg.bearing_std_deg)
        r_m = max(0.5, rng + self.rng.normal(0.0, sr))
        b_m = bearing_w + self.rng.normal(0.0, sb)
        if self.rng.random() < self.cfg.class_accuracy:
            cls, conf = a.object_type, float(np.clip(self.rng.normal(0.85, 0.08), 0.4, 0.99))
        else:
            choices = [o for o in ObjectType if o not in (a.object_type, ObjectType.UNKNOWN)]
            cls, conf = choices[int(self.rng.integers(len(choices)))], float(np.clip(self.rng.normal(0.55, 0.1), 0.3, 0.8))
        return Detection(SensorType.CAMERA, t, sx + r_m * math.cos(b_m), sy + r_m * math.sin(b_m),
                         _polar_to_cov(rng, bearing_w, sr, sb), object_type=cls, class_confidence=conf)


class LidarModel(SensorModel):
    sensor_type = SensorType.LIDAR

    def measure(self, a, sx, sy, ego, t):
        sp = self.cfg.position_std_m
        se = self.cfg.extent_std_m
        return Detection(SensorType.LIDAR, t, a.x + self.rng.normal(0, sp), a.y + self.rng.normal(0, sp),
                         np.eye(2) * sp ** 2,
                         length=max(0.3, a.length + self.rng.normal(0, se)),
                         width=max(0.3, a.width + self.rng.normal(0, se)),
                         heading=float(wrap_angle(a.heading + self.rng.normal(0, math.radians(self.cfg.heading_std_deg)))))


class RadarModel(SensorModel):
    sensor_type = SensorType.RADAR

    def measure(self, a, sx, sy, ego, t):
        if min(a.length, a.width) < self.cfg.min_rcs_size_m and self.rng.random() < self.cfg.small_object_dropout_prob:
            self.dropped += 1
            return None
        dx, dy = a.x - sx, a.y - sy
        rng = math.hypot(dx, dy)
        bearing_w = math.atan2(dy, dx)
        sr, sb = self.cfg.range_std_m, math.radians(self.cfg.bearing_std_deg)
        r_m = max(0.5, rng + self.rng.normal(0.0, sr))
        b_m = bearing_w + self.rng.normal(0.0, sb)
        # radial speed of the object relative to the (moving) sensor, positive = receding
        ux, uy = dx / rng, dy / rng
        ego_vx, ego_vy = ego.longitudinal_velocity * math.cos(ego.yaw), ego.longitudinal_velocity * math.sin(ego.yaw)
        v_rad = (a.vx - ego_vx) * ux + (a.vy - ego_vy) * uy + self.rng.normal(0.0, self.cfg.radial_speed_std_mps)
        return Detection(SensorType.RADAR, t, sx + r_m * math.cos(b_m), sy + r_m * math.sin(b_m),
                         _polar_to_cov(rng, bearing_w, sr, sb), radial_speed=v_rad,
                         radial_speed_std=self.cfg.radial_speed_std_mps)


@dataclass
class SensorSuite:
    """All enabled sensors plus the latency queue. `sense()` returns detections whose arrival time has come."""
    sensors: list[SensorModel]
    _pending: list[tuple[float, Detection]] = field(default_factory=list)

    @staticmethod
    def from_config(data: dict[str, Any], seed: int) -> "SensorSuite":
        rng = np.random.default_rng(seed + 7919)
        sensors: list[SensorModel] = []
        for name, cls in (("camera", CameraModel), ("lidar", LidarModel), ("radar", RadarModel)):
            raw = dict(data.get(name, {}))
            if not raw.get("enabled", True):
                continue
            sensors.append(cls(SensorConfig(**raw), rng))
        return SensorSuite(sensors)

    def sense(self, agents: list[Agent], ego: VehicleState, t: float) -> list[Detection]:
        for s in self.sensors:
            for d in s.sense(agents, ego, t):
                self._pending.append((d.timestamp + s.cfg.latency_s, d))
        ready = [d for arrival, d in self._pending if arrival <= t + 1e-9]
        self._pending = [(arr, d) for arr, d in self._pending if arr > t + 1e-9]
        return ready

    def stats(self) -> dict[str, dict[str, int]]:
        return {s.sensor_type.value: {"emitted": s.emitted, "dropped": s.dropped} for s in self.sensors}
