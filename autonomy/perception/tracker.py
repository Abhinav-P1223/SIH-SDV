"""Multi-object tracking and sensor fusion.

Implements `ObjectStateProvider`, so the planner cannot tell whether objects
come from here or from the simulator's ground truth. Input is the list of
`Detection`s from any number of sensors; output is the list of confirmed
`ObjectState` tracks with real covariance, confidence and persistent ids.

Per track: linear Kalman filter, state [x, y, vx, vy], constant-velocity
process model with acceleration process noise q:

    F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
    Q = q^2 * [[dt^4/4, 0, dt^3/2, 0], [0, dt^4/4, 0, dt^3/2], [dt^3/2, 0, dt^2, 0], [0, dt^3/2, 0, dt^2]]

Measurement models:
    position (camera / LiDAR / radar x,y) : H = [[1,0,0,0],[0,1,0,0]], R = detection covariance
    radial speed (radar)                  : h(x) = ((vx - v_ego) ux + (vy - v_ego) uy), linearised in vx, vy
                                            with u the unit vector sensor -> track (EKF update)

Association: gated global nearest neighbour on the Mahalanobis distance of
the position innovation (gate chi2, 2 dof), greedy per sensor batch.
Lifecycle: tentative on first detection, confirmed after `confirm_hits`
hits, deleted after `max_misses` consecutive misses or `max_age_s` without an
update. Class: confidence-weighted votes from camera detections; UNKNOWN when
no camera has seen the object or the winning vote is below the floor.
Dimensions: running mean of LiDAR extents (profile defaults otherwise).
Heading: from velocity when moving, else from LiDAR, else last value.

Ablations that must change the output (and are tested): no camera -> all
types UNKNOWN; no radar -> slower velocity convergence and larger velocity
covariance; no LiDAR -> larger positional covariance and profile dimensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from autonomy.core.config import ObjectProfiles
from autonomy.core.interfaces import ObjectStateProvider
from autonomy.core.types import Detection, ObjectState, ObjectType, SensorType, VehicleState


@dataclass
class TrackerConfig:
    process_noise_accel_mps2: float = 1.5
    gate_chi2: float = 9.21
    confirm_hits: int = 3
    max_misses: int = 6
    max_age_s: float = 1.0
    init_velocity_std_mps: float = 3.0
    min_speed_for_heading_mps: float = 0.5
    class_confidence_floor: float = 0.35


@dataclass
class Track:
    id: str
    x: np.ndarray                      # state [x, y, vx, vy]
    P: np.ndarray                      # 4x4
    last_update: float
    created: float
    hits: int = 1
    misses: int = 0
    class_votes: dict[ObjectType, float] = field(default_factory=dict)
    length_sum: float = 0.0
    width_sum: float = 0.0
    extent_n: int = 0
    lidar_heading: Optional[float] = None
    last_heading: float = 0.0
    sensors_seen: set = field(default_factory=set)

    @property
    def confirmed_type(self) -> tuple[ObjectType, float]:
        if not self.class_votes:
            return ObjectType.UNKNOWN, 0.0
        total = sum(self.class_votes.values())
        best = max(self.class_votes, key=self.class_votes.get)
        return best, self.class_votes[best] / total if total > 0 else 0.0


class SensorFusionTracker(ObjectStateProvider):
    def __init__(self, cfg: TrackerConfig, profiles: ObjectProfiles):
        self.cfg = cfg
        self.profiles = profiles
        self.tracks: list[Track] = []
        self._next_id = 1
        self.last_time: Optional[float] = None
        self.latency_samples: list[float] = []
        self.ego: Optional[VehicleState] = None

    # ------------------------------------------------------------------ #
    def _predict(self, tr: Track, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q2 = self.cfg.process_noise_accel_mps2 ** 2
        Q = q2 * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                           [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                           [dt ** 3 / 2, 0, dt ** 2, 0],
                           [0, dt ** 3 / 2, 0, dt ** 2]])
        tr.x = F @ tr.x
        tr.P = F @ tr.P @ F.T + Q

    def _update_position(self, tr: Track, z: np.ndarray, R: np.ndarray) -> None:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - H @ tr.x)
        tr.P = (np.eye(4) - K @ H) @ tr.P

    def _update_radial(self, tr: Track, det: Detection) -> None:
        if det.radial_speed is None or self.ego is None:
            return
        dx, dy = tr.x[0] - det.sensor_x, tr.x[1] - det.sensor_y
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return
        ux, uy = dx / n, dy / n
        ego_vx = self.ego.longitudinal_velocity * math.cos(self.ego.yaw)
        ego_vy = self.ego.longitudinal_velocity * math.sin(self.ego.yaw)
        h = (tr.x[2] - ego_vx) * ux + (tr.x[3] - ego_vy) * uy
        H = np.array([[0.0, 0.0, ux, uy]])
        R = np.array([[max(det.radial_speed_std, 0.05) ** 2]])
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + (K * (det.radial_speed - h)).ravel()
        tr.P = (np.eye(4) - K @ H) @ tr.P

    def _mahalanobis2(self, tr: Track, det: Detection) -> float:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        S = H @ tr.P @ H.T + np.asarray(det.covariance)
        v = np.array([det.x, det.y]) - H @ tr.x
        return float(v @ np.linalg.solve(S, v))

    def _new_track(self, det: Detection, t: float) -> Track:
        P = np.zeros((4, 4))
        P[:2, :2] = np.asarray(det.covariance) + np.eye(2) * 0.05
        P[2, 2] = P[3, 3] = self.cfg.init_velocity_std_mps ** 2
        tr = Track(f"trk_{self._next_id}", np.array([det.x, det.y, 0.0, 0.0]), P, t, t)
        self._next_id += 1
        return tr

    def _absorb(self, tr: Track, det: Detection) -> None:
        if det.sensor == SensorType.CAMERA and det.object_type is not None:
            tr.class_votes[det.object_type] = tr.class_votes.get(det.object_type, 0.0) + det.class_confidence
        if det.sensor == SensorType.LIDAR and det.length is not None:
            tr.length_sum += det.length
            tr.width_sum += det.width
            tr.extent_n += 1
            tr.lidar_heading = det.heading
        tr.sensors_seen.add(det.sensor.value)

    # ------------------------------------------------------------------ #
    def ingest(self, detections: list[Detection], now: float, ego: Optional[VehicleState] = None) -> None:
        """Predict all tracks to `now`, associate and update with this batch, manage lifecycle."""
        self.ego = ego or self.ego
        dt = 0.0 if self.last_time is None else max(now - self.last_time, 0.0)
        for tr in self.tracks:
            self._predict(tr, dt)          # every track advances by the time since the last ingest
        self.last_time = now
        for d in detections:
            assert d.truth_id is not None  # never read beyond this assertion: evaluation-only field
            self.latency_samples.append(now - d.timestamp)

        # process sensor by sensor so one object gets at most one update per sensor per batch
        updated: set[str] = set()
        for sensor in (SensorType.LIDAR, SensorType.RADAR, SensorType.CAMERA):
            batch = [d for d in detections if d.sensor == sensor]
            if not batch:
                continue
            pairs = []
            for i, d in enumerate(batch):
                for tr in self.tracks:
                    m2 = self._mahalanobis2(tr, d)
                    if m2 <= self.cfg.gate_chi2:
                        pairs.append((m2, i, tr))
            pairs.sort(key=lambda p: p[0])
            used_d: set[int] = set()
            used_t: set[str] = set()
            for m2, i, tr in pairs:
                if i in used_d or tr.id in used_t:
                    continue
                d = batch[i]
                self._update_position(tr, np.array([d.x, d.y]), np.asarray(d.covariance))
                self._update_radial(tr, d)
                self._absorb(tr, d)
                tr.hits += 1
                tr.last_update = now
                used_d.add(i)
                used_t.add(tr.id)
                updated.add(tr.id)
            for i, d in enumerate(batch):
                if i not in used_d:
                    tr = self._new_track(d, now)
                    self._absorb(tr, d)
                    self.tracks.append(tr)
                    updated.add(tr.id)

        for tr in self.tracks:
            if tr.id in updated:
                tr.misses = 0
            else:
                tr.misses += 1
        self.tracks = [tr for tr in self.tracks
                       if tr.misses <= self.cfg.max_misses and now - tr.last_update <= self.cfg.max_age_s]
        self._merge_duplicates()

    def _merge_duplicates(self) -> None:
        """Two tracks explaining the same object (position gate on each other's covariance) are merged:
        the older track keeps its id and absorbs the younger one's hits, votes and extents."""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        keep: list[Track] = []
        for tr in sorted(self.tracks, key=lambda t: t.created):
            merged = False
            for k in keep:
                S = H @ (k.P + tr.P) @ H.T
                v = H @ (tr.x - k.x)
                if float(v @ np.linalg.solve(S, v)) <= self.cfg.gate_chi2:
                    k.hits += tr.hits
                    for c, w in tr.class_votes.items():
                        k.class_votes[c] = k.class_votes.get(c, 0.0) + w
                    k.length_sum += tr.length_sum; k.width_sum += tr.width_sum; k.extent_n += tr.extent_n
                    if k.lidar_heading is None:
                        k.lidar_heading = tr.lidar_heading
                    k.sensors_seen |= tr.sensors_seen
                    merged = True
                    break
            if not merged:
                keep.append(tr)
        self.tracks = keep

    def get_object_states(self, timestamp: float) -> list[ObjectState]:
        out: list[ObjectState] = []
        for tr in self.tracks:
            if tr.hits < self.cfg.confirm_hits:
                continue
            dt = max(timestamp - tr.last_update, 0.0)
            x = tr.x.copy()
            x[0] += x[2] * dt
            x[1] += x[3] * dt
            speed = math.hypot(x[2], x[3])
            if speed >= self.cfg.min_speed_for_heading_mps:
                heading = math.atan2(x[3], x[2])
            elif tr.lidar_heading is not None:
                heading = tr.lidar_heading
            else:
                heading = tr.last_heading
            tr.last_heading = heading
            otype, conf = tr.confirmed_type
            if conf < self.cfg.class_confidence_floor:
                otype = ObjectType.UNKNOWN
            prof = self.profiles.get(otype)
            if tr.extent_n > 0:
                length, width = tr.length_sum / tr.extent_n, tr.width_sum / tr.extent_n
            else:
                length, width = prof.length_m, prof.width_m
            out.append(ObjectState(
                id=tr.id, object_type=otype, timestamp=timestamp,
                x=float(x[0]), y=float(x[1]), vx=float(x[2]), vy=float(x[3]), heading=float(heading),
                length=float(length), width=float(width),
                confidence=float(min(1.0, tr.hits / (tr.hits + tr.misses + 1))),
                covariance=tr.P[:2, :2].copy(),
            ))
        return out

    # ------------------------------------------------------------------ #
    def mean_latency_s(self) -> float:
        return float(np.mean(self.latency_samples)) if self.latency_samples else 0.0

    def velocity_covariance(self, track_id: str) -> Optional[np.ndarray]:
        for tr in self.tracks:
            if tr.id == track_id:
                return tr.P[2:, 2:].copy()
        return None
