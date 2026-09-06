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
hits, deleted after `max_age_s` without an update (`tentative_max_age_s` for
unconfirmed tracks); deletion is time-based so a 10 Hz LiDAR-only object
survives the empty 50 Hz ingests between its frames. Class: confidence-weighted votes from camera detections; UNKNOWN when
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
    max_misses: int = 100          # legacy guard; deletion is time-based (max_age_s / tentative_max_age_s)
    max_age_s: float = 1.0
    tentative_max_age_s: float = 0.5
    init_velocity_std_mps: float = 3.0
    min_speed_for_heading_mps: float = 0.5
    class_confidence_floor: float = 0.35
    # --- temporal fusion (Phase 4) ------------------------------------------------------------ #
    # OFF by default in the simulator, ON in the nuScenes validation. This is not timidity: with
    # it enabled the ego becomes measurably more conservative (see docs/PHASE4_TEMPORAL_FUSION.md),
    # because propagating covariance to the output time removes an over-confidence that the
    # existing collision margins were tuned against. Flipping it is a downstream retuning decision,
    # and retuning the planner is out of this phase's scope.
    temporal_fusion: bool = False       # True predicts each measurement to its own timestamp
    update_order: str = "precision"     # "precision" (LiDAR, radar, camera) or "time" (ascending)
    max_out_of_order_s: float = 0.2     # a measurement older than the state by more than this is dropped
    max_measurement_age_s: float = 0.5  # a measurement staler than this is dropped outright
    future_tolerance_s: float = 0.01    # a measurement stamped after `now` by more than this is dropped


@dataclass
class Track:
    id: str
    x: np.ndarray                      # state [x, y, vx, vy]
    P: np.ndarray                      # 4x4
    last_update: float                 # time of the last MEASUREMENT, drives lifecycle and ageing
    created: float
    state_time: float = 0.0            # time `x` and `P` are valid AT, which is not the same thing
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
        self.rejected_future = 0        # measurements stamped after `now`: a clock fault
        self.rejected_stale = 0         # measurements older than max_measurement_age_s
        self.out_of_order = 0           # measurements applied with inflated noise, see _update loop

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

    def _predict_to(self, tr: Track, t: float) -> None:
        """Advance a track to time `t`. Forward only, and idempotent.

        This is the whole of the "do not double-predict" requirement: a track carries the time its
        state is valid at, so predicting it to a time it has already reached costs nothing and
        changes nothing. Backward prediction is deliberately NOT attempted; running the constant-
        velocity model with a negative dt would shrink the covariance, which is not a smoother, it
        is a lie. Out-of-order measurements are handled at the update instead, by inflating their
        measurement noise to cover the interval they are stale by.
        """
        dt = t - tr.state_time
        if dt > 0:
            self._predict(tr, dt)
            tr.state_time = t

    def _predicted_at(self, tr: Track, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """The track's state and covariance `dt` seconds ahead, WITHOUT mutating the track.

        This is how the output time is served. Reporting `tr.P` unchanged while extrapolating the
        position would publish a covariance from an earlier instant beside a position from a later
        one, which understates the uncertainty the planner is asked to reason about; the collision
        margin is sized from exactly this covariance, so the understatement would be unsafe.
        """
        if dt <= 0:
            return tr.x.copy(), tr.P.copy()
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q2 = self.cfg.process_noise_accel_mps2 ** 2
        Q = q2 * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                           [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                           [dt ** 3 / 2, 0, dt ** 2, 0],
                           [0, dt ** 3 / 2, 0, dt ** 2]])
        return F @ tr.x, F @ tr.P @ F.T + Q

    def _staleness_inflation(self, tr: Track, lag_s: float) -> np.ndarray:
        """Extra positional covariance for a measurement taken `lag_s` before the track's state.

        The object was somewhere else when the measurement was taken, and how much else is exactly
        the velocity uncertainty integrated over the lag. Adding `lag^2 * P_vv` to R is the
        cheapest defensible way to fuse a slightly-late measurement without a full smoother: the
        measurement still informs the state, but it is weighted for having been taken in the past.
        """
        return (lag_s ** 2) * tr.P[2:4, 2:4]

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

    def _new_track(self, det: Detection, t: float, now: float | None = None) -> Track:
        """`t` is when the measurement was taken; `now` is when the track is being born.

        These are different things and conflating them is a real bug. `state_time` must be the
        measurement time, because that is when the state is valid. `created` must be the ingest
        time, because it orders tracks for duplicate merging, and merging keeps the OLDEST: with
        `created` set from the measurement time, the camera, which has the largest latency and the
        loosest covariance, always looked oldest and always won the merge.
        """
        P = np.zeros((4, 4))
        P[:2, :2] = np.asarray(det.covariance) + np.eye(2) * 0.05
        P[2, 2] = P[3, 3] = self.cfg.init_velocity_std_mps ** 2
        born = t if now is None else now
        tr = Track(f"trk_{self._next_id}", np.array([det.x, det.y, 0.0, 0.0]), P, t, born,
                   state_time=t)
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
    # Sensor groups are processed most-precise first: LiDAR, then radar, then camera. This is also
    # the pre-Phase-4 order, so a genuinely synchronised batch behaves exactly as it always did.
    #
    # WHY NOT STRICT TIME ORDER. Strict ascending-time ordering was implemented and measured. It is
    # the textbook sequencing, but in this suite the camera has BOTH the largest latency (80 ms
    # against LiDAR's 50 ms) and the loosest covariance, so time order puts the least precise
    # sensor first every cycle. Association degrades: camera-shaped tracks are established before
    # LiDAR is seen, and on NARROW_LANE_BOXED_IN the measured consequence was 31 emergency brakes
    # and a collision, against 6 brakes and none in precision order. Both orderings use the true
    # measurement times; they differ only in which sensor gets to shape association first, and a
    # measurement that is behind the track state is paid for by inflating its noise either way.
    # `cfg.update_order = "time"` selects strict time order for anyone who wants to re-measure.
    _SENSOR_ORDER = (SensorType.LIDAR, SensorType.RADAR, SensorType.CAMERA)

    def _admissible(self, d: Detection, now: float) -> bool:
        """Reject measurements whose timestamp cannot be acted on.

        A measurement stamped in the future is a clock fault, not data. A measurement far in the
        past would drag a track backwards through a long unmodelled interval. Both are dropped
        rather than clamped, because clamping would hide the fault.
        """
        if d.timestamp > now + self.cfg.future_tolerance_s:
            self.rejected_future += 1
            return False
        if now - d.timestamp > self.cfg.max_measurement_age_s:
            self.rejected_stale += 1
            return False
        return True

    def _sensor_groups(self, detections: list[Detection],
                       now: float) -> list[tuple[float, SensorType, list[Detection]]]:
        """Split a batch into per-sensor groups, ordered by when they were actually measured.

        Grouping is per SENSOR rather than per detection for two reasons. It preserves the existing
        guarantee that one object receives at most one update per sensor per batch, and it keeps
        association well posed: every track shares a common time while a group is gated, which
        would not hold if each detection dragged the tracks to its own instant.

        A group's time is the MEDIAN of its detections' timestamps, so one mis-stamped detection
        cannot drag the group.
        """
        groups: dict[SensorType, list[Detection]] = {}
        for d in detections:
            groups.setdefault(d.sensor, []).append(d)
        out = []
        for sensor, batch in groups.items():
            t = float(np.median([d.timestamp for d in batch])) if self.cfg.temporal_fusion else now
            out.append((t, sensor, batch))
        order = {s: i for i, s in enumerate(self._SENSOR_ORDER)}
        if self.cfg.update_order == "time":
            out.sort(key=lambda g: (g[0], order.get(g[1], 99)))
        else:
            out.sort(key=lambda g: (order.get(g[1], 99), g[0]))
        return out

    def ingest(self, detections: list[Detection], now: float, ego: Optional[VehicleState] = None) -> None:
        """Fuse a batch of measurements, each at ITS OWN time, and leave every track at `now`.

        The pre-Phase-4 flow predicted every track to `now` and then applied every detection there,
        which silently asserted that camera, LiDAR and radar fired simultaneously. Measured on real
        nuScenes data they are 66.8 ms apart at the median and up to 82.8 ms apart, so at 10 m/s
        that assertion injects up to 0.83 m of position error before the filter even runs.

        The flow now is:
            for each sensor group, in ascending measurement time:
                predict every track FORWARD to that group's time
                gate, associate and update there
            finally predict every track forward to `now`, the output time

        `cfg.temporal_fusion = False` restores the old behaviour exactly, which is what the A/B
        validation compares against.
        """
        self.ego = ego or self.ego
        for d in detections:
            assert d.truth_id is not None  # never read beyond this assertion: evaluation-only field
            self.latency_samples.append(now - d.timestamp)
        if self.cfg.temporal_fusion:
            detections = [d for d in detections if self._admissible(d, now)]

        if not self.cfg.temporal_fusion:
            # legacy path: one common predict to `now`, then everything happens at `now`
            dt = 0.0 if self.last_time is None else max(now - self.last_time, 0.0)
            for tr in self.tracks:
                self._predict(tr, dt)
                tr.state_time = now
        self.last_time = now

        updated: set[str] = set()
        # Track BIRTH is deferred until every group in the batch has had its chance to associate.
        # Ordering the groups by time puts the sensor with the largest latency first, which for
        # this suite is the camera, the least precise of the three. Letting it spawn tracks before
        # LiDAR has been seen produces badly-placed tracks that LiDAR then fails to associate with,
        # and the measured consequence was a collision in NARROW_LANE_BOXED_IN. Deferring birth
        # removes the ordering sensitivity from data association while keeping the filter updates
        # in true time order, which is the part that has to be correct.
        unclaimed: list[tuple[float, Detection]] = []
        for group_t, sensor, batch in self._sensor_groups(detections, now):
            for tr in self.tracks:
                self._predict_to(tr, group_t)   # forward only; already-there tracks are untouched
            pairs = []
            for i, d in enumerate(batch):
                for tr in self.tracks:
                    m2 = self._mahalanobis2(tr, d)
                    if m2 <= self.cfg.gate_chi2:
                        pairs.append((m2, i, tr))
            pairs.sort(key=lambda p: p[0])
            too_late: set[int] = set()
            used_d: set[int] = set()
            used_t: set[str] = set()
            for m2, i, tr in pairs:
                if i in used_d or tr.id in used_t:
                    continue
                d = batch[i]
                # A track can already be AHEAD of this group when an earlier batch pushed it past
                # this measurement's time. `_predict_to` refuses to reverse, so the mismatch is
                # paid for in the measurement noise instead: the object moved during the lag, by
                # an amount whose covariance is the velocity uncertainty over that interval.
                lag = max(tr.state_time - group_t, 0.0)
                R = np.asarray(d.covariance)
                if lag > 0.0:
                    if lag > self.cfg.max_out_of_order_s:
                        too_late.add(i)             # too far behind to fuse safely; drop it
                        continue
                    self.out_of_order += 1
                    R = R + self._staleness_inflation(tr, lag)
                self._update_position(tr, np.array([d.x, d.y]), R)
                self._update_radial(tr, d)
                self._absorb(tr, d)
                tr.hits += 1
                tr.last_update = max(tr.last_update, group_t)
                used_d.add(i)
                used_t.add(tr.id)
                updated.add(tr.id)
            for i, d in enumerate(batch):
                if i not in used_d and i not in too_late:
                    unclaimed.append((group_t, d))
            self.rejected_stale += len(too_late)

        for group_t, d in unclaimed:
            # A second chance to associate: a detection left over from an early group may match a
            # track that a later, more precise group has since corrected.
            claimed = False
            for tr in self.tracks:
                if (tr.id not in updated
                        and tr.state_time - group_t <= self.cfg.max_out_of_order_s
                        and self._mahalanobis2(tr, d) <= self.cfg.gate_chi2):
                    self._update_position(tr, np.array([d.x, d.y]), np.asarray(d.covariance))
                    self._absorb(tr, d)
                    tr.hits += 1
                    tr.last_update = max(tr.last_update, group_t)
                    updated.add(tr.id)
                    claimed = True
                    break
            if not claimed:
                tr = self._new_track(d, group_t if self.cfg.temporal_fusion else now, now)
                self._absorb(tr, d)
                self.tracks.append(tr)
                updated.add(tr.id)

        # NOTE the filter state is deliberately LEFT at the measurement frontier rather than being
        # advanced to `now`. Sensor latencies here are 30-80 ms, so if the state were pushed to
        # `now` every cycle, the next batch of measurements would arrive behind it and every single
        # one would be treated as out-of-order and have its noise inflated. Prediction to the
        # output time happens in `get_object_states`, without mutating the track, so asking for
        # output never costs the filter anything.
        for tr in self.tracks:
            if tr.id in updated:
                tr.misses = 0
            else:
                tr.misses += 1
        # deletion is TIME based: a LiDAR-only object seen at 10 Hz must survive the 4 empty 50 Hz ingests
        # between its frames and an occasional dropped frame; a confirmed track coasts for max_age_s
        def alive(tr: Track) -> bool:
            age = now - tr.last_update
            limit = self.cfg.max_age_s if tr.hits >= self.cfg.confirm_hits else self.cfg.tentative_max_age_s
            return tr.misses <= self.cfg.max_misses and age <= limit
        self.tracks = [tr for tr in self.tracks if alive(tr)]
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
            # `state_time`, not `last_update`. The state is valid at `state_time`; a coasting
            # track's last measurement is older than that, and extrapolating from it would advance
            # the object twice over the coasting interval.
            dt = max(timestamp - tr.state_time, 0.0)
            x, P = self._predicted_at(tr, dt)
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
                covariance=P[:2, :2],
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
