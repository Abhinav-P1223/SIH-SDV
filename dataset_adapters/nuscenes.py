"""nuScenes v1.0-mini -> `Detection`: a validation adapter for the existing fusion interface.

WHAT THIS IS
    A translator. It reads nuScenes metadata and emits `autonomy.core.types.Detection` objects
    that the existing `SensorFusionTracker` consumes unmodified. Its purpose is to prove that
    real multimodal observations, with real timestamps and real calibration, satisfy the contract
    our simulated sensors satisfy.

WHAT THIS IS NOT
    A perception system. nuScenes ships images and point clouds, not detections, so the
    observations here are derived from the annotated 3D boxes rather than from a detector. That
    validates the INTERFACE, the timing path and the calibration path. It says nothing about
    detection quality, and no accuracy claim should be made from it.

FRAME CONVENTION
    The nuScenes global (map) frame is used directly as the stack's world frame. Annotations are
    already global, so no transform is invented for them. The sensor's world position comes from
    real calibration: ego_translation + R(ego_rotation) @ sensor_translation, evaluated at that
    sensor's OWN timestamp. That position is what the tracker's radar EKF needs to form the
    line-of-sight direction, so it must be right.

TIMESTAMPS
    Every emitted `Detection` carries the timestamp of the sample_data row it came from, in
    seconds. Sensors inside one keyframe are NOT simultaneous: measured spread on v1.0-mini is a
    median of 70 ms and up to 83 ms. See `SENSOR_SYNC` below and the note on `ingest`.

WHICH SENSOR SEES WHAT
    Visibility is decided with the real calibration, not a hard-coded rule:
      camera  the box centre is projected with the actual 3x3 intrinsic and must land inside the
              image and in front of the lens
      lidar   360 degrees, range limited
      radar   the box must fall inside that unit's horizontal field of view and range
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from autonomy.core.types import Detection, ObjectType, SensorType, VehicleState

# Measured on v1.0-mini by scripts/audit_datasets.py. Recorded here so the mismatch with
# config/sensors.yaml (camera 20 Hz, lidar 10 Hz, radar 20 Hz) is visible in code, not folklore.
SENSOR_SYNC = {
    "camera_hz": 10.0,
    "lidar_hz": 20.1,
    "radar_hz": 13.4,
    "keyframe_spread_median_ms": 70.3,
    "keyframe_spread_max_ms": 82.8,
}

# nuScenes categories -> our ObjectType. Only what genuinely corresponds; anything else is
# UNKNOWN rather than being forced into a class it is not.
CATEGORY_MAP = {
    "human.pedestrian.adult": ObjectType.PEDESTRIAN,
    "human.pedestrian.child": ObjectType.PEDESTRIAN,
    "human.pedestrian.construction_worker": ObjectType.PEDESTRIAN,
    "human.pedestrian.police_officer": ObjectType.PEDESTRIAN,
    "human.pedestrian.personal_mobility": ObjectType.PEDESTRIAN,
    "vehicle.bicycle": ObjectType.BICYCLE,
    "vehicle.motorcycle": ObjectType.MOTORCYCLE,
    "vehicle.car": ObjectType.CAR,
    "vehicle.bus.rigid": ObjectType.BUS,
    "vehicle.bus.bendy": ObjectType.BUS,
    "vehicle.truck": ObjectType.TRUCK,
    "vehicle.trailer": ObjectType.TRUCK,
    "vehicle.construction": ObjectType.TRUCK,
}

LIDAR_RANGE_M = 80.0
RADAR_RANGE_M = 120.0
RADAR_FOV_DEG = 60.0


def quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix from a nuScenes (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_yaw(q) -> float:
    """Heading about +z, which is all our 2-D stack uses."""
    m = quat_to_matrix(q)
    return math.atan2(m[1, 0], m[0, 0])


@dataclass
class SensorObservation:
    """One sensor's view of one annotated object, before it becomes a `Detection`."""
    channel: str
    modality: str
    timestamp_s: float
    sensor_xy: tuple[float, float]
    annotation: dict


class NuScenesMini:
    """Read-only index over the nuScenes metadata tables.

    Deliberately avoids `nuscenes-devkit`: the project has four dependencies and this adapter is
    not a reason to add a fifth. Only the JSON tables are read, never the sensor blobs, so an
    audit run costs megabytes rather than gigabytes.
    """

    TABLES = ("scene", "sample", "sample_data", "sample_annotation", "instance", "category",
              "sensor", "calibrated_sensor", "ego_pose")

    def __init__(self, root: Path | str):
        self.root = Path(root)
        meta = [d for d in self.root.iterdir() if d.is_dir() and d.name.startswith("v1.0")]
        if not meta:
            raise FileNotFoundError(f"no v1.0-* metadata directory under {self.root}")
        self.meta_dir = meta[0]
        for t in self.TABLES:
            path = self.meta_dir / f"{t}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing nuScenes table: {path}")
            setattr(self, t, json.loads(path.read_text(encoding="utf-8")))

        self.by_token = {t: {r["token"]: r for r in getattr(self, t)} for t in self.TABLES}
        self.category_of_instance = {
            i["token"]: self.by_token["category"][i["category_token"]]["name"]
            for i in self.instance
        }
        self.data_by_sample: dict[str, list[dict]] = {}
        for sd in self.sample_data:
            self.data_by_sample.setdefault(sd["sample_token"], []).append(sd)
        self.ann_by_sample: dict[str, list[dict]] = {}
        for a in self.sample_annotation:
            self.ann_by_sample.setdefault(a["sample_token"], []).append(a)

    # ---------------------------------------------------------------- frames -- #
    def channel_of(self, sd: dict) -> tuple[str, str]:
        cal = self.by_token["calibrated_sensor"][sd["calibrated_sensor_token"]]
        s = self.by_token["sensor"][cal["sensor_token"]]
        return s["channel"], s["modality"]

    def sensor_world_pose(self, sd: dict) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation) of the sensor in the global frame at this row's timestamp.

        Uses the real extrinsics: the ego pose recorded for THIS sample_data row composed with
        the sensor's calibrated offset. Nothing is assumed about where a sensor sits.
        """
        ego = self.by_token["ego_pose"][sd["ego_pose_token"]]
        cal = self.by_token["calibrated_sensor"][sd["calibrated_sensor_token"]]
        R_ego = quat_to_matrix(ego["rotation"])
        t_ego = np.asarray(ego["translation"], dtype=float)
        R_cal = quat_to_matrix(cal["rotation"])
        t_cal = np.asarray(cal["translation"], dtype=float)
        return t_ego + R_ego @ t_cal, R_ego @ R_cal

    def ego_state(self, sd: dict) -> VehicleState:
        """The ego as our stack represents it, at this row's timestamp."""
        ego = self.by_token["ego_pose"][sd["ego_pose_token"]]
        t = np.asarray(ego["translation"], dtype=float)
        return VehicleState(timestamp=sd["timestamp"] / 1e6, x=float(t[0]), y=float(t[1]),
                            yaw=quat_to_yaw(ego["rotation"]), longitudinal_velocity=0.0)

    # ------------------------------------------------------------- velocity -- #
    def box_velocity(self, ann: dict) -> Optional[tuple[float, float]]:
        """Global-frame (vx, vy) from the neighbouring boxes of the same instance.

        This is how nuScenes velocity is defined: a finite difference over consecutive
        annotations. Nothing is invented -- when an object has no usable neighbour the caller
        gets None and must leave velocity out rather than guess.
        """
        prev_t, next_t = ann.get("prev"), ann.get("next")
        a0 = self.by_token["sample_annotation"].get(prev_t) if prev_t else None
        a1 = self.by_token["sample_annotation"].get(next_t) if next_t else None
        if a0 is None and a1 is None:
            return None
        a0 = a0 or ann
        a1 = a1 or ann
        t0 = self.by_token["sample"][a0["sample_token"]]["timestamp"] / 1e6
        t1 = self.by_token["sample"][a1["sample_token"]]["timestamp"] / 1e6
        dt = t1 - t0
        if dt <= 1e-6:
            return None
        p0 = np.asarray(a0["translation"], dtype=float)
        p1 = np.asarray(a1["translation"], dtype=float)
        v = (p1 - p0) / dt
        return float(v[0]), float(v[1])

    # ------------------------------------------------------------ visibility -- #
    def _visible(self, channel: str, modality: str, R_s: np.ndarray, t_s: np.ndarray,
                 cal: dict, p_world: np.ndarray) -> bool:
        local = R_s.T @ (p_world - t_s)          # object in the sensor frame
        if modality == "lidar":
            return float(np.linalg.norm(local[:2])) <= LIDAR_RANGE_M
        if modality == "radar":
            if float(np.linalg.norm(local[:2])) > RADAR_RANGE_M or local[0] <= 0.0:
                return False
            return abs(math.degrees(math.atan2(local[1], local[0]))) <= RADAR_FOV_DEG / 2.0
        # camera: project with the real intrinsic and require the pixel to land on the sensor
        K = np.asarray(cal.get("camera_intrinsic") or [], dtype=float)
        if K.shape != (3, 3) or local[2] <= 1.0:
            return False
        uvw = K @ local
        u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
        return 0.0 <= u < 1600.0 and 0.0 <= v < 900.0     # nuScenes camera resolution

    # ----------------------------------------------------------- detections -- #
    def detections_for_sample(self, sample_token: str, keyframes_only: bool = True,
                              one_per_modality: bool = True) -> list[Detection]:
        """Every sensor's observations of every annotated object in this keyframe.

        Each detection carries ITS OWN sensor's timestamp, so the ~70 ms spread between sensors
        inside a keyframe survives into the tracker rather than being flattened.

        `one_per_modality` (default) keeps ONE detection per object per modality, from the unit
        that sees it best (nearest). nuScenes carries six cameras and five radars, whereas our
        `Detection` contract and the tracker's per-sensor association assume a single unit of
        each modality, which is all the simulator has. Collapsing here keeps the contract honest
        instead of teaching the tracker about sensor arrays.

        Measured effect on two scenes: camera observations 7378 -> 6759 and radar 6214 -> 4816,
        while the live track count barely moves (77.3 -> 76.9 per sample). So the duplicate units
        were NOT inflating the track count -- the tracker's gating already merged most of them --
        and ~77 tracks against ~86 objects visible within LiDAR range is correct behaviour, not
        over-tracking. Set it False to feed every unit raw and compare.
        """
        anns = self.ann_by_sample.get(sample_token, [])
        rows = [sd for sd in self.data_by_sample.get(sample_token, [])
                if sd["is_key_frame"] or not keyframes_only]
        # (instance, modality) -> (range, detection); the nearest unit wins
        best: dict[tuple[str, str], tuple[float, Detection]] = {}
        out: list[Detection] = []
        for sd in rows:
            channel, modality = self.channel_of(sd)
            cal = self.by_token["calibrated_sensor"][sd["calibrated_sensor_token"]]
            t_s, R_s = self.sensor_world_pose(sd)
            stamp = sd["timestamp"] / 1e6
            for ann in anns:
                p = np.asarray(ann["translation"], dtype=float)
                if not self._visible(channel, modality, R_s, t_s, cal, p):
                    continue
                det = self._detection(ann, modality, stamp, t_s)
                if det is None:
                    continue
                if not one_per_modality:
                    out.append(det)
                    continue
                key = (ann["instance_token"], modality)
                rng = float(np.linalg.norm(p[:2] - t_s[:2]))
                if key not in best or rng < best[key][0]:
                    best[key] = (rng, det)
        return out if not one_per_modality else [d for _, d in best.values()]

    def _detection(self, ann: dict, modality: str, stamp: float,
                   sensor_pos: np.ndarray) -> Optional[Detection]:
        """One `Detection`, populated only with what this modality can actually measure.

        The field split mirrors `Detection`'s own docstring: camera contributes class, LiDAR
        contributes extents and heading, radar contributes range rate. Nothing is cross-filled.
        """
        p = np.asarray(ann["translation"], dtype=float)
        w, l = float(ann["size"][0]), float(ann["size"][1])   # nuScenes size is (w, l, h)
        name = self.category_of_instance[ann["instance_token"]]

        if modality == "camera":
            sensor, cov = SensorType.CAMERA, np.diag([0.50, 0.50])
            return Detection(sensor, stamp, float(p[0]), float(p[1]), cov,
                             object_type=CATEGORY_MAP.get(name, ObjectType.UNKNOWN),
                             class_confidence=0.9,
                             sensor_x=float(sensor_pos[0]), sensor_y=float(sensor_pos[1]),
                             truth_id=ann["instance_token"])
        if modality == "lidar":
            return Detection(SensorType.LIDAR, stamp, float(p[0]), float(p[1]),
                             np.diag([0.10, 0.10]), length=l, width=w,
                             heading=quat_to_yaw(ann["rotation"]),
                             sensor_x=float(sensor_pos[0]), sensor_y=float(sensor_pos[1]),
                             truth_id=ann["instance_token"])
        if modality == "radar":
            vel = self.box_velocity(ann)
            radial = None
            if vel is not None:
                d = p[:2] - sensor_pos[:2]
                n = float(np.linalg.norm(d))
                if n > 1e-6:
                    radial = float((vel[0] * d[0] + vel[1] * d[1]) / n)   # receding positive
            return Detection(SensorType.RADAR, stamp, float(p[0]), float(p[1]),
                             np.diag([0.30, 0.60]), radial_speed=radial,
                             radial_speed_std=0.2 if radial is not None else 0.0,
                             sensor_x=float(sensor_pos[0]), sensor_y=float(sensor_pos[1]),
                             truth_id=ann["instance_token"])
        return None

    # --------------------------------------------------------------- scenes -- #
    def samples_of_scene(self, scene_token: str) -> Iterator[dict]:
        """Keyframes of one scene in temporal order, following the linked list."""
        scene = self.by_token["scene"][scene_token]
        token = scene["first_sample_token"]
        while token:
            s = self.by_token["sample"][token]
            yield s
            token = s["next"]
