"""nuScenes 3D annotations -> 2D camera boxes, for EVALUATING a detector.

These boxes are ground truth. They are never presented as detector output, and the detector being
evaluated was trained on COCO, so it has never seen this data either. That makes nuScenes-mini a
legitimate held-out evaluation set for it, which is the whole reason this module exists.

Measured yield over the mini split: 2,424 keyframe camera images across six cameras, 15,475 usable
boxes after clipping to the frame, an 8 px floor and dropping the lowest visibility bucket. That is
6.4 boxes per image, median 88 x 88 px, with 12% of images containing none.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autonomy.core.types import ObjectType

from .nuscenes import quat_to_matrix

# nuScenes category -> our ObjectType. Only honest equivalences; anything absent is not evaluated
# rather than being forced into a near-miss class.
NUSCENES_TO_OBJECT_TYPE: dict[str, ObjectType] = {
    "vehicle.car": ObjectType.CAR,
    "vehicle.truck": ObjectType.TRUCK,
    "vehicle.bus.rigid": ObjectType.BUS,
    "vehicle.bus.bendy": ObjectType.BUS,
    "vehicle.motorcycle": ObjectType.MOTORCYCLE,
    "vehicle.bicycle": ObjectType.BICYCLE,
    "human.pedestrian.adult": ObjectType.PEDESTRIAN,
    "human.pedestrian.child": ObjectType.PEDESTRIAN,
    "human.pedestrian.construction_worker": ObjectType.PEDESTRIAN,
    "human.pedestrian.police_officer": ObjectType.PEDESTRIAN,
}


@dataclass
class GroundTruthBox:
    x0: float
    y0: float
    x1: float
    y1: float
    object_type: ObjectType
    category: str
    instance_token: str
    distance_m: float          # range from the camera, for range-stratified reporting

    @property
    def wh(self) -> tuple[float, float]:
        return self.x1 - self.x0, self.y1 - self.y0


@dataclass
class CameraFrame:
    path: Path
    width: int
    height: int
    intrinsic: np.ndarray
    channel: str
    sample_token: str
    timestamp: float
    boxes: list[GroundTruthBox]


class NuScenesCamera2D:
    """Reads the mini tables directly and projects annotations into each camera keyframe."""

    TABLES = ("sample", "sample_data", "sample_annotation", "calibrated_sensor",
              "sensor", "category", "instance", "ego_pose", "scene", "visibility")

    def __init__(self, root: Path | str):
        self.root = Path(root)
        meta = self.root / "v1.0-mini"
        if not meta.is_dir():
            cands = [d for d in self.root.iterdir() if d.is_dir() and d.name.startswith("v1.0")]
            if not cands:
                raise FileNotFoundError(f"no v1.0-* metadata directory under {self.root}")
            meta = cands[0]
        self.t = {}
        for name in self.TABLES:
            p = meta / f"{name}.json"
            self.t[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        self.by = {n: {r["token"]: r for r in rows} for n, rows in self.t.items()}
        self.ann_by_sample = defaultdict(list)
        for a in self.t["sample_annotation"]:
            self.ann_by_sample[a["sample_token"]].append(a)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _corners(ann: dict) -> np.ndarray:
        w, l, h = ann["size"]
        R = quat_to_matrix(ann["rotation"])
        x = np.array([l, l, l, l, -l, -l, -l, -l]) / 2
        y = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
        z = np.array([h, h, -h, -h, h, h, -h, -h]) / 2
        return (R @ np.vstack([x, y, z])).T + np.array(ann["translation"])

    def camera_keyframes(self, channels: tuple[str, ...] | None = None) -> list[dict]:
        out = []
        for d in self.t["sample_data"]:
            if not d["is_key_frame"]:
                continue
            cs = self.by["calibrated_sensor"][d["calibrated_sensor_token"]]
            sen = self.by["sensor"][cs["sensor_token"]]
            if sen["modality"] != "camera":
                continue
            if channels and sen["channel"] not in channels:
                continue
            out.append(d)
        return out

    def frame(self, sd: dict, min_box_px: int = 8,
              drop_visibility: tuple[str, ...] = ("1",)) -> CameraFrame:
        """Project every annotation of this sample into this camera.

        `drop_visibility` removes the 0-40% bucket by default: a box that is almost entirely
        occluded is not something a detector can reasonably be scored against, and keeping it would
        depress recall for a reason that has nothing to do with the detector.
        """
        cs = self.by["calibrated_sensor"][sd["calibrated_sensor_token"]]
        K = np.asarray(cs["camera_intrinsic"], dtype=float)
        ep = self.by["ego_pose"][sd["ego_pose_token"]]
        R_e, t_e = quat_to_matrix(ep["rotation"]), np.asarray(ep["translation"], dtype=float)
        R_c, t_c = quat_to_matrix(cs["rotation"]), np.asarray(cs["translation"], dtype=float)
        W, H = int(sd["width"]), int(sd["height"])

        boxes: list[GroundTruthBox] = []
        for ann in self.ann_by_sample[sd["sample_token"]]:
            if ann.get("visibility_token") in drop_visibility:
                continue
            pts = (self._corners(ann) - t_e) @ R_e
            pts = (pts - t_c) @ R_c
            if (pts[:, 2] <= 0.1).any():          # any corner behind the image plane
                continue
            uv = (K @ pts.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            x0, y0 = uv.min(0)
            x1, y1 = uv.max(0)
            x0, y0 = max(float(x0), 0.0), max(float(y0), 0.0)
            x1, y1 = min(float(x1), float(W)), min(float(y1), float(H))
            if x1 - x0 < min_box_px or y1 - y0 < min_box_px:
                continue
            cat = self.by["category"][self.by["instance"][ann["instance_token"]]["category_token"]]["name"]
            otype = NUSCENES_TO_OBJECT_TYPE.get(cat)
            if otype is None:
                continue                          # barriers, cones, debris: not our object set
            boxes.append(GroundTruthBox(x0, y0, x1, y1, otype, cat, ann["instance_token"],
                                        float(np.linalg.norm(pts.mean(0)))))
        sen = self.by["sensor"][cs["sensor_token"]]
        return CameraFrame(self.root / sd["filename"], W, H, K, sen["channel"],
                           sd["sample_token"], sd["timestamp"] / 1e6, boxes)
