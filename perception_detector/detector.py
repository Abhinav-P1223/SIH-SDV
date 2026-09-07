"""A real, learned camera object detector: Faster R-CNN MobileNetV3-Large-320-FPN, COCO-pretrained.

WHAT THIS IS, STATED BEFORE ANYTHING ELSE
    The weights are torchvision's COCO-trained detector weights. **We did not train them.** It
    has never seen an Indian road, an auto-rickshaw, or a pushcart. It is a genuine learned
    detector running real inference on real photographs, and it is not a detector we can claim as
    a contribution. Every number this package reports is a number about a COCO detector.

WHY NOT TRAIN OUR OWN
    Measured during the Phase 6 audit, not assumed. The only labelled detection data available
    without a new download is nuScenes v1.0-mini, whose 18,538 annotations are just 911 distinct
    physical objects across ten Boston and Singapore scenes. Only car (390 instances) and
    pedestrian (213) have enough diversity to learn; truck (28) is marginal and motorcycle (20),
    bicycle (15) and bus (15) are not. Auto-rickshaw, cattle and pushcart have none at all.
    Training a detector from scratch on that measured 10.3 min per epoch on this CPU, so 40 epochs
    is 6.9 hours, to fit two classes from the wrong continent. IDD-Lite cannot help: its
    `_inst_label.png` files are not instance maps, they repeat the seven semantic ids, so a derived
    box covers most of the frame.

    A COCO detector reaches seven of our nine object types today, runs on real Indian-road images,
    and can be measured against held-out nuScenes projections it also never trained on. That is a
    better and more honest result than seven hours of CPU spent overfitting 911 objects.

WHY THIS ARCHITECTURE AND NOT SSDLITE
    SSDLite320 was the audit's recommendation on size and speed alone (2.3 M parameters, 99 ms).
    Measured on 8 real nuScenes CAM_FRONT images carrying 147 ground-truth boxes in our classes, it
    was not usable: 4 detections at a 0.35 score threshold, 18 at 0.2, with a score distribution so
    flat that its 90th percentile sat at 0.14. Faster R-CNN MobileNetV3-Large-320-FPN found 31 and
    72 on the same images for 16% more time (263 ms against 226 ms). Both are torchvision,
    BSD-3-Clause, and neither adds a dependency. Eight times the recall for 16% of the time is not
    a close decision, so the audit's recommendation was overridden by measurement.

    `model_name` still accepts "ssdlite320_mobilenet_v3_large" so the comparison can be rerun.

WHAT IT CANNOT SEE
    AUTO_RICKSHAW and PUSHCART have no COCO equivalent. They are the two classes that matter most
    on an Indian road and the detector is blind to both. This is not a tuning problem and no
    confidence threshold fixes it; it needs Indian training data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from autonomy.core.types import ObjectType

# COCO index -> our ObjectType. Only mappings that are genuinely the same thing are listed; a
# guessed mapping would put a wrong class label into the tracker's voting, which is worse than
# UNKNOWN. `cow` is included because it is the honest match for CATTLE on an Indian road, and
# `horse` because a cart animal reads as one; both are reported as CATTLE.
COCO_TO_OBJECT_TYPE: dict[int, ObjectType] = {
    1: ObjectType.PEDESTRIAN,
    2: ObjectType.BICYCLE,
    3: ObjectType.CAR,
    4: ObjectType.MOTORCYCLE,
    6: ObjectType.BUS,
    8: ObjectType.TRUCK,
    19: ObjectType.CATTLE,      # horse
    21: ObjectType.CATTLE,      # cow
}

# Stated so a reader does not have to diff the map against the enum to find the holes.
UNREACHABLE_TYPES = (ObjectType.AUTO_RICKSHAW, ObjectType.PUSHCART)


@dataclass
class DetectorConfig:
    """Thresholds, in one place so a reviewer can argue with them."""
    score_threshold: float = 0.35      # below this a COCO box on this data is mostly noise
    max_detections: int = 40
    min_box_px: int = 8                # a box smaller than this cannot be localised usefully
    device: str = "cpu"
    model_name: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    # Phase 7C: when set, load a checkpoint whose head predicts OUR classes directly instead of
    # COCO's 91. Leaving it None keeps the Phase 6 behaviour byte for byte, which is what the A/B
    # baseline arm relies on.
    finetuned_checkpoint: "Path | str | None" = None


@dataclass
class RawDetection:
    """One detector output, still in IMAGE space. No world geometry has been applied yet."""
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    coco_id: int
    object_type: ObjectType

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def bottom_centre(self) -> tuple[float, float]:
        """Where the object meets the ground, which is what the range estimate keys off."""
        return 0.5 * (self.x0 + self.x1), self.y1


# torchvision builder -> its COCO weights enum, so the config names one string and nothing else.
_WEIGHTS_ENUM = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
}


class CameraObjectDetector:
    """A COCO-pretrained torchvision detector, wrapped to emit `RawDetection`s.

    Deterministic: eval mode, no dropout, fixed preprocessing, fixed thresholds. The same image
    always yields the same boxes, which the tests assert directly.
    """

    def __init__(self, cfg: DetectorConfig | None = None, weights: str = "COCO_V1"):
        import torch
        from torchvision.models import detection as tvdet
        self.cfg = cfg or DetectorConfig()
        self.torch = torch
        self.device = torch.device(self.cfg.device)
        name = self.cfg.model_name
        if name not in _WEIGHTS_ENUM:
            raise ValueError(f"unsupported detector {name!r}; expected one of {sorted(_WEIGHTS_ENUM)}")
        w = getattr(getattr(tvdet, _WEIGHTS_ENUM[name]), weights)
        self.finetuned = self.cfg.finetuned_checkpoint is not None
        if self.finetuned:
            # Our own head: labels are already ObjectTypes, so no COCO translation happens.
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from .uvh26 import ID_TO_OBJECT_TYPE, NUM_CLASSES
            state = torch.load(Path(self.cfg.finetuned_checkpoint), map_location=self.device,
                               weights_only=False)
            m = getattr(tvdet, name)(weights=None, weights_backbone=None)
            in_f = m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
            m.load_state_dict(state["model"])
            self.model = m.eval().to(self.device)
            self.categories = list(state.get("class_names", []))
            self.label_map = dict(ID_TO_OBJECT_TYPE)
            self.checkpoint_meta = {k: state.get(k) for k in ("epoch", "git_commit", "config")}
            self.coco_map = float("nan")
        else:
            self.model = getattr(tvdet, name)(weights=w).eval().to(self.device)
            self.categories = list(w.meta["categories"])
            self.label_map = dict(COCO_TO_OBJECT_TYPE)
            self.checkpoint_meta = {}
            self.coco_map = float(w.meta.get("_metrics", {}).get("COCO-val2017", {}).get("box_map", float("nan")))
        self.last_inference_ms = 0.0

    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> list[RawDetection]:
        """RGB uint8 (H, W, 3) -> detections in image pixel coordinates.

        Boxes whose COCO class has no honest `ObjectType` equivalent are DROPPED rather than
        reported as UNKNOWN. A traffic light or a bench is not an object the planner should be
        asked to avoid, and feeding it in as UNKNOWN would manufacture obstacles.
        """
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(image).copy()).permute(2, 0, 1).float() / 255.0
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model([x.to(self.device)])[0]
        self.last_inference_ms = (time.perf_counter() - t0) * 1e3

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        dets: list[RawDetection] = []
        for (x0, y0, x1, y1), s, lab in zip(boxes, scores, labels):
            if s < self.cfg.score_threshold:
                continue
            otype = self.label_map.get(int(lab))
            if otype is None:
                continue
            if (x1 - x0) < self.cfg.min_box_px or (y1 - y0) < self.cfg.min_box_px:
                continue
            dets.append(RawDetection(float(x0), float(y0), float(x1), float(y1),
                                     float(s), int(lab), otype))
            if len(dets) >= self.cfg.max_detections:
                break
        return dets

    def coco_name(self, coco_id: int) -> str:
        return self.categories[coco_id] if 0 <= coco_id < len(self.categories) else str(coco_id)
