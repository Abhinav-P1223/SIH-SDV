"""Phase 6: real learned camera object detection feeding the EXISTING detection interface.

Nothing in this package touches the tracker, fusion, prediction, risk, controller or planner. It
produces `Detection` objects, which is what the simulated camera already produces, and the rest of
the stack cannot tell the difference.
"""
from .detector import COCO_TO_OBJECT_TYPE, CameraObjectDetector, DetectorConfig, RawDetection
from .geometry import CameraGeometry, boxes_to_detections

__all__ = ["CameraObjectDetector", "DetectorConfig", "RawDetection", "COCO_TO_OBJECT_TYPE",
           "CameraGeometry", "boxes_to_detections"]
