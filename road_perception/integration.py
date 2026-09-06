"""The integration boundary between learned perception and the existing planner.

THE BOUNDARY, STATED PLAINLY
    The planner consumes a `RoadModel`: a reference polyline plus left and right boundaries, in
    metres. It does not know or care where that came from. The simulator builds one from scenario
    YAML; this module builds one from a camera image through a trained network. Both satisfy the
    same contract, and the planner is not modified in either case.

WHAT IS AND IS NOT DEMONSTRATED
    Real: the image is a real Indian-road photograph, the segmentation is a real learned model,
    the drivable mask and corridor come from that prediction, and the resulting `RoadModel` is the
    same class the planner already uses.

    Not real: IDD-Lite is a set of still photographs with no ego motion, no calibration and no
    dynamic objects. It cannot be driven through. Closing the loop on it would mean inventing a
    vehicle trajectory and a camera pose, which would be exactly the fake integration this phase
    forbids. So the demonstration ends where the honesty ends: at a planner-compatible corridor
    built from a real prediction, compared against the simulator's own corridor on the same terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from simulation.world.road import DrivableSpace

from .dataset import INPUT_H, INPUT_W, MEAN, STD, NUM_CLASSES
from .drivable import Corridor, GroundProjection, clean_mask, drivable_mask, extract_corridor
from .model import FastSCNN


@dataclass
class PerceivedRoad:
    """Everything one image produced, kept together so a demo can show each stage."""
    prediction: np.ndarray            # (H, W) class ids
    drivable: np.ndarray              # (H, W) bool, after cleaning
    corridor: Corridor
    road: DrivableSpace | None        # None when the corridor was too short to build one
    inference_ms: float
    postprocess_ms: float


class DrivableSpacePerception:
    """Camera image -> `DrivableSpace`, using the trained network.

    Deterministic: eval mode, no dropout, no augmentation, fixed preprocessing. The same image
    always yields the same corridor, which the tests rely on.
    """

    def __init__(self, checkpoint: Path | str, width: float = 1.0,
                 projection: GroundProjection | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        self.model = FastSCNN(NUM_CLASSES, width=state.get("config", {}).get("width", width))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.projection = projection or GroundProjection()
        self.trained_val = state.get("val", {})

    # ------------------------------------------------------------------ #
    @staticmethod
    def preprocess(image: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H, W, 3) -> normalised tensor at the network's input size."""
        from PIL import Image
        img = Image.fromarray(image).resize((INPUT_W, INPUT_H), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)

    def segment(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        import time
        x = self.preprocess(image).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(x)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return pred, (time.perf_counter() - t0) * 1e3

    def perceive(self, image: np.ndarray) -> PerceivedRoad:
        """The whole path, timed at each stage."""
        import time
        pred, infer_ms = self.segment(image)
        t0 = time.perf_counter()
        mask = clean_mask(drivable_mask(pred))
        corridor = extract_corridor(mask, self.projection)
        road = self.to_road_model(corridor)
        post_ms = (time.perf_counter() - t0) * 1e3
        return PerceivedRoad(pred, mask, corridor, road, infer_ms, post_ms)

    # ------------------------------------------------------------------ #
    @staticmethod
    def to_road_model(corridor: Corridor, min_points: int = 3) -> DrivableSpace | None:
        """Corridor -> the planner's own `DrivableSpace`, or None if the evidence is too thin.

        Returning None matters: a corridor built from two rows of pixels is not a road, and
        handing the planner a degenerate one would be worse than admitting we cannot see.
        """
        if corridor.valid_rows < min_points:
            return None
        ref = np.asarray(corridor.reference, dtype=float)
        left = np.asarray(corridor.left, dtype=float)
        right = np.asarray(corridor.right, dtype=float)
        if len(np.unique(np.round(ref[:, 0], 3))) < min_points:
            return None
        # DrivableSpace expects increasing arc length; our rows go near -> far already
        return DrivableSpace(ref, left, right)


def corridor_agreement(predicted: Corridor, truth: Corridor) -> dict:
    """Compare two corridors sampled at the same ranges.

    Used for Mode A versus Mode B. Both are resampled onto the overlapping range interval so the
    comparison is like for like, and the metrics are the ones a planner cares about: how far off
    the boundaries are, and how wrong the width is.
    """
    if predicted.valid_rows < 2 or truth.valid_rows < 2:
        return {"overlap_m": 0.0, "left_rmse_m": None, "right_rmse_m": None,
                "width_bias_m": None}
    lo = max(predicted.reference[:, 0].min(), truth.reference[:, 0].min())
    hi = min(predicted.reference[:, 0].max(), truth.reference[:, 0].max())
    if hi - lo < 1.0:
        return {"overlap_m": 0.0, "left_rmse_m": None, "right_rmse_m": None,
                "width_bias_m": None}
    grid = np.linspace(lo, hi, 25)

    def sample(c: Corridor, arr: np.ndarray) -> np.ndarray:
        order = np.argsort(c.reference[:, 0])
        return np.interp(grid, c.reference[order, 0], arr[order, 1])

    lp, lt = sample(predicted, predicted.left), sample(truth, truth.left)
    rp, rt = sample(predicted, predicted.right), sample(truth, truth.right)
    return {
        "overlap_m": float(hi - lo),
        "left_rmse_m": float(np.sqrt(np.mean((lp - lt) ** 2))),
        "right_rmse_m": float(np.sqrt(np.mean((rp - rt) ** 2))),
        "width_bias_m": float(np.mean(np.abs(lp - rp) - np.abs(lt - rt))),
    }
