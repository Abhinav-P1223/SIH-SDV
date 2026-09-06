"""Segmentation metrics that mean something for a planner.

Pixel accuracy is deliberately reported last and never alone: with drivable at 32% of pixels and
non-drivable at 2.4%, a network that predicts drivable everywhere scores well on it and is
useless. What matters here, in order:

  1. drivable IoU              can we find the road at all
  2. boundary F1               is the EDGE of the road in the right place, which is the only
                               part a corridor estimate actually consumes
  3. inference latency         does it fit inside the 10 Hz planning budget
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from .dataset import CLASS_NAMES, DRIVABLE, IGNORE_INDEX, NON_DRIVABLE, NUM_CLASSES


def confusion_matrix(pred: np.ndarray, target: np.ndarray,
                     num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Rows are ground truth, columns are prediction. Ignore pixels are dropped."""
    valid = target != IGNORE_INDEX
    p, t = pred[valid].astype(np.int64), target[valid].astype(np.int64)
    keep = (t < num_classes) & (p < num_classes)
    idx = t[keep] * num_classes + p[keep]
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU. Classes absent from both prediction and truth give NaN, not a free 1.0."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(1) + cm.sum(0) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels where the binary drivable label changes across a 4-neighbourhood.

    Strictly 2-D. A batched array would silently treat the batch axis as image rows and compare
    one image against the next, which is how an inflated boundary score gets reported, so the
    shape is checked rather than trusted.
    """
    if mask.ndim != 2:
        raise ValueError(f"binary_boundary expects a single (H, W) mask, got {mask.shape}")
    m = mask.astype(bool)
    b = np.zeros_like(m)
    b[:-1, :] |= m[:-1, :] != m[1:, :]
    b[1:, :] |= m[:-1, :] != m[1:, :]
    b[:, :-1] |= m[:, :-1] != m[:, 1:]
    b[:, 1:] |= m[:, :-1] != m[:, 1:]
    return b


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance_px: int = 2) -> dict:
    """F1 of the drivable/non-drivable boundary, within a pixel tolerance.

    Definition, stated exactly because the number is meaningless otherwise:
      * reduce both maps to the binary question "is this pixel drivable"
      * a boundary pixel is one whose binary label differs from a 4-neighbour
      * precision = fraction of PREDICTED boundary pixels lying within `tolerance_px` of any
        ground-truth boundary pixel, measured by Euclidean distance transform
      * recall    = fraction of GROUND-TRUTH boundary pixels within `tolerance_px` of any
        predicted boundary pixel
      * F1        = harmonic mean

    At 320x224 a 2 px tolerance is about 0.6% of the image width. If either map has no boundary
    at all the score is 0 unless both are empty, in which case it is 1: a scene with no road edge
    is perfectly described by predicting none.
    """
    gt_b = binary_boundary(target == DRIVABLE)
    pr_b = binary_boundary(pred == DRIVABLE)
    if not gt_b.any() and not pr_b.any():
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "gt_px": 0, "pred_px": 0}
    if not gt_b.any() or not pr_b.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "gt_px": int(gt_b.sum()), "pred_px": int(pr_b.sum())}

    dist_to_gt = distance_transform_edt(~gt_b)
    dist_to_pr = distance_transform_edt(~pr_b)
    precision = float((dist_to_gt[pr_b] <= tolerance_px).mean())
    recall = float((dist_to_pr[gt_b] <= tolerance_px).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "gt_px": int(gt_b.sum()), "pred_px": int(pr_b.sum())}


@dataclass
class SegmentationMetrics:
    """Accumulates over a split so every number below comes from the same pass."""
    num_classes: int = NUM_CLASSES
    tolerance_px: int = 2
    cm: np.ndarray = field(default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64))
    boundary: list = field(default_factory=list)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one image, or a batch of them.

        The confusion matrix is shape-agnostic because it works on flattened pixels, but the
        boundary score is NOT: it uses a Euclidean distance transform, and running that over a
        batched array lets a boundary pixel in one image be matched against a different image.
        So a batch is split and scored image by image.
        """
        self.cm += confusion_matrix(pred, target, self.num_classes)
        if pred.ndim == 2:
            self.boundary.append(boundary_f1(pred, target, self.tolerance_px))
            return
        if pred.ndim != 3:
            raise ValueError(f"expected (H, W) or (B, H, W), got {pred.shape}")
        for i in range(pred.shape[0]):
            self.boundary.append(boundary_f1(pred[i], target[i], self.tolerance_px))

    def summary(self) -> dict:
        iou = iou_from_confusion(self.cm)
        tp = float(self.cm[DRIVABLE, DRIVABLE])
        pred_pos = float(self.cm[:, DRIVABLE].sum())
        true_pos = float(self.cm[DRIVABLE, :].sum())
        return {
            "mean_iou": float(np.nanmean(iou)),
            "per_class_iou": {CLASS_NAMES[i]: (None if np.isnan(v) else float(v))
                              for i, v in enumerate(iou)},
            "drivable_iou": float(iou[DRIVABLE]) if not np.isnan(iou[DRIVABLE]) else None,
            "non_drivable_iou": (float(iou[NON_DRIVABLE])
                                 if not np.isnan(iou[NON_DRIVABLE]) else None),
            "drivable_precision": tp / pred_pos if pred_pos else 0.0,
            "drivable_recall": tp / true_pos if true_pos else 0.0,
            "boundary_f1": float(np.mean([b["f1"] for b in self.boundary])) if self.boundary else 0.0,
            "boundary_precision": float(np.mean([b["precision"] for b in self.boundary])) if self.boundary else 0.0,
            "boundary_recall": float(np.mean([b["recall"] for b in self.boundary])) if self.boundary else 0.0,
            "pixel_accuracy": float(np.diag(self.cm).sum() / max(self.cm.sum(), 1)),
            "confusion_matrix": self.cm.tolist(),
        }


def format_summary(s: dict) -> str:
    lines = [
        f"drivable IoU        {s['drivable_iou']:.4f}" if s["drivable_iou"] is not None else "drivable IoU        n/a",
        f"non-drivable IoU    {s['non_drivable_iou']:.4f}" if s["non_drivable_iou"] is not None else "non-drivable IoU    n/a",
        f"mean IoU            {s['mean_iou']:.4f}",
        f"boundary F1 (2 px)  {s['boundary_f1']:.4f}  (P {s['boundary_precision']:.3f} / R {s['boundary_recall']:.3f})",
        f"drivable P / R      {s['drivable_precision']:.4f} / {s['drivable_recall']:.4f}",
        f"pixel accuracy      {s['pixel_accuracy']:.4f}   <- reported last on purpose",
        "",
        "per-class IoU:",
    ]
    for name, v in s["per_class_iou"].items():
        lines.append(f"    {name:20s} {'n/a' if v is None else f'{v:.4f}'}")
    return "\n".join(lines)
