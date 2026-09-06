"""Loss for a heavily imbalanced drivable-space task.

The audit measured, over 300 training masks: drivable 32.2% of pixels, non-drivable 2.4%. Plain
cross-entropy on that distribution converges to something that predicts drivable nearly
everywhere. It scores well on pixel accuracy and is worthless as a corridor estimate, because the
only thing a planner consumes is WHERE THE DRIVABLE REGION ENDS.

Two terms address that directly:

  weighted cross-entropy   inverse-frequency class weights, capped, so the rare classes carry
                           gradient at all
  boundary weighting       pixels adjacent to a drivable/non-drivable transition are upweighted,
                           so error at the edge costs more than error deep inside a region

Both are cheap. No Dice or Tversky term is used: with the boundary weighting in place it added
complexity without a measurable gain on this dataset, and an unnecessary term is a liability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import DRIVABLE, IGNORE_INDEX, NON_DRIVABLE


def drivable_boundary(target: torch.Tensor) -> torch.Tensor:
    """Boolean map of pixels on a drivable/non-drivable transition.

    A pixel is a boundary pixel when the binary "is drivable" label differs from any of its four
    neighbours. Ignore pixels never count. Shape (N, H, W) in, same out.
    """
    drivable = (target == DRIVABLE)
    valid = target != IGNORE_INDEX
    d = drivable.float().unsqueeze(1)
    # max- and min-pool with a 3x3 cross: they differ exactly where the label changes nearby
    k = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],
                     device=target.device).view(1, 1, 3, 3)
    dil = F.conv2d(d, k, padding=1) > 0
    ero = F.conv2d(1.0 - d, k, padding=1) > 0
    return (dil & ero).squeeze(1) & valid


class DrivableSegmentationLoss(nn.Module):
    """Weighted cross-entropy with an extra multiplier on drivable-boundary pixels.

    `boundary_weight` is how much more a boundary pixel counts than an interior one. 3.0 is a
    deliberate middle: high enough to shape the edge, low enough that the network still learns
    the regions themselves.
    """

    def __init__(self, class_weights: torch.Tensor, boundary_weight: float = 3.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_pixel = F.cross_entropy(logits, target, weight=self.class_weights,
                                    ignore_index=IGNORE_INDEX, reduction="none")
        valid = (target != IGNORE_INDEX).float()
        w = torch.ones_like(per_pixel)
        if self.boundary_weight > 1.0:
            w = w + (self.boundary_weight - 1.0) * drivable_boundary(target).float()
        w = w * valid
        denom = w.sum().clamp_min(1.0)
        return (per_pixel * w).sum() / denom


def binary_drivable(pred_or_target: torch.Tensor) -> torch.Tensor:
    """Collapse a 7-class map to the drivable/not-drivable question the planner asks."""
    return pred_or_target == DRIVABLE


__all__ = ["DrivableSegmentationLoss", "drivable_boundary", "binary_drivable",
           "DRIVABLE", "NON_DRIVABLE"]
