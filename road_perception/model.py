"""Fast-SCNN: a two-branch segmentation network sized for real-time use on weak hardware.

Implemented from the published design (Poudel et al., 2019) rather than vendored, so there is no
licence to inherit. The shape of it:

    learning-to-downsample   3 layers to 1/8 resolution, the shared "stem"
    global feature extractor bottleneck blocks to 1/32, the deep branch
    feature fusion             1/8 detail + upsampled 1/32 context
    classifier                 depthwise-separable, then upsample to input size

Measured on this machine, CPU only, 320x224 input: 0.69 M parameters, 11 ms per frame (94 FPS),
0.24 s per training step at batch 8. That is why this architecture and not a heavier one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _ds_conv(cin: int, cout: int, s: int = 1) -> nn.Sequential:
    """Depthwise-separable convolution: the cheap workhorse of this network."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, s, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Bottleneck(nn.Module):
    """Inverted residual: expand, depthwise, project. Residual only when shape is preserved."""

    def __init__(self, cin: int, cout: int, stride: int, expand: int = 6):
        super().__init__()
        hidden = cin * expand
        self.residual = stride == 1 and cin == cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) if self.residual else self.block(x)


class PyramidPooling(nn.Module):
    """Context at four scales, concatenated with the input and projected back.

    Cheap global context matters here: whether a patch of tarmac is drivable often depends on the
    whole scene, not on its local texture.
    """

    def __init__(self, channels: int, bins: tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        inter = channels // 4
        # No BatchNorm on these branches: the 1x1 bin yields a (N, C, 1, 1) feature, which
        # BatchNorm refuses in training mode at batch size 1. Conv + ReLU avoids the degenerate
        # case entirely and costs nothing here.
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, inter, 1, bias=True), nn.ReLU(inplace=True))
            for _ in bins])
        self.bins = bins
        self.project = _conv_bn(channels + inter * len(bins), channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = [x]
        for bin_, stage in zip(self.bins, self.stages):
            p = F.adaptive_avg_pool2d(x, bin_)
            feats.append(F.interpolate(stage(p), size=size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class FastSCNN(nn.Module):
    def __init__(self, num_classes: int = 7, width: float = 1.0):
        super().__init__()
        c = lambda v: max(8, int(v * width))                                 # noqa: E731

        self.downsample = nn.Sequential(
            _conv_bn(3, c(32), 3, 2, 1),        # 1/2
            _ds_conv(c(32), c(48), 2),          # 1/4
            _ds_conv(c(48), c(64), 2))          # 1/8

        self.global_features = nn.Sequential(
            Bottleneck(c(64), c(64), 2),        # 1/16
            Bottleneck(c(64), c(64), 1),
            Bottleneck(c(64), c(96), 2),        # 1/32
            Bottleneck(c(96), c(96), 1),
            Bottleneck(c(96), c(128), 1),
            Bottleneck(c(128), c(128), 1),
            PyramidPooling(c(128)))

        self.fuse_low = nn.Sequential(nn.Conv2d(c(64), c(128), 1, bias=False),
                                      nn.BatchNorm2d(c(128)))
        self.fuse_high = nn.Sequential(nn.Conv2d(c(128), c(128), 1, bias=False),
                                       nn.BatchNorm2d(c(128)))
        self.classifier = nn.Sequential(
            _ds_conv(c(128), c(128)), _ds_conv(c(128), c(128)),
            nn.Dropout2d(0.1), nn.Conv2d(c(128), num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        detail = self.downsample(x)
        context = self.global_features(detail)
        context = F.interpolate(context, size=detail.shape[-2:], mode="bilinear",
                                align_corners=False)
        fused = F.relu(self.fuse_low(detail) + self.fuse_high(context), inplace=True)
        return F.interpolate(self.classifier(fused), size=size, mode="bilinear",
                             align_corners=False)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
