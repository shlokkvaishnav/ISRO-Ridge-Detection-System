"""DBR-Net-inspired dual-branch ridge detector.

Adapted, not reproduced, from Lu, Sun, Shu, Zhao & Li 2025, "Detecting the
Lunar Wrinkle Ridges Through Deep Learning Based on DEM and Aspect Data",
Research in Astronomy and Astrophysics, DOI 10.1088/1674-4527/ade352 (read
in full -- see research/RELATED_WORK.md and
research/arm_c_deep_learning/SPEC.md).

Two deliberate deviations from the paper, both because this project's
constraints are genuinely different, not because the paper's choices were
wrong for its own setting:

1. Each branch here is a small 3-block CNN, not a ResNet-34. The paper
   trained on 1,069 hand-labeled 512x512 tiles on an RTX A5000; this
   project has ~150 weakly-labeled tiles and CPU/Kaggle-GPU-notebook
   compute. A ResNet-34-per-branch model on that much less data would very
   likely just memorize the training set.
2. Fusion happens once, at the bottleneck, not at multiple stages. The
   paper fuses low-level features additively and high-level features via
   ACFF; this keeps only the ACFF-style high-level fusion, again to keep
   the parameter count appropriate for the smaller dataset.

The ACFF (Attention Complementary Feature Fusion) module itself follows the
paper's described three-step mechanism: elementwise-maximize the two
branches' features, aggregate local channel context via a 1x1 conv, then
fuse via an elementwise-weighted combination of both branches.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Branch(nn.Module):
    """A small encoder -- see module docstring for why this isn't a
    ResNet-34 the way DBR-Net's own branches are."""

    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()
        self.b1 = ConvBlock(in_ch, base)
        self.b2 = ConvBlock(base, base * 2)
        self.b3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.b1(x)
        f2 = self.b2(self.pool(f1))
        f3 = self.b3(self.pool(f2))
        return f3


class ACFF(nn.Module):
    """Attention Complementary Feature Fusion, per Lu et al. 2025 Sec. 2.2's
    description: maximize the two branches' features, aggregate local
    channel context with a 1x1 conv, then fuse via an elementwise-weighted
    combination of both inputs."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        maxed = torch.maximum(a, b)
        weights = torch.sigmoid(self.conv1x1(maxed))
        return torch.cat([a * weights, b * weights], dim=1)


class DualBranchRidgeNet(nn.Module):
    """DEM branch + aspect-variance branch, fused via ACFF at the
    bottleneck, decoded back to a per-pixel ridge probability map."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.dem_branch = Branch(in_ch=1, base=base)
        self.aspect_branch = Branch(in_ch=1, base=base)
        self.fuse = ACFF(base * 4)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base * 4 * 2, base * 2, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, dem: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        d = self.dem_branch(dem)
        a = self.aspect_branch(aspect)
        fused = self.fuse(d, a)
        up = self.up(fused)
        return self.head(up)
