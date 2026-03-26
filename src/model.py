"""
model.py — RoadSegNet architecture (Encoder–LMCM–Decoder with three output heads).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv2D → BN → ReLU → Conv2D → BN → ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LMCM(nn.Module):
    """
    Lightweight Multi-Scale Context Module.
    B1 — 1×1 conv              : point-wise context
    B2 — 3×3 dilated (rate=2)  : small-range context
    B3 — 3×3 dilated (rate=4)  : medium-range context
    B4 — global avg pool       : global context
    All branches output branch_ch channels; concatenated then fused back to in_ch.
    """

    def __init__(self, in_ch: int = 256, branch_ch: int = 64):
        super().__init__()

        def _branch(kernel, dilation=1):
            pad = dilation if kernel == 3 else 0
            return nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, kernel, padding=pad,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )

        self.b1      = _branch(1)
        self.b2      = _branch(3, dilation=2)
        self.b3      = _branch(3, dilation=4)
        self.b4_pool = nn.AdaptiveAvgPool2d(1)
        self.b4_conv = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * branch_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        b4   = F.interpolate(
            self.b4_conv(self.b4_pool(x)),
            size=(h, w), mode="bilinear", align_corners=False,
        )
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), b4], dim=1))


class RoadSegNet(nn.Module):
    """
    Encoder–LMCM–Decoder with three output heads:
      seg        — binary road mask      (main)
      edge       — road boundary map     (auxiliary)
      centerline — road skeleton map     (auxiliary)
    """

    def __init__(self, in_channels: int = 3,
                 encoder_channels: list[int] | None = None,
                 lmcm_branch_ch: int = 64):
        super().__init__()
        ec = encoder_channels or [32, 64, 128, 256]
        c1, c2, c3, c4 = ec

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.lmcm = LMCM(in_ch=c4, branch_ch=lmcm_branch_ch)
        self.dec1 = DoubleConv(c4 + c4, c3)
        self.dec2 = DoubleConv(c3 + c3, c2)
        self.dec3 = DoubleConv(c2 + c2, c1)
        self.dec4 = DoubleConv(c1 + c1, c1)
        self.head_seg        = nn.Conv2d(c1, 1, 1)
        self.head_edge       = nn.Conv2d(c1, 1, 1)
        self.head_centerline = nn.Conv2d(c1, 1, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b  = self.lmcm(self.pool(s4))
        d  = self.dec1(self._up(b,  s4))
        d  = self.dec2(self._up(d,  s3))
        d  = self.dec3(self._up(d,  s2))
        d  = self.dec4(self._up(d,  s1))
        return self.head_seg(d), self.head_edge(d), self.head_centerline(d)
