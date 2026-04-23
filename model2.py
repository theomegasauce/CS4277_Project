"""
model2.py — Structure-aware road segmentation for RoadSegNet.

Encoder–EnhancedLMCM–Decoder with three output heads (seg / edge / centerline)
and optional deep supervision from D2 and D3. Designed to preserve road
continuity, sharp boundaries, and centerline connectivity.

Contains:
  - MassRoadsDataset + build_loaders (data pipeline, mirrors model1.py)
  - ResidualBlock, CBAMGate, EnhancedLMCM, SkipFuse, RoadSegNetV2
  - Auxiliary target builders (edge, centerline)
  - Losses: BCE + Dice + soft-clDice (road), weighted BCE (edge),
    BCE + Dice (centerline), deep-supervision aux loss
  - Segmentation metrics (IoU, Dice, precision, recall, F1, accuracy)
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.morphology import skeletonize
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class MassRoadsDataset(Dataset):
    """
    Massachusetts Roads Dataset — binary road segmentation.

    Each sample is a (image, mask) pair where:
      image : FloatTensor [3, H, W]  — normalized RGB satellite image
      mask  : FloatTensor [1, H, W]  — binary road mask (1 = road, 0 = background)
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        stems: list[str],
        images_dir: Path,
        masks_dir: Path,
        img_size: int = 512,
        augment: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        image_ext: str = ".tiff",
        mask_ext: str = ".tif",
    ):
        self.stems        = stems
        self.images_dir   = images_dir
        self.masks_dir    = masks_dir
        self.img_size     = img_size
        self.augment      = augment
        self.mean         = mean or self.MEAN
        self.std          = std or self.STD
        self.normalize    = T.Normalize(mean=self.mean, std=self.std)
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        self.image_ext    = image_ext
        self.mask_ext     = mask_ext

    def __len__(self) -> int:
        return len(self.stems)

    def _load_pair(self, idx: int):
        stem  = self.stems[idx]
        img   = Image.open(self.images_dir / f"{stem}{self.image_ext}").convert("RGB")
        label = Image.open(self.masks_dir  / f"{stem}{self.mask_ext}").convert("L")
        return img, label

    def _apply_transforms(self, img: Image.Image, label: Image.Image):
        img   = TF.resize(img,   [self.img_size, self.img_size],
                          interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        label = TF.resize(label, [self.img_size, self.img_size],
                          interpolation=T.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, label = TF.hflip(img), TF.hflip(label)
            if random.random() > 0.5:
                img, label = TF.vflip(img), TF.vflip(label)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                img   = TF.rotate(img,   angle)
                label = TF.rotate(label, angle)
            img = self.color_jitter(img)

        img   = TF.to_tensor(img)
        label = TF.to_tensor(label)
        label = (label > 0.5).float()
        img   = self.normalize(img)
        return img, label

    def __getitem__(self, idx: int):
        img, label = self._load_pair(idx)
        return self._apply_transforms(img, label)

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(MassRoadsDataset.MEAN).view(3, 1, 1)
        std  = torch.tensor(MassRoadsDataset.STD).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)


def _read_stems(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    dataset_root: Path,
    img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 0 if os.name == "nt" else 4,
    seed: int = 42,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.1,
    image_ext: str = ".tiff",
    mask_ext: str = ".tif",
):
    """Build train / val / test DataLoaders from split files."""
    images_dir = dataset_root / "images"
    masks_dir  = dataset_root / "masks"
    splits_dir = dataset_root / "splits"

    train_stems = _read_stems(splits_dir / "train.txt")
    val_stems   = _read_stems(splits_dir / "val.txt")
    test_stems  = _read_stems(splits_dir / "test.txt")

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = dict(images_dir=images_dir, masks_dir=masks_dir, img_size=img_size,
                  mean=mean, std=std, brightness=brightness, contrast=contrast,
                  saturation=saturation, image_ext=image_ext, mask_ext=mask_ext)

    train_ds = MassRoadsDataset(train_stems, augment=True,  **common)
    val_ds   = MassRoadsDataset(val_stems,   augment=False, **common)
    test_ds  = MassRoadsDataset(test_stems,  augment=False, **common)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════════════════════
# Model — building blocks
# ═════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Pre-activation-free residual block: Conv-BN-ReLU-Conv-BN + shortcut → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CBAMGate(nn.Module):
    """Lightweight channel + spatial attention gate (CBAM-style)."""

    def __init__(self, ch: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        r = max(ch // reduction, 4)
        self.ch_mlp = nn.Sequential(
            nn.Conv2d(ch, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, ch, 1, bias=False),
        )
        self.sp_conv = nn.Conv2d(2, 1, spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch_att = torch.sigmoid(
            self.ch_mlp(F.adaptive_avg_pool2d(x, 1))
            + self.ch_mlp(F.adaptive_max_pool2d(x, 1))
        )
        x = x * ch_att
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.amax(dim=1, keepdim=True)
        sp_att = torch.sigmoid(self.sp_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sp_att


class EnhancedLMCM(nn.Module):
    """
    Enhanced Lightweight Multi-Scale Context Module.

    Branches:
      B1 — 1×1 conv                           : point-wise context
      B2 — 3×3 dilated (rate=2)               : small-range context
      B3 — 3×3 dilated (rate=4)               : medium-range context
      B4 — 3×3 dilated (rate=8)               : long-range context
      B5 — 1×9 then 9×1 asymmetric conv       : elongated road structure
      B6 — global average pool → 1×1 conv     : image-level context
    """

    def __init__(self, in_ch: int = 256, branch_ch: int = 64):
        super().__init__()

        def cbr(kernel, dilation=1, padding=None):
            pad = padding if padding is not None else (dilation if kernel == 3 else 0)
            return nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, kernel, padding=pad,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )

        self.b1 = cbr(1)
        self.b2 = cbr(3, dilation=2)
        self.b3 = cbr(3, dilation=4)
        self.b4 = cbr(3, dilation=8)
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, (1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, (9, 1), padding=(4, 0), bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.b6_pool = nn.AdaptiveAvgPool2d(1)
        self.b6_conv = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(6 * branch_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        b6 = F.interpolate(self.b6_conv(self.b6_pool(x)),
                           size=(h, w), mode="bilinear", align_corners=False)
        cat = torch.cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(x), self.b5(x), b6],
            dim=1,
        )
        return self.fuse(cat)


class SkipFuse(nn.Module):
    """Compress encoder skip (1×1 conv + optional gate), fuse with upsampled decoder, refine."""

    def __init__(self, skip_ch: int, dec_in_ch: int, out_ch: int,
                 compress_ch: int, gate: bool = False):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(skip_ch, compress_ch, 1, bias=False),
            nn.BatchNorm2d(compress_ch),
            nn.ReLU(inplace=True),
        )
        self.gate   = CBAMGate(compress_ch) if gate else nn.Identity()
        self.refine = ResidualBlock(dec_in_ch + compress_ch, out_ch)

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dec = F.interpolate(dec, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip_c = self.gate(self.compress(skip))
        return self.refine(torch.cat([dec, skip_c], dim=1))


# ═════════════════════════════════════════════════════════════════════════════
# Model — RoadSegNetV2
# ═════════════════════════════════════════════════════════════════════════════

class RoadSegNetV2(nn.Module):
    """
    Structure-aware encoder–decoder with three heads and deep supervision.

    Resolutions (512 input):
      s1 : 256×256, c1        s3 : 64×64,  c3
      s2 : 128×128, c2        s4 : 32×32,  c4        bottleneck : 32×32, c4

    Heads return:
      seg        — binary road mask         [B, 1, 512, 512]
      edge       — road boundary            [B, 1, 512, 512]
      centerline — skeleton / medial axis   [B, 1, 512, 512]
      aux_d2, aux_d3 — deep supervision (training only)
    """

    def __init__(self, in_channels: int = 3,
                 encoder_channels: list[int] | None = None,
                 decoder_channels: list[int] | None = None,
                 compress_channels: list[int] | None = None,
                 lmcm_branch_ch: int = 64):
        super().__init__()
        ec = encoder_channels  or [64, 64, 128, 256]
        dc = decoder_channels  or [128, 64, 32, 32]
        cc = compress_channels or [64,  32, 32, 32]
        c1, c2, c3, c4 = ec
        d1_out, d2_out, d3_out, d4_out = dc
        s4_c, s3_c, s2_c, s1_c = cc

        # Encoder — residual blocks, each stage downsamples (stride-2 first block)
        self.enc1 = nn.Sequential(
            ResidualBlock(in_channels, c1, stride=2),
            ResidualBlock(c1, c1),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2),
            ResidualBlock(c2, c2),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2),
            ResidualBlock(c3, c3),
        )
        self.enc4 = nn.Sequential(
            ResidualBlock(c3, c4, stride=2),
            ResidualBlock(c4, c4),
        )

        # Bottleneck
        self.bottleneck = EnhancedLMCM(in_ch=c4, branch_ch=lmcm_branch_ch)

        # Decoder — gates on the higher-level (deeper) skips
        self.fuse1 = SkipFuse(skip_ch=c4, dec_in_ch=c4,     out_ch=d1_out,
                              compress_ch=s4_c, gate=True)
        self.fuse2 = SkipFuse(skip_ch=c3, dec_in_ch=d1_out, out_ch=d2_out,
                              compress_ch=s3_c, gate=True)
        self.fuse3 = SkipFuse(skip_ch=c2, dec_in_ch=d2_out, out_ch=d3_out,
                              compress_ch=s2_c, gate=False)
        self.fuse4 = SkipFuse(skip_ch=c1, dec_in_ch=d3_out, out_ch=d4_out,
                              compress_ch=s1_c, gate=False)

        # Final upsample from 256 → 512, then refine.
        self.final_refine = ResidualBlock(d4_out, d4_out)

        # Heads
        self.head_seg        = nn.Conv2d(d4_out, 1, 1)
        self.head_edge       = nn.Conv2d(d4_out, 1, 1)
        self.head_centerline = nn.Conv2d(d4_out, 1, 1)

        # Deep-supervision aux heads (only used in training)
        self.aux_head_d2 = nn.Conv2d(d2_out, 1, 1)
        self.aux_head_d3 = nn.Conv2d(d3_out, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = x.shape[2:]
        s1 = self.enc1(x)     # 256×256, c1
        s2 = self.enc2(s1)    # 128×128, c2
        s3 = self.enc3(s2)    # 64×64,   c3
        s4 = self.enc4(s3)    # 32×32,   c4
        b  = self.bottleneck(s4)

        d1 = self.fuse1(b,  s4)  # 32×32,  d1_out
        d2 = self.fuse2(d1, s3)  # 64×64,  d2_out
        d3 = self.fuse3(d2, s2)  # 128×128, d3_out
        d4 = self.fuse4(d3, s1)  # 256×256, d4_out

        d4 = F.interpolate(d4, size=(h, w), mode="bilinear", align_corners=False)
        d4 = self.final_refine(d4)

        out: dict[str, torch.Tensor] = {
            "seg":        self.head_seg(d4),
            "edge":       self.head_edge(d4),
            "centerline": self.head_centerline(d4),
        }
        if self.training:
            out["aux_d2"] = F.interpolate(self.aux_head_d2(d2), size=(h, w),
                                          mode="bilinear", align_corners=False)
            out["aux_d3"] = F.interpolate(self.aux_head_d3(d3), size=(h, w),
                                          mode="bilinear", align_corners=False)
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Auxiliary targets
# ═════════════════════════════════════════════════════════════════════════════

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def make_edge_targets(masks: torch.Tensor) -> torch.Tensor:
    """Sobel-based road boundary map from binary masks."""
    kx        = _SOBEL_X.to(masks.device)
    ky        = _SOBEL_Y.to(masks.device)
    magnitude = torch.sqrt(F.conv2d(masks, kx, padding=1) ** 2 +
                           F.conv2d(masks, ky, padding=1) ** 2)
    return (magnitude > 0.5).float()


def make_centerline_targets(masks: torch.Tensor) -> torch.Tensor:
    """Morphological skeleton (centerline) from binary masks."""
    masks_np = masks.cpu().numpy().astype(bool)
    skels = np.stack([
        skeletonize(masks_np[i, 0])[np.newaxis]
        for i in range(masks_np.shape[0])
    ]).astype(np.float32)
    return torch.from_numpy(skels).to(masks.device)


# ═════════════════════════════════════════════════════════════════════════════
# Losses
# ═════════════════════════════════════════════════════════════════════════════

def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred   = torch.sigmoid(logits).reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# ---- Soft skeletonization for differentiable clDice ------------------------

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), 1, (1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Differentiable morphological skeleton (clDice paper)."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img  = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel  = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor,
                     iters: int = 3, smooth: float = 1.0) -> torch.Tensor:
    pred  = torch.sigmoid(logits)
    sp    = soft_skel(pred,   iters)
    st    = soft_skel(target, iters)
    tprec = (sp * target).sum() / (sp.sum() + smooth)
    tsens = (st * pred  ).sum() / (st.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + smooth)


# ---- Head-level losses -----------------------------------------------------

def road_loss(logits: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0, cldice_weight: float = 0.5) -> torch.Tensor:
    """BCE + Dice + 0.5·soft-clDice."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    d   = dice_loss(logits, target, smooth)
    cld = soft_cldice_loss(logits, target)
    return bce + d + cldice_weight * cld


def weighted_bce(logits: torch.Tensor, target: torch.Tensor,
                 max_pos_weight: float = 200.0, eps: float = 1e-6) -> torch.Tensor:
    """BCE with positive-class reweighting — edge maps are very sparse."""
    pos = target.sum()
    neg = target.numel() - pos
    pw  = (neg / (pos + eps)).clamp(max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def centerline_loss(logits: torch.Tensor, target: torch.Tensor,
                    smooth: float = 1.0) -> torch.Tensor:
    """BCE + Dice for the 1-pixel-wide centerline target."""
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target, smooth)


def _seg_bce_dice(logits: torch.Tensor, target: torch.Tensor,
                  smooth: float = 1.0) -> torch.Tensor:
    """BCE + Dice without clDice — used for deep-supervision aux heads."""
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target, smooth)


# ---- Total loss ------------------------------------------------------------

def total_loss(outputs: dict[str, torch.Tensor],
               target_seg: torch.Tensor,
               target_edge: torch.Tensor,
               target_centerline: torch.Tensor,
               edge_weight: float = 0.25,
               centerline_weight: float = 0.5,
               aux_weight: float = 0.2,
               dice_smooth: float = 1.0):
    """
    L = L_road + 0.25·L_edge + 0.5·L_centerline + 0.2·L_aux

    Returns (loss_tensor, component_dict).
    """
    l_road = road_loss(outputs["seg"],        target_seg, smooth=dice_smooth)
    l_edge = weighted_bce(outputs["edge"],    target_edge)
    l_cent = centerline_loss(outputs["centerline"], target_centerline, smooth=dice_smooth)

    l_total = l_road + edge_weight * l_edge + centerline_weight * l_cent
    comps = {
        "loss/road":       l_road.item(),
        "loss/edge":       l_edge.item(),
        "loss/centerline": l_cent.item(),
    }

    if "aux_d2" in outputs and "aux_d3" in outputs:
        l_aux = 0.5 * (_seg_bce_dice(outputs["aux_d2"], target_seg, smooth=dice_smooth)
                       + _seg_bce_dice(outputs["aux_d3"], target_seg, smooth=dice_smooth))
        l_total = l_total + aux_weight * l_aux
        comps["loss/aux"] = l_aux.item()

    comps["loss/total"] = l_total.item()
    return l_total, comps


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred         = _binarize(logits, threshold)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred         = _binarize(logits, threshold)
    intersection = (pred * target).sum()
    return (2.0 * intersection / (pred.sum() + target.sum() + 1e-6)).item()


def precision_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = _binarize(logits, threshold)
    tp   = (pred * target).sum()
    fp   = (pred * (1 - target)).sum()
    return (tp / (tp + fp + 1e-6)).item()


def recall_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = _binarize(logits, threshold)
    tp   = (pred * target).sum()
    fn   = ((1 - pred) * target).sum()
    return (tp / (tp + fn + 1e-6)).item()


def f1_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    p = precision_score(logits, target, threshold)
    r = recall_score(logits, target, threshold)
    return 2.0 * p * r / (p + r + 1e-6)


def accuracy_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred    = _binarize(logits, threshold)
    correct = (pred == target).float().sum()
    total   = target.numel()
    return (correct / total).item()
