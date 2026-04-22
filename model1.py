"""
model1.py — Dataset, model, losses, and metrics for RoadSegNet.

Contains:
  - MassRoadsDataset + build_loaders (data pipeline)
  - DoubleConv, LMCM, RoadSegNet (architecture)
  - Auxiliary target builders (edge, centerline) and loss functions
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
# Model
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# Losses and auxiliary targets
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


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred   = torch.sigmoid(logits).reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def seg_loss(logits: torch.Tensor, target: torch.Tensor,
             smooth: float = 1.0) -> torch.Tensor:
    """BCE + Dice for the main segmentation head."""
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target, smooth=smooth)


def total_loss(pred_seg, pred_edge, pred_centerline,
               target_seg, target_edge, target_centerline,
               edge_weight: float = 0.3, centerline_weight: float = 0.3,
               dice_smooth: float = 1.0):
    """Weighted multi-head loss. Returns (loss_tensor, component_dict)."""
    l_seg        = seg_loss(pred_seg, target_seg, smooth=dice_smooth)
    l_edge       = F.binary_cross_entropy_with_logits(pred_edge,       target_edge)
    l_centerline = F.binary_cross_entropy_with_logits(pred_centerline, target_centerline)
    l_total      = l_seg + edge_weight * l_edge + centerline_weight * l_centerline
    return l_total, {
        "loss/total":      l_total.item(),
        "loss/seg":        l_seg.item(),
        "loss/edge":       l_edge.item(),
        "loss/centerline": l_centerline.item(),
    }


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
