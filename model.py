# =============================================================================
# RoadSegNet — Massachusetts Roads Dataset
# =============================================================================

# ── 1. Imports & Configuration ────────────────────────────────────────────────

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from tqdm import tqdm

from skimage.morphology import skeletonize

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"C:\Users\bryce\.cache\kagglehub\datasets\balraj98\massachusetts-roads-dataset\versions\1")

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE    = 512
BATCH_SIZE  = 8
EPOCHS      = 50
LR          = 1e-4
CKPT_PATH   = Path("best_model.pth")
# Windows uses 'spawn' for multiprocessing — num_workers=0 is faster here
NUM_WORKERS = 0 if os.name == "nt" else 4

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device      : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
print(f"NUM_WORKERS : {NUM_WORKERS}")
print(f"Dataset root: {DATASET_ROOT}")


# ── 2. Dataset Exploration ────────────────────────────────────────────────────

meta = pd.read_csv(DATASET_ROOT / "metadata.csv")
print(f"\nMetadata: {meta.shape[0]} images")
print(meta["split"].value_counts().to_string())


# ── 3. Dataset Class ──────────────────────────────────────────────────────────

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
        metadata: pd.DataFrame,
        root: Path,
        split: str,
        img_size: int = 512,
        augment: bool = False,
    ):
        self.root         = root
        self.img_size     = img_size
        self.augment      = augment
        self.normalize    = T.Normalize(mean=self.MEAN, std=self.STD)
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        self.samples      = (
            metadata[metadata["split"] == split]
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pair(self, idx: int):
        row   = self.samples.iloc[idx]
        img   = Image.open(self.root / row["tiff_image_path"]).convert("RGB")
        label = Image.open(self.root / row["tif_label_path"]).convert("L")
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


# ── 4. DataLoaders ────────────────────────────────────────────────────────────

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

_generator = torch.Generator()
_generator.manual_seed(SEED)

train_ds = MassRoadsDataset(meta, DATASET_ROOT, split="train", img_size=IMG_SIZE, augment=True)
val_ds   = MassRoadsDataset(meta, DATASET_ROOT, split="val",   img_size=IMG_SIZE, augment=False)
test_ds  = MassRoadsDataset(meta, DATASET_ROOT, split="test",  img_size=IMG_SIZE, augment=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
    worker_init_fn=seed_worker, generator=_generator,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

print(f"\nTrain : {len(train_ds):>5} images → {len(train_loader):>4} batches (augmented)")
print(f"Val   : {len(val_ds):>5} images → {len(val_loader):>4} batches")
print(f"Test  : {len(test_ds):>5} images → {len(test_loader):>4} batches")


# ── 5. Sanity Check ───────────────────────────────────────────────────────────

N_SAMPLES = 4
t0 = time.time()
samples = [train_ds[i] for i in range(N_SAMPLES)]
imgs    = torch.stack([s[0] for s in samples])
masks   = torch.stack([s[1] for s in samples])
print(f"\nLoaded {N_SAMPLES} samples in {time.time() - t0:.1f}s")
print(f"Image shape: {tuple(imgs.shape)}  |  Mask shape: {tuple(masks.shape)}")
print(f"Avg road coverage: {masks.mean() * 100:.2f}%")


# ── 6. Model Architecture ─────────────────────────────────────────────────────

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

    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3,   32)
        self.enc2 = DoubleConv(32,  64)
        self.enc3 = DoubleConv(64,  128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.lmcm = LMCM(in_ch=256, branch_ch=64)
        self.dec1 = DoubleConv(256 + 256, 128)
        self.dec2 = DoubleConv(128 + 128, 64)
        self.dec3 = DoubleConv(64  + 64,  32)
        self.dec4 = DoubleConv(32  + 32,  32)
        self.head_seg        = nn.Conv2d(32, 1, 1)
        self.head_edge       = nn.Conv2d(32, 1, 1)
        self.head_centerline = nn.Conv2d(32, 1, 1)

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
        # Return raw logits — loss functions use BCEWithLogitsLoss (AMP-safe)
        return self.head_seg(d), self.head_edge(d), self.head_centerline(d)


# Shape verification
model = RoadSegNet().to(DEVICE)
with torch.no_grad():
    _seg, _edge, _cl = model(torch.zeros(2, 3, 512, 512, device=DEVICE))
print(f"\nOutput shapes — seg: {tuple(_seg.shape)}  edge: {tuple(_edge.shape)}  centerline: {tuple(_cl.shape)}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")


# ── 7. Loss Functions & Auxiliary Targets ────────────────────────────────────

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def make_edge_targets(masks: torch.Tensor) -> torch.Tensor:
    kx        = _SOBEL_X.to(masks.device)
    ky        = _SOBEL_Y.to(masks.device)
    magnitude = torch.sqrt(F.conv2d(masks, kx, padding=1) ** 2 +
                           F.conv2d(masks, ky, padding=1) ** 2)
    return (magnitude > 0.5).float()


def make_centerline_targets(masks: torch.Tensor) -> torch.Tensor:
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


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # BCEWithLogitsLoss is AMP-safe; dice_loss applies sigmoid internally
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def total_loss(pred_seg, pred_edge, pred_centerline,
               target_seg, target_edge, target_centerline):
    l_seg        = seg_loss(pred_seg, target_seg)
    l_edge       = F.binary_cross_entropy_with_logits(pred_edge,       target_edge)
    l_centerline = F.binary_cross_entropy_with_logits(pred_centerline, target_centerline)
    l_total      = l_seg + 0.3 * l_edge + 0.3 * l_centerline
    return l_total, {
        "loss/total":      l_total.item(),
        "loss/seg":        l_seg.item(),
        "loss/edge":       l_edge.item(),
        "loss/centerline": l_centerline.item(),
    }


# ── 8. Training ───────────────────────────────────────────────────────────────

model     = RoadSegNet().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))


def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin     = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union        = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    totals = {"loss/total": 0.0, "loss/seg": 0.0, "loss/edge": 0.0, "loss/centerline": 0.0}

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d} train", leave=False,
                unit="batch", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        t_edge = make_edge_targets(masks)
        t_cl   = make_centerline_targets(masks)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            p_seg, p_edge, p_cl = model(imgs)
            loss, comps = total_loss(p_seg, p_edge, p_cl, masks, t_edge, t_cl)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for k in totals:
            totals[k] += comps[k]

        pbar.set_postfix(
            loss=f"{comps['loss/total']:.3f}",
            seg=f"{comps['loss/seg']:.3f}",
            edge=f"{comps['loss/edge']:.3f}",
        )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    total_loss_val = 0.0
    total_iou      = 0.0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d}   val", leave=False,
                unit="batch", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        t_edge = make_edge_targets(masks)
        t_cl   = make_centerline_targets(masks)

        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            p_seg, p_edge, p_cl = model(imgs)
            _, comps = total_loss(p_seg, p_edge, p_cl, masks, t_edge, t_cl)

        batch_iou = iou_score(p_seg, masks)
        total_loss_val += comps["loss/total"]
        total_iou      += batch_iou
        pbar.set_postfix(loss=f"{comps['loss/total']:.3f}", iou=f"{batch_iou:.3f}")

    n = len(loader)
    return total_loss_val / n, total_iou / n


# ── Training loop ─────────────────────────────────────────────────────────────
best_iou = 0.0
history  = {"train_loss": [], "val_loss": [], "val_iou": []}

print(f"\nTraining RoadSegNet — {EPOCHS} epochs on {DEVICE}")
print(f"{'Epoch':>7}  {'Train Loss':>10}  {'Seg':>6}  {'Edge':>6}  {'CL':>6}  {'Val Loss':>9}  {'Val IoU':>8}")
print("-" * 70)

epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch", dynamic_ncols=True)
for epoch in epoch_bar:
    train_comps       = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
    val_loss, val_iou = validate(model, val_loader, epoch)
    scheduler.step()

    history["train_loss"].append(train_comps["loss/total"])
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)

    improved = val_iou > best_iou
    if improved:
        best_iou = val_iou
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "val_iou": best_iou},
                   CKPT_PATH)

    epoch_bar.set_postfix(val_iou=f"{val_iou:.4f}", best=f"{best_iou:.4f}")
    tqdm.write(
        f"{epoch:7d}  {train_comps['loss/total']:10.4f}  "
        f"{train_comps['loss/seg']:6.3f}  "
        f"{train_comps['loss/edge']:6.3f}  "
        f"{train_comps['loss/centerline']:6.3f}  "
        f"{val_loss:9.4f}  {val_iou:8.4f}"
        + ("  *" if improved else "")
    )

print(f"\nBest val IoU: {best_iou:.4f}  →  {CKPT_PATH}")


# ── Training curves ───────────────────────────────────────────────────────────
epochs_range = range(1, len(history["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(epochs_range, history["train_loss"], label="train loss")
ax1.plot(epochs_range, history["val_loss"],   label="val loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
ax2.plot(epochs_range, history["val_iou"], color="tab:green", label="val IoU")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU")
ax2.set_title("Validation IoU"); ax2.legend(); ax2.grid(True)
plt.suptitle("Training History", fontsize=13)
plt.tight_layout()
plt.show()
