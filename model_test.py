# =============================================================================
# RoadSegNet — Test Set Evaluation
# =============================================================================

import os
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

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(r"/Users/brycewishart/.cache/kagglehub/datasets/balraj98/massachusetts-roads-dataset/versions/1")
CKPT_PATH    = Path("best_model.pth")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 512
BATCH_SIZE  = 8
THRESHOLD   = 0.5
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT_PATH}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class MassRoadsDataset(Dataset):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, metadata, root, split, img_size=512):
        self.root      = root
        self.img_size  = img_size
        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)
        self.samples   = metadata[metadata["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row   = self.samples.iloc[idx]
        img   = Image.open(self.root / row["tiff_image_path"]).convert("RGB")
        label = Image.open(self.root / row["tif_label_path"]).convert("L")

        img   = TF.resize(img,   [self.img_size, self.img_size],
                          interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        label = TF.resize(label, [self.img_size, self.img_size],
                          interpolation=T.InterpolationMode.NEAREST)

        img   = TF.to_tensor(img)
        label = TF.to_tensor(label)
        label = (label > 0.5).float()
        img   = self.normalize(img)
        return img, label

    @staticmethod
    def denormalize(tensor):
        mean = torch.tensor(MassRoadsDataset.MEAN).view(3, 1, 1)
        std  = torch.tensor(MassRoadsDataset.STD).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)


# ── Model Architecture ────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LMCM(nn.Module):
    def __init__(self, in_ch=256, branch_ch=64):
        super().__init__()

        def _branch(kernel, dilation=1):
            pad = dilation if kernel == 3 else 0
            return nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, kernel, padding=pad,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )

        self.b1      = _branch(1)
        self.b2      = _branch(3, dilation=2)
        self.b3      = _branch(3, dilation=4)
        self.b4_pool = nn.AdaptiveAvgPool2d(1)
        self.b4_conv = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * branch_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        b4   = F.interpolate(self.b4_conv(self.b4_pool(x)),
                             size=(h, w), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), b4], dim=1))


class RoadSegNet(nn.Module):
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
    def _up(x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
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


# ── Load Model ────────────────────────────────────────────────────────────────

model = RoadSegNet().to(DEVICE)
ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt['epoch']}  (val IoU: {ckpt['val_iou']:.4f})")


# ── Test DataLoader ───────────────────────────────────────────────────────────

meta     = pd.read_csv(DATASET_ROOT / "metadata.csv")
test_ds  = MassRoadsDataset(meta, DATASET_ROOT, split="test", img_size=IMG_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)
print(f"Test set: {len(test_ds)} images → {len(test_loader)} batches\n")


# ── Evaluation ────────────────────────────────────────────────────────────────

tp_total = torch.zeros(1, device=DEVICE)
fp_total = torch.zeros(1, device=DEVICE)
fn_total = torch.zeros(1, device=DEVICE)
tn_total = torch.zeros(1, device=DEVICE)

with torch.no_grad():
    for imgs, masks in tqdm(test_loader, desc="Evaluating", unit="batch"):
        imgs  = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        p_seg, _, _ = model(imgs)
        pred = (torch.sigmoid(p_seg) > THRESHOLD).float()

        tp_total += (pred * masks).sum()
        fp_total += (pred * (1 - masks)).sum()
        fn_total += ((1 - pred) * masks).sum()
        tn_total += ((1 - pred) * (1 - masks)).sum()

tp = tp_total.item()
fp = fp_total.item()
fn = fn_total.item()
tn = tn_total.item()

precision    = tp / (tp + fp + 1e-6)
recall       = tp / (tp + fn + 1e-6)
f1           = 2 * precision * recall / (precision + recall + 1e-6)
iou          = tp / (tp + fp + fn + 1e-6)
pixel_acc    = (tp + tn) / (tp + fp + fn + tn + 1e-6)

print("\n" + "=" * 45)
print(f"{'Test Set Metrics':^45}")
print("=" * 45)
print(f"  Pixel Accuracy : {pixel_acc * 100:6.2f}%")
print(f"  IoU (Jaccard)  : {iou * 100:6.2f}%")
print(f"  F1 / Dice      : {f1 * 100:6.2f}%")
print(f"  Precision      : {precision * 100:6.2f}%")
print(f"  Recall         : {recall * 100:6.2f}%")
print("=" * 45)


# ── Qualitative Visualizations ────────────────────────────────────────────────

N_VIZ = 6
indices = random.sample(range(len(test_ds)), N_VIZ)

fig, axes = plt.subplots(N_VIZ, 3, figsize=(12, N_VIZ * 4))
fig.suptitle("Test Set Predictions", fontsize=14)

with torch.no_grad():
    for row, idx in enumerate(indices):
        img, mask = test_ds[idx]
        p_seg, _, _ = model(img.unsqueeze(0).to(DEVICE))
        pred = (torch.sigmoid(p_seg) > THRESHOLD).float().squeeze().cpu()

        img_show = MassRoadsDataset.denormalize(img).permute(1, 2, 0).numpy()

        axes[row, 0].imshow(img_show)
        axes[row, 0].set_title("Image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask.squeeze().numpy(), cmap="gray")
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred.numpy(), cmap="gray")
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("test_predictions.png", dpi=100, bbox_inches="tight")
plt.show()
print("\nSaved visualization → test_predictions.png")
