"""
evaluation.py — Standalone evaluation script for a trained RoadSegNet checkpoint.

Loads the best model, runs it on the test set, computes all metrics,
generates visual predictions, and prints a full report.

Usage:
    python evaluation.py
    python evaluation.py --checkpoint checkpoints/best_model.pth
    python evaluation.py --config config.yaml --num_samples 8
"""

import argparse
import random
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.dataset import build_loaders, MassRoadsDataset
from src.model import RoadSegNet
from src.metrics import (
    iou_score, dice_score, precision_score, recall_score, f1_score, accuracy_score,
)


# ── Config / CLI ────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained RoadSegNet checkpoint")
    p.add_argument("--config",      type=str, default="config.yaml")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to .pth checkpoint (default: from config training.save_path)")
    p.add_argument("--num_samples", type=int, default=6,
                   help="Number of sample predictions to visualize")
    p.add_argument("--save_dir",    type=str, default="evaluation_results",
                   help="Directory to save evaluation outputs")
    return p.parse_args()


# ── Reproducibility ─────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Per-batch metric collection ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, loader, device, threshold=0.5):
    """Run the model on the test set, returning per-batch and aggregate metrics."""
    model.eval()
    metric_fns = {
        "IoU":       lambda p, t: iou_score(p, t, threshold),
        "Dice":      lambda p, t: dice_score(p, t, threshold),
        "Precision": lambda p, t: precision_score(p, t, threshold),
        "Recall":    lambda p, t: recall_score(p, t, threshold),
        "F1":        lambda p, t: f1_score(p, t, threshold),
        "Accuracy":  lambda p, t: accuracy_score(p, t, threshold),
    }

    per_batch = {k: [] for k in metric_fns}

    pbar = tqdm(loader, desc="  Test eval", leave=False, unit="batch", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p_seg, _, _ = model(imgs)

        for name, fn in metric_fns.items():
            per_batch[name].append(fn(p_seg, masks))

    aggregated = {k: np.mean(v) for k, v in per_batch.items()}
    return aggregated, per_batch


# ── Confusion matrix components (dataset-wide) ─────────────────────────────

@torch.no_grad()
def compute_confusion(model, loader, device, threshold=0.5):
    """Accumulate TP, FP, FN, TN over the entire test set."""
    model.eval()
    tp = fp = fn = tn = 0

    for imgs, masks in tqdm(loader, desc="  Confusion", leave=False, unit="batch"):
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p_seg, _, _ = model(imgs)

        pred = (torch.sigmoid(p_seg) > threshold).float()
        tp += (pred * masks).sum().item()
        fp += (pred * (1 - masks)).sum().item()
        fn += ((1 - pred) * masks).sum().item()
        tn += ((1 - pred) * (1 - masks)).sum().item()

    total = tp + fp + fn + tn
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Global Precision": tp / (tp + fp + 1e-6),
        "Global Recall":    tp / (tp + fn + 1e-6),
        "Global F1":        2 * tp / (2 * tp + fp + fn + 1e-6),
        "Global Accuracy":  (tp + tn) / total,
        "Global IoU":       tp / (tp + fp + fn + 1e-6),
    }


# ── Visualizations ──────────────────────────────────────────────────────────

@torch.no_grad()
def visualize_predictions(model, loader, device, num_samples, save_path, threshold=0.5):
    """Save a grid of (image, ground truth, prediction, overlay) for random test samples."""
    model.eval()

    # Collect all test images/masks/preds
    all_imgs, all_masks, all_preds = [], [], []
    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p_seg, _, _ = model(imgs)
        pred = (torch.sigmoid(p_seg) > threshold).float()
        all_imgs.append(imgs.cpu())
        all_masks.append(masks.cpu())
        all_preds.append(pred.cpu())

    all_imgs  = torch.cat(all_imgs)
    all_masks = torch.cat(all_masks)
    all_preds = torch.cat(all_preds)

    num_samples = min(num_samples, len(all_imgs))
    indices = random.sample(range(len(all_imgs)), num_samples)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img  = MassRoadsDataset.denormalize(all_imgs[idx]).permute(1, 2, 0).numpy()
        gt   = all_masks[idx, 0].numpy()
        pred = all_preds[idx, 0].numpy()

        # Overlay: green = TP, red = FN, blue = FP
        overlay = img.copy()
        tp_mask = (pred == 1) & (gt == 1)
        fn_mask = (pred == 0) & (gt == 1)
        fp_mask = (pred == 1) & (gt == 0)
        overlay[tp_mask] = [0, 1, 0]
        overlay[fn_mask] = [1, 0, 0]
        overlay[fp_mask] = [0, 0, 1]

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Input Image")
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 2].set_title("Prediction")
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay (G=TP R=FN B=FP)")

        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Test Set Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Predictions saved to {save_path}")


def plot_metric_distributions(per_batch, save_path):
    """Box plot of per-batch metric distributions."""
    names  = list(per_batch.keys())
    values = [per_batch[k] for k in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(values, labels=names, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Score")
    ax.set_title("Per-Batch Metric Distributions on Test Set")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution plot saved to {save_path}")


def plot_confusion_matrix(confusion, save_path):
    """Plot a 2x2 confusion matrix heatmap."""
    cm = np.array([[confusion["TN"], confusion["FP"]],
                   [confusion["FN"], confusion["TP"]]])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["Actual Neg", "Actual Pos"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Pixel-Level Confusion Matrix")

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            label = f"{val:,.0f}\n({val / cm.sum() * 100:.1f}%)"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=11, color="white" if val > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    data_cfg    = cfg["data"]
    model_cfg   = cfg["model"]
    train_cfg   = cfg["training"]
    metrics_cfg = cfg.get("metrics", {})

    dataset_root = Path(data_cfg["dataset_root"])
    checkpoint   = Path(args.checkpoint or train_cfg["save_path"])
    save_dir     = Path(args.save_dir)
    threshold    = metrics_cfg.get("threshold", 0.5)
    seed         = train_cfg["seed"]

    seed_everything(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("  RoadSegNet — Evaluation")
    print("=" * 50)
    print(f"  Config     : {args.config}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Device     : {device}")
    print(f"  Threshold  : {threshold}")
    print(f"  Save dir   : {save_dir}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    aug_cfg = data_cfg.get("augmentation", {})
    _, _, test_loader = build_loaders(
        dataset_root,
        img_size=data_cfg["img_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=seed,
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        brightness=aug_cfg.get("brightness", 0.2),
        contrast=aug_cfg.get("contrast", 0.2),
        saturation=aug_cfg.get("saturation", 0.1),
    )
    print(f"  Test set: {len(test_loader.dataset)} images, {len(test_loader)} batches")
    print()

    # ── Load model ────────────────────────────────────────────────────────
    model = RoadSegNet(
        in_channels=model_cfg.get("in_channels", 3),
        encoder_channels=model_cfg.get("encoder_channels"),
        lmcm_branch_ch=model_cfg.get("lmcm_branch_ch", 64),
    ).to(device)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch   = ckpt.get("epoch", "?")
    val_iou = ckpt.get("val_iou", None)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded — {total_params:,} params, trained {epoch} epochs"
          + (f", val IoU {val_iou:.4f}" if val_iou else ""))
    print()

    # ── Aggregate metrics ─────────────────────────────────────────────────
    print("Computing test metrics ...")
    aggregated, per_batch = evaluate_test(model, test_loader, device, threshold)

    print()
    print("=" * 40)
    print("  Test Set Metrics (batch-averaged)")
    print("=" * 40)
    for name, value in aggregated.items():
        print(f"  {name:<12s}: {value:.4f}")
    print("=" * 40)
    print()

    # ── Global confusion matrix metrics ───────────────────────────────────
    print("Computing pixel-level confusion matrix ...")
    confusion = compute_confusion(model, test_loader, device, threshold)

    print()
    print("=" * 40)
    print("  Pixel-Level Confusion Stats")
    print("=" * 40)
    print(f"  TP: {confusion['TP']:>14,}")
    print(f"  FP: {confusion['FP']:>14,}")
    print(f"  FN: {confusion['FN']:>14,}")
    print(f"  TN: {confusion['TN']:>14,}")
    print(f"  {'─' * 36}")
    print(f"  Global Precision : {confusion['Global Precision']:.4f}")
    print(f"  Global Recall    : {confusion['Global Recall']:.4f}")
    print(f"  Global F1        : {confusion['Global F1']:.4f}")
    print(f"  Global IoU       : {confusion['Global IoU']:.4f}")
    print(f"  Global Accuracy  : {confusion['Global Accuracy']:.4f}")
    print("=" * 40)
    print()

    # ── Plots ─────────────────────────────────────────────────────────────
    print("Generating visualizations ...")
    visualize_predictions(
        model, test_loader, device, args.num_samples,
        save_path=save_dir / "predictions.png", threshold=threshold)
    plot_metric_distributions(
        per_batch, save_path=save_dir / "metric_distributions.png")
    plot_confusion_matrix(
        confusion, save_path=save_dir / "confusion_matrix.png")

    # ── Save metrics to text file ─────────────────────────────────────────
    report_path = save_dir / "metrics_report.txt"
    with open(report_path, "w") as f:
        f.write(f"RoadSegNet Evaluation Report\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Epoch:      {epoch}\n")
        f.write(f"Threshold:  {threshold}\n")
        f.write(f"Test size:  {len(test_loader.dataset)} images\n\n")

        f.write("Batch-Averaged Metrics\n")
        f.write("-" * 30 + "\n")
        for name, value in aggregated.items():
            f.write(f"  {name:<12s}: {value:.4f}\n")

        f.write(f"\nPixel-Level Confusion Stats\n")
        f.write("-" * 30 + "\n")
        f.write(f"  TP: {confusion['TP']:>14,}\n")
        f.write(f"  FP: {confusion['FP']:>14,}\n")
        f.write(f"  FN: {confusion['FN']:>14,}\n")
        f.write(f"  TN: {confusion['TN']:>14,}\n")
        f.write(f"  Global Precision : {confusion['Global Precision']:.4f}\n")
        f.write(f"  Global Recall    : {confusion['Global Recall']:.4f}\n")
        f.write(f"  Global F1        : {confusion['Global F1']:.4f}\n")
        f.write(f"  Global IoU       : {confusion['Global IoU']:.4f}\n")
        f.write(f"  Global Accuracy  : {confusion['Global Accuracy']:.4f}\n")

    print(f"  Report saved to {report_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
