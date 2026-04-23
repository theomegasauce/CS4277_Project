"""
model_comparison.py — Side-by-side evaluation of RoadSegNet (v1) vs RoadSegNetV2 (v2).

Loads both checkpoints, runs the test-set evaluation from testing.py for each,
and writes comparison plots + a combined report:
  - metrics_bar.png            grouped bar chart of batch-averaged metrics
  - metric_distributions.png   per-metric box plots (v1 vs v2) side by side
  - confusion_matrices.png     side-by-side confusion heatmaps
  - predictions_compare.png    same images predicted by v1 and v2
  - comparison_report.txt      text summary with deltas

Usage:
    python model_comparison.py
    python model_comparison.py --ckpt_v1 checkpoints/best_model_v1.pth \\
                               --ckpt_v2 checkpoints/best_model_v2.pth \\
                               --num_samples 6 --save_dir comparison_results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model1 import MassRoadsDataset, build_loaders
from testing import (
    build_model,
    compute_confusion,
    evaluate_test,
    load_config,
    seed_everything,
    seg_logits_from,
)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compare v1 and v2 RoadSegNet checkpoints")
    p.add_argument("--config",      type=str, default="config.yaml")
    p.add_argument("--ckpt_v1",     type=str, default="checkpoints/best_model_v1.pth")
    p.add_argument("--ckpt_v2",     type=str, default="checkpoints/best_model_v2.pth")
    p.add_argument("--num_samples", type=int, default=6,
                   help="Number of shared test samples to visualize")
    p.add_argument("--save_dir",    type=str, default="comparison_results")
    return p.parse_args()


# ── Shared-sample prediction rendering ──────────────────────────────────────

@torch.no_grad()
def predict_all(model, loader, device, threshold):
    """Run a model over the test loader; return stacked imgs, masks, preds on CPU."""
    model.eval()
    imgs_all, masks_all, preds_all = [], [], []
    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)
        p_seg = seg_logits_from(outputs)
        preds = (torch.sigmoid(p_seg) > threshold).float()
        imgs_all.append(imgs.cpu())
        masks_all.append(masks.cpu())
        preds_all.append(preds.cpu())
    return torch.cat(imgs_all), torch.cat(masks_all), torch.cat(preds_all)


def _overlay(img_np, gt_np, pred_np):
    out = img_np.copy()
    tp = (pred_np == 1) & (gt_np == 1)
    fn = (pred_np == 0) & (gt_np == 1)
    fp = (pred_np == 1) & (gt_np == 0)
    out[tp] = [0, 1, 0]
    out[fn] = [1, 0, 0]
    out[fp] = [0, 0, 1]
    return out


def plot_side_by_side_predictions(imgs, masks, preds_v1, preds_v2,
                                  indices, save_path):
    """Grid: image | GT | v1 pred | v1 overlay | v2 pred | v2 overlay."""
    n = len(indices)
    fig, axes = plt.subplots(n, 6, figsize=(22, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img_np = MassRoadsDataset.denormalize(imgs[idx]).permute(1, 2, 0).numpy()
        gt_np  = masks[idx, 0].numpy()
        p1     = preds_v1[idx, 0].numpy()
        p2     = preds_v2[idx, 0].numpy()

        axes[row, 0].imshow(img_np);                      axes[row, 0].set_title("Image")
        axes[row, 1].imshow(gt_np, cmap="gray");          axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(p1,    cmap="gray");          axes[row, 2].set_title("v1 Pred")
        axes[row, 3].imshow(_overlay(img_np, gt_np, p1)); axes[row, 3].set_title("v1 Overlay")
        axes[row, 4].imshow(p2,    cmap="gray");          axes[row, 4].set_title("v2 Pred")
        axes[row, 5].imshow(_overlay(img_np, gt_np, p2)); axes[row, 5].set_title("v2 Overlay")
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Prediction Comparison — v1 vs v2 (G=TP R=FN B=FP)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Side-by-side predictions saved to {save_path}")


# ── Comparison plots ────────────────────────────────────────────────────────

_METRIC_ORDER = ["IoU", "Dice", "Precision", "Recall", "F1", "Accuracy"]


def plot_metrics_bar(agg_v1, agg_v2, save_path):
    names = [m for m in _METRIC_ORDER if m in agg_v1 and m in agg_v2]
    v1 = [agg_v1[m] for m in names]
    v2 = [agg_v2[m] for m in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, v1, width, label="v1", color="#4C72B0")
    b2 = ax.bar(x + width / 2, v2, width, label="v2", color="#55A868")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(v1), max(v2)) * 1.12)
    ax.set_title("Test-Set Metrics — v1 vs v2")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrics bar chart saved to {save_path}")


def plot_paired_distributions(per_v1, per_v2, save_path):
    """Grouped box plots: for each metric, show v1 and v2 side by side."""
    names = [m for m in _METRIC_ORDER if m in per_v1 and m in per_v2]
    positions_v1 = np.arange(len(names)) * 3.0
    positions_v2 = positions_v1 + 1.0

    fig, ax = plt.subplots(figsize=(12, 5))
    bp1 = ax.boxplot([per_v1[m] for m in names], positions=positions_v1,
                     widths=0.8, patch_artist=True, manage_ticks=False)
    bp2 = ax.boxplot([per_v2[m] for m in names], positions=positions_v2,
                     widths=0.8, patch_artist=True, manage_ticks=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#4C72B0"); patch.set_alpha(0.7)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#55A868"); patch.set_alpha(0.7)

    ax.set_xticks(positions_v1 + 0.5)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Per-Batch Metric Distributions — v1 (blue) vs v2 (green)")
    ax.grid(True, axis="y", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#4C72B0", alpha=0.7),
               plt.Rectangle((0, 0), 1, 1, facecolor="#55A868", alpha=0.7)]
    ax.legend(handles, ["v1", "v2"], loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Paired distributions saved to {save_path}")


def plot_dual_confusion(conf_v1, conf_v2, save_path):
    def _draw(ax, conf, title):
        cm = np.array([[conf["TN"], conf["FP"]],
                       [conf["FN"], conf["TP"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Neg", "Pred Pos"])
        ax.set_yticklabels(["Actual Neg", "Actual Pos"])
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                ax.text(j, i, f"{v:,.0f}\n({v / cm.sum() * 100:.1f}%)",
                        ha="center", va="center", fontsize=10,
                        color="white" if v > cm.max() / 2 else "black")
        return im

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    _draw(ax1, conf_v1, "v1 — Pixel Confusion")
    im2 = _draw(ax2, conf_v2, "v2 — Pixel Confusion")
    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    plt.suptitle("Pixel-Level Confusion Matrix Comparison", fontsize=13)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dual confusion saved to {save_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def _load_and_eval(ckpt_path: Path, forced_name: str | None,
                   model_cfg: dict, test_loader, device, threshold):
    """Load one checkpoint and run evaluate_test + compute_confusion + predict_all."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    name = (forced_name or ckpt.get("model_name") or "v1").lower()
    model = build_model(name, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    params  = sum(p.numel() for p in model.parameters())
    epoch   = ckpt.get("epoch", "?")
    val_iou = ckpt.get("val_iou", None)
    print(f"  {name}: {params:,} params, epoch {epoch}"
          + (f", val IoU {val_iou:.4f}" if val_iou else ""))

    aggregated, per_batch = evaluate_test(model, test_loader, device, threshold)
    confusion             = compute_confusion(model, test_loader, device, threshold)
    imgs, masks, preds    = predict_all(model, test_loader, device, threshold)

    return {
        "name":       name,
        "params":     params,
        "epoch":      epoch,
        "val_iou":    val_iou,
        "aggregated": aggregated,
        "per_batch":  per_batch,
        "confusion":  confusion,
        "imgs":       imgs,
        "masks":      masks,
        "preds":      preds,
    }


def _write_report(path, results_v1, results_v2, threshold):
    agg1, agg2   = results_v1["aggregated"], results_v2["aggregated"]
    conf1, conf2 = results_v1["confusion"], results_v2["confusion"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("RoadSegNet v1 vs v2 — Comparison Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"v1 params: {results_v1['params']:,}   epoch: {results_v1['epoch']}\n")
        f.write(f"v2 params: {results_v2['params']:,}   epoch: {results_v2['epoch']}\n\n")

        f.write("Batch-Averaged Metrics (higher = better)\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<12} {'v1':>10} {'v2':>10} {'Δ (v2-v1)':>12}\n")
        for m in _METRIC_ORDER:
            if m in agg1 and m in agg2:
                d = agg2[m] - agg1[m]
                f.write(f"{m:<12} {agg1[m]:>10.4f} {agg2[m]:>10.4f} {d:>+12.4f}\n")

        f.write("\nPixel-Level Confusion\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Field':<20} {'v1':>14} {'v2':>14}\n")
        for k in ("TP", "FP", "FN", "TN"):
            f.write(f"{k:<20} {conf1[k]:>14,} {conf2[k]:>14,}\n")
        for k in ("Global Precision", "Global Recall", "Global F1",
                  "Global IoU", "Global Accuracy"):
            f.write(f"{k:<20} {conf1[k]:>14.4f} {conf2[k]:>14.4f}\n")

    print(f"  Report saved to {path}")


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    data_cfg    = cfg["data"]
    model_cfg   = cfg["model"]
    train_cfg   = cfg["training"]
    metrics_cfg = cfg.get("metrics", {})

    dataset_root = Path(data_cfg["dataset_root"])
    save_dir     = Path(args.save_dir)
    threshold    = metrics_cfg.get("threshold", 0.5)
    seed         = train_cfg["seed"]

    seed_everything(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("  RoadSegNet — v1 vs v2 Comparison")
    print("=" * 50)
    print(f"  Config     : {args.config}")
    print(f"  Device     : {device}")
    print(f"  Threshold  : {threshold}")
    print(f"  Save dir   : {save_dir}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
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
    print(f"  Test set: {len(test_loader.dataset)} images, {len(test_loader)} batches\n")

    # ── Evaluate both models ──────────────────────────────────────────────
    print("Evaluating v1 ...")
    results_v1 = _load_and_eval(Path(args.ckpt_v1), "v1", model_cfg,
                                test_loader, device, threshold)
    print("\nEvaluating v2 ...")
    results_v2 = _load_and_eval(Path(args.ckpt_v2), "v2", model_cfg,
                                test_loader, device, threshold)

    # ── Summary table ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  {'Metric':<12} {'v1':>12} {'v2':>12} {'Δ (v2-v1)':>14}")
    print("=" * 60)
    for m in _METRIC_ORDER:
        if m in results_v1["aggregated"] and m in results_v2["aggregated"]:
            a, b = results_v1["aggregated"][m], results_v2["aggregated"][m]
            print(f"  {m:<12} {a:>12.4f} {b:>12.4f} {b - a:>+14.4f}")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating comparison plots ...")
    plot_metrics_bar(results_v1["aggregated"], results_v2["aggregated"],
                     save_dir / "metrics_bar.png")
    plot_paired_distributions(results_v1["per_batch"], results_v2["per_batch"],
                              save_dir / "metric_distributions.png")
    plot_dual_confusion(results_v1["confusion"], results_v2["confusion"],
                        save_dir / "confusion_matrices.png")

    # Same-sample visual comparison — deterministic indices under the fixed seed.
    n_total   = len(results_v1["imgs"])
    n_samples = min(args.num_samples, n_total)
    rng       = np.random.default_rng(seed)
    indices   = rng.choice(n_total, size=n_samples, replace=False).tolist()

    plot_side_by_side_predictions(
        imgs=results_v1["imgs"], masks=results_v1["masks"],
        preds_v1=results_v1["preds"], preds_v2=results_v2["preds"],
        indices=indices, save_path=save_dir / "predictions_compare.png",
    )

    # ── Report ────────────────────────────────────────────────────────────
    _write_report(save_dir / "comparison_report.txt",
                  results_v1, results_v2, threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
