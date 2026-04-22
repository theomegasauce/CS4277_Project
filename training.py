"""
training.py — Train RoadSegNet on the Massachusetts Roads Dataset.

Usage:
    python training.py
    python training.py --epochs 100 --lr 3e-4 --batch_size 4 --img_size 256
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

from model1 import (
    RoadSegNet,
    build_loaders,
    iou_score,
    make_centerline_targets,
    make_edge_targets,
    total_loss,
)


# ── Config / CLI ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Train RoadSegNet on Massachusetts Roads")
    p.add_argument("--config",       type=str,   default="config.yaml")
    p.add_argument("--dataset_root", type=str,   default=None)
    p.add_argument("--img_size",     type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--save_path",    type=str,   default=None)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--num_workers",  type=int,   default=None)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_checkpoint_path(base: Path) -> Path:
    """Return the next unused `{stem}_v{N}{suffix}` alongside `base`."""
    base.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = base.parent / f"{base.stem}_v{n}{base.suffix}"
        if not candidate.exists():
            return candidate
        n += 1


# ── Training / validation loops ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, epoch,
                    edge_weight=0.3, centerline_weight=0.3, dice_smooth=1.0):
    model.train()
    totals = {"loss/total": 0.0, "loss/seg": 0.0, "loss/edge": 0.0, "loss/centerline": 0.0}

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d} train", leave=False,
                unit="batch", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        t_edge = make_edge_targets(masks)
        t_cl   = make_centerline_targets(masks)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p_seg, p_edge, p_cl = model(imgs)
            loss, comps = total_loss(p_seg, p_edge, p_cl, masks, t_edge, t_cl,
                                     edge_weight=edge_weight,
                                     centerline_weight=centerline_weight,
                                     dice_smooth=dice_smooth)

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
def validate(model, loader, device, epoch,
             edge_weight=0.3, centerline_weight=0.3, dice_smooth=1.0):
    model.eval()
    total_loss_val = 0.0
    total_iou      = 0.0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:3d}   val", leave=False,
                unit="batch", dynamic_ncols=True)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        t_edge = make_edge_targets(masks)
        t_cl   = make_centerline_targets(masks)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            p_seg, p_edge, p_cl = model(imgs)
            _, comps = total_loss(p_seg, p_edge, p_cl, masks, t_edge, t_cl,
                                  edge_weight=edge_weight,
                                  centerline_weight=centerline_weight,
                                  dice_smooth=dice_smooth)

        batch_iou = iou_score(p_seg, masks)
        total_loss_val += comps["loss/total"]
        total_iou      += batch_iou
        pbar.set_postfix(loss=f"{comps['loss/total']:.3f}", iou=f"{batch_iou:.3f}")

    n = len(loader)
    return total_loss_val / n, total_iou / n


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(history, save_path="training_history.png"):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs_range, history["val_iou"], color="tab:green", label="Val IoU")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU")
    ax2.set_title("Validation IoU"); ax2.legend(); ax2.grid(True)

    plt.suptitle("Training History", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    loss_cfg  = cfg["loss"]
    train_cfg = cfg["training"]

    dataset_root = Path(args.dataset_root or data_cfg["dataset_root"])
    img_size     = args.img_size    or data_cfg["img_size"]
    batch_size   = args.batch_size  or data_cfg["batch_size"]
    num_workers  = args.num_workers if args.num_workers is not None else data_cfg["num_workers"]
    seed         = args.seed        or train_cfg["seed"]
    epochs       = args.epochs      or train_cfg["epochs"]
    lr           = args.lr          or train_cfg["lr"]
    weight_decay = args.weight_decay or train_cfg["weight_decay"]
    base_save_path = Path(args.save_path or train_cfg["save_path"])
    # When resuming, keep writing to the resumed file; otherwise allocate a new
    # numbered file so each run's best checkpoint is preserved.
    save_path = (Path(args.resume) if args.resume is not None
                 else next_checkpoint_path(base_save_path))

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Config      : {args.config}")
    print(f"Device      : {device}")
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Dataset root: {dataset_root}")
    print(f"Checkpoint  : {save_path}")

    # ── Data ──────────────────────────────────────────────────────────────
    aug_cfg = data_cfg.get("augmentation", {})
    train_loader, val_loader, _ = build_loaders(
        dataset_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        brightness=aug_cfg.get("brightness", 0.2),
        contrast=aug_cfg.get("contrast", 0.2),
        saturation=aug_cfg.get("saturation", 0.1),
    )
    print(f"\nTrain : {len(train_loader.dataset):>5} images -> {len(train_loader):>4} batches (augmented)")
    print(f"Val   : {len(val_loader.dataset):>5} images -> {len(val_loader):>4} batches")

    # ── Model ─────────────────────────────────────────────────────────────
    model = RoadSegNet(
        in_channels=model_cfg.get("in_channels", 3),
        encoder_channels=model_cfg.get("encoder_channels"),
        lmcm_branch_ch=model_cfg.get("lmcm_branch_ch", 64),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nRoadSegNet — {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=train_cfg.get("scheduler_eta_min", 1e-6))
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 1
    best_iou    = 0.0
    history     = {"train_loss": [], "val_loss": [], "val_iou": []}

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"\nResuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "history" in ckpt:
            history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        best_iou    = ckpt.get("val_iou", 0.0)
        print(f"  Resumed at epoch {start_epoch}, best IoU so far: {best_iou:.4f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    edge_w       = loss_cfg.get("edge_weight", 0.3)
    centerline_w = loss_cfg.get("centerline_weight", 0.3)
    dice_smooth  = loss_cfg.get("dice_smooth", 1.0)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nTraining RoadSegNet — epochs {start_epoch}..{epochs} on {device}")
    header = f"{'Epoch':>7}  {'Train Loss':>10}  {'Seg':>6}  {'Edge':>6}  {'CL':>6}  {'Val Loss':>9}  {'Val IoU':>8}"
    print(header)
    print("-" * len(header))

    t_start = time.time()
    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="Training", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        loss_kwargs = dict(edge_weight=edge_w, centerline_weight=centerline_w, dice_smooth=dice_smooth)
        train_comps       = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, **loss_kwargs)
        val_loss, val_iou = validate(model, val_loader, device, epoch, **loss_kwargs)
        scheduler.step()

        history["train_loss"].append(train_comps["loss/total"])
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        improved = val_iou > best_iou
        if improved:
            best_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "val_iou": best_iou,
                "history": history,
            }, save_path)

        epoch_bar.set_postfix(val_iou=f"{val_iou:.4f}", best=f"{best_iou:.4f}")
        tqdm.write(
            f"{epoch:7d}  {train_comps['loss/total']:10.4f}  "
            f"{train_comps['loss/seg']:6.3f}  "
            f"{train_comps['loss/edge']:6.3f}  "
            f"{train_comps['loss/centerline']:6.3f}  "
            f"{val_loss:9.4f}  {val_iou:8.4f}"
            + ("  *" if improved else "")
        )

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Best val IoU: {best_iou:.4f}  ->  {save_path}")

    plot_history(history)


if __name__ == "__main__":
    main()
