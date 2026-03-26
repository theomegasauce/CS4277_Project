"""
metrics.py — Evaluation metrics for binary segmentation.
"""

import torch


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Sigmoid + threshold → binary predictions."""
    return (torch.sigmoid(logits) > threshold).float()


def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Intersection-over-Union (Jaccard index)."""
    pred         = _binarize(logits, threshold)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Dice coefficient (F1 at the pixel level)."""
    pred         = _binarize(logits, threshold)
    intersection = (pred * target).sum()
    return (2.0 * intersection / (pred.sum() + target.sum() + 1e-6)).item()


def precision_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Pixel-wise precision: TP / (TP + FP)."""
    pred = _binarize(logits, threshold)
    tp   = (pred * target).sum()
    fp   = (pred * (1 - target)).sum()
    return (tp / (tp + fp + 1e-6)).item()


def recall_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Pixel-wise recall (sensitivity): TP / (TP + FN)."""
    pred = _binarize(logits, threshold)
    tp   = (pred * target).sum()
    fn   = ((1 - pred) * target).sum()
    return (tp / (tp + fn + 1e-6)).item()


def f1_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Pixel-wise F1 = 2·P·R / (P+R). Equivalent to Dice on binary masks."""
    p = precision_score(logits, target, threshold)
    r = recall_score(logits, target, threshold)
    return 2.0 * p * r / (p + r + 1e-6)


def accuracy_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Pixel-wise accuracy: (TP + TN) / total."""
    pred    = _binarize(logits, threshold)
    correct = (pred == target).float().sum()
    total   = target.numel()
    return (correct / total).item()
