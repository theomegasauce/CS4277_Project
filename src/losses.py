"""
losses.py — Loss functions and auxiliary target generation for RoadSegNet.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize

# ── Sobel kernels (registered once, moved to device on first call) ───────────

_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


# ── Auxiliary target builders ────────────────────────────────────────────────

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


# ── Loss functions ───────────────────────────────────────────────────────────

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
    """
    Weighted multi-head loss.

    Returns (loss_tensor, component_dict) where component_dict has scalar
    values for logging.
    """
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
