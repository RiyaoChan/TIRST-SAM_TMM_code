"""Lightweight spatial-probe training utilities for Experiment 1."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label, regionprops


class SpatialProbeHead(nn.Module):
    """A small targetness head used with a frozen EfficientSAM encoder."""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, 1, 1),
        )
        nn.init.kaiming_normal_(self.net[0].weight, mode="fan_out", nonlinearity="relu")
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        logits = self.net(features)
        if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


def component_center_heatmap(
    masks: torch.Tensor,
    sigma_min: float = 0.7,
    sigma_max: float = 2.5,
) -> torch.Tensor:
    """Create one adaptive Gaussian center peak for every GT component."""
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError("masks must have shape [B,H,W] or [B,1,H,W]")
    masks_np = (masks.detach().cpu().numpy() > 0.5)
    output = np.zeros(masks_np.shape, dtype=np.float32)
    for batch_index, mask in enumerate(masks_np):
        components = label(mask.astype(np.uint8), connectivity=2)
        for region in regionprops(components):
            center_y, center_x = region.centroid
            sigma = min(float(sigma_max), max(float(sigma_min), math.sqrt(float(region.area)) / 2.0))
            radius = max(1, int(math.ceil(3.0 * sigma)))
            y0 = max(0, int(math.floor(center_y)) - radius)
            y1 = min(mask.shape[0], int(math.floor(center_y)) + radius + 1)
            x0 = max(0, int(math.floor(center_x)) - radius)
            x1 = min(mask.shape[1], int(math.floor(center_x)) + radius + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gaussian = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2))
            gaussian = gaussian / max(float(gaussian.max()), 1e-12)
            output[batch_index, y0:y1, x0:x1] = np.maximum(
                output[batch_index, y0:y1, x0:x1],
                gaussian.astype(np.float32),
            )
    return torch.from_numpy(output).unsqueeze(1).to(device=masks.device, dtype=torch.float32)


def center_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = target * probabilities + (1.0 - target) * (1.0 - probabilities)
    alpha_factor = target * float(alpha) + (1.0 - target) * (1.0 - float(alpha))
    return (alpha_factor * (1.0 - pt).pow(float(gamma)) * bce).mean()


def foreground_dice_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    target = masks.to(device=logits.device, dtype=logits.dtype)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def component_peak_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Penalize GT components that have no high targetness response."""
    if masks.ndim == 4:
        masks = masks[:, 0]
    probabilities = torch.sigmoid(logits[:, 0])
    if probabilities.shape[-2:] != masks.shape[-2:]:
        probabilities = F.interpolate(
            probabilities.unsqueeze(1),
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    losses = []
    masks_np = (masks.detach().cpu().numpy() > 0.5)
    for batch_index, mask in enumerate(masks_np):
        components = label(mask.astype(np.uint8), connectivity=2)
        for component_id in range(1, int(components.max()) + 1):
            component_mask = torch.from_numpy(components == component_id).to(probabilities.device)
            peak = probabilities[batch_index][component_mask].amax()
            losses.append(-torch.log(peak.clamp_min(1e-6)))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def prompt_probe_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    center_weight: float = 1.0,
    foreground_weight: float = 0.5,
    component_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    center_target = component_center_heatmap(masks)
    if center_target.shape[-2:] != logits.shape[-2:]:
        center_target = F.interpolate(center_target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    center = center_focal_loss(logits, center_target)
    foreground = foreground_dice_loss(logits, masks)
    component = component_peak_loss(logits, masks)
    total = float(center_weight) * center + float(foreground_weight) * foreground + float(component_weight) * component
    return {
        "total": total,
        "center_focal": center,
        "foreground_dice": foreground,
        "component_peak": component,
    }
