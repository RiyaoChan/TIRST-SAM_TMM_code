"""End-to-end MicroQuery utilities used by the matched full-training study.

The deployable forward path consumes image features and cached image-only
candidate data only. Ground truth is deliberately isolated in the assignment
and loss helpers in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure
from torch import nn


@dataclass(frozen=True)
class EndToEndHeadOutput:
    hidden: torch.Tensor
    object_logits: torch.Tensor
    candidate_token: torch.Tensor


@dataclass(frozen=True)
class CandidateAssignment:
    component_map: np.ndarray
    component_ids: np.ndarray
    semantic_labels: np.ndarray
    valid: np.ndarray
    multi_match_count: int


class EndToEndMicroQueryHead(nn.Module):
    """Shared 451->256 MicroQuery head for C1, F1 and F2.

    The candidate token branch is instantiated in every variant so their
    trainable parameter counts and initial states are exactly matched. F1/C1
    simply do not consume the returned token in the decoder.
    """

    def __init__(self, input_dim: int = 451, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.object_head = nn.Linear(self.hidden_dim, 2)
        self.token_norm = nn.LayerNorm(self.hidden_dim)
        self.token_scale = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))

    def forward(self, descriptors: torch.Tensor, candidate_valid: torch.Tensor) -> EndToEndHeadOutput:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.input_dim:
            raise ValueError(f"descriptors must have shape [B,K,{self.input_dim}]")
        if candidate_valid.shape != descriptors.shape[:2]:
            raise ValueError("candidate_valid must have shape [B,K]")
        hidden = self.projection(descriptors)
        hidden = torch.where(candidate_valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        object_logits = self.object_head(hidden)
        invalid_logits = object_logits.new_tensor([0.0, -30.0])
        object_logits = torch.where(candidate_valid.unsqueeze(-1), object_logits, invalid_logits)
        token = self.token_scale.to(hidden.dtype) * self.token_norm(hidden)
        token = torch.where(candidate_valid.unsqueeze(-1), token, torch.zeros_like(token))
        return EndToEndHeadOutput(hidden=hidden, object_logits=object_logits, candidate_token=token)


def append_candidate_sparse_token(
    sparse_point: torch.Tensor,
    candidate_token: torch.Tensor,
    candidate_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Append one candidate token after each native point-prompt sequence."""

    if sparse_point.ndim != 3 or candidate_token.ndim != 2:
        raise ValueError("expected sparse_point [N,T,C] and candidate_token [N,C]")
    if sparse_point.shape[0] != candidate_token.shape[0] or sparse_point.shape[2] != candidate_token.shape[1]:
        raise ValueError("candidate token batch/channel dimensions must match sparse prompts")
    token = candidate_token
    if candidate_valid is not None:
        valid = candidate_valid.reshape(-1)
        if valid.shape[0] != token.shape[0]:
            raise ValueError("candidate_valid must contain one entry per flattened token")
        token = torch.where(valid.unsqueeze(-1), token, torch.zeros_like(token))
    return torch.cat((sparse_point, token.unsqueeze(1)), dim=1)


def gate_temperature(epoch: int) -> float:
    epoch = max(1, int(epoch))
    if epoch >= 30:
        return 1.0
    return 2.0 - float(epoch - 1) / 29.0


def gate_warmup_rho(epoch: int) -> float:
    return max(0.0, 1.0 - float(max(0, int(epoch))) / 20.0)


def soft_gate_schedule(
    object_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    epoch: int,
    *,
    force_all_one: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return raw gate, effective gate, warm-up rho and temperature."""

    if object_logits.shape[:-1] != candidate_valid.shape or object_logits.shape[-1] != 2:
        raise ValueError("object_logits must have shape [B,K,2]")
    temperature = gate_temperature(epoch)
    raw = torch.softmax(object_logits / temperature, dim=-1)[..., 1]
    raw = torch.where(candidate_valid, raw, torch.zeros_like(raw))
    rho = gate_warmup_rho(epoch)
    if force_all_one:
        effective = candidate_valid.to(raw.dtype)
    else:
        effective = rho + (1.0 - rho) * raw
        effective = torch.where(candidate_valid, effective, torch.zeros_like(effective))
    return raw, effective, rho, temperature


def aggregate_soft_gated_max(
    mask_logits: torch.Tensor,
    gates: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> torch.Tensor:
    """Differentiable pixel-wise maximum of gated independent query masks."""

    if mask_logits.ndim != 4 or gates.shape != mask_logits.shape[:2]:
        raise ValueError("mask_logits must be [B,K,H,W] and gates [B,K]")
    if candidate_valid.shape != gates.shape:
        raise ValueError("candidate_valid must match gates")
    safe_gates = torch.where(candidate_valid, gates.to(mask_logits.dtype).clamp(0.0, 1.0), 0.0)
    if mask_logits.shape[1] == 0:
        return mask_logits.new_zeros(mask_logits.shape[0], *mask_logits.shape[-2:])
    return (torch.sigmoid(mask_logits) * safe_gates[..., None, None]).amax(dim=1)


def assign_candidates_to_components(
    candidate_xy: np.ndarray | torch.Tensor,
    candidate_valid: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    *,
    dilation_radius: int = 2,
    centroid_radius: float = 3.0,
) -> CandidateAssignment:
    """Assign valid candidates to connected GT components without changing xy."""

    xy = np.asarray(candidate_xy.detach().cpu() if torch.is_tensor(candidate_xy) else candidate_xy, dtype=np.float32)
    valid = np.asarray(candidate_valid.detach().cpu() if torch.is_tensor(candidate_valid) else candidate_valid, dtype=bool)
    mask = np.asarray(gt_mask.detach().cpu() if torch.is_tensor(gt_mask) else gt_mask) > 0.5
    if xy.ndim != 2 or xy.shape[-1] != 2 or valid.shape != xy.shape[:1]:
        raise ValueError("candidate_xy must be [K,2] and candidate_valid [K]")
    if mask.ndim != 2:
        raise ValueError("gt_mask must be 2-D")
    component_map = measure.label(mask.astype(np.uint8), connectivity=2).astype(np.int32)
    regions = list(measure.regionprops(component_map))
    ids = np.full((xy.shape[0],), -1, dtype=np.int64)
    semantic = np.zeros((xy.shape[0],), dtype=bool)
    if not regions:
        return CandidateAssignment(component_map, ids, semantic, valid.copy(), 0)
    structure = ndimage.generate_binary_structure(2, 2)
    dilated = {
        int(region.label): ndimage.binary_dilation(
            component_map == int(region.label), structure=structure, iterations=max(0, int(dilation_radius))
        )
        for region in regions
    }
    centroids_xy = {
        int(region.label): np.asarray([region.centroid[1], region.centroid[0]], dtype=np.float32)
        for region in regions
    }
    height, width = mask.shape
    multi_match_count = 0
    for index, (point, is_valid) in enumerate(zip(xy, valid)):
        if not is_valid:
            continue
        x, y = float(point[0]), float(point[1])
        px = int(np.clip(np.rint(x), 0, width - 1))
        py = int(np.clip(np.rint(y), 0, height - 1))
        matches: list[tuple[float, int]] = []
        for region in regions:
            label_id = int(region.label)
            distance = float(np.linalg.norm(point - centroids_xy[label_id]))
            if bool(dilated[label_id][py, px]) or distance <= float(centroid_radius):
                matches.append((distance, label_id))
        if len(matches) > 1:
            multi_match_count += 1
        if matches:
            _, label_id = min(matches, key=lambda item: (item[0], item[1]))
            ids[index] = label_id
            semantic[index] = True
    return CandidateAssignment(component_map, ids, semantic, valid.copy(), multi_match_count)


def build_query_targets(
    component_map: np.ndarray | torch.Tensor,
    component_ids: np.ndarray | torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    component = torch.as_tensor(component_map, dtype=torch.int64, device=device)
    ids = torch.as_tensor(component_ids, dtype=torch.int64, device=device)
    if component.ndim != 2 or ids.ndim != 1:
        raise ValueError("component_map must be [H,W] and component_ids [K]")
    return (component.unsqueeze(0) == ids[:, None, None]) & (ids[:, None, None] > 0)


def build_covered_gt_mask(
    component_map: np.ndarray | torch.Tensor,
    component_ids: np.ndarray | torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    query_targets = build_query_targets(component_map, component_ids, device=device)
    if query_targets.shape[0] == 0:
        return query_targets.new_zeros(query_targets.shape[-2:])
    return query_targets.any(dim=0)


def component_survival_loss(
    gates: torch.Tensor,
    component_ids: torch.Tensor,
    semantic_labels: torch.Tensor,
    candidate_valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage at least one candidate gate to survive per covered component."""

    if not (gates.shape == component_ids.shape == semantic_labels.shape == candidate_valid.shape):
        raise ValueError("all component-survival tensors must have shape [B,K]")
    losses: list[torch.Tensor] = []
    for batch_index in range(gates.shape[0]):
        ids = torch.unique(component_ids[batch_index][semantic_labels[batch_index] & candidate_valid[batch_index]])
        for component_id in ids:
            if int(component_id) <= 0:
                continue
            members = (
                (component_ids[batch_index] == component_id)
                & semantic_labels[batch_index]
                & candidate_valid[batch_index]
            )
            survive = 1.0 - torch.prod(1.0 - gates[batch_index][members].clamp(0.0, 1.0))
            losses.append(-torch.log(survive + float(eps)))
    return torch.stack(losses).mean() if losses else gates.sum() * 0.0


def _probability_segmentation_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    bce = F.binary_cross_entropy(probability, target)
    reduce_dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=reduce_dims)
    denominator = probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + dice.mean()


def _query_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)


def microquery_full_loss(
    *,
    variant: str,
    final_probability: torch.Tensor,
    full_target: torch.Tensor,
    covered_target: torch.Tensor,
    query_logits: Optional[torch.Tensor] = None,
    query_targets: Optional[torch.Tensor] = None,
    semantic_labels: Optional[torch.Tensor] = None,
    candidate_valid: Optional[torch.Tensor] = None,
    object_logits: Optional[torch.Tensor] = None,
    raw_gates: Optional[torch.Tensor] = None,
    component_ids: Optional[torch.Tensor] = None,
    iou_predictions: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    candidate_tokens: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Compute the exact matched-control/full MicroQuery objective in FP32."""

    if variant not in {"c0_one_query", "c1_independent_aux", "f1_soft_gate", "f2_gate_token"}:
        raise ValueError(f"unknown variant: {variant}")
    full = _probability_segmentation_loss(final_probability, full_target)
    covered = _probability_segmentation_loss(final_probability, covered_target)
    zero = final_probability.sum() * 0.0
    pieces = {
        "full": full,
        "covered": covered,
        "positive_query": zero,
        "background_query": zero,
        "objectness": zero,
        "survival": zero,
        "iou": zero,
        "token": zero,
    }
    if variant == "c0_one_query":
        pieces["total"] = full + 0.25 * covered
        return pieces

    required = (query_logits, query_targets, semantic_labels, candidate_valid, object_logits, raw_gates, component_ids, iou_predictions)
    if any(value is None for value in required):
        raise ValueError("C1/F1/F2 loss requires query, semantic, gate, component and IoU tensors")
    assert query_logits is not None and query_targets is not None
    assert semantic_labels is not None and candidate_valid is not None
    assert object_logits is not None and raw_gates is not None
    assert component_ids is not None and iou_predictions is not None
    positive = candidate_valid & semantic_labels.bool()
    negative = candidate_valid & ~semantic_labels.bool()
    dice_each = _query_dice_loss(query_logits, query_targets)
    bce_each = F.binary_cross_entropy_with_logits(
        query_logits.float(), query_targets.float(), reduction="none"
    ).mean(dim=(-2, -1))
    pieces["positive_query"] = (
        (bce_each[positive] + dice_each[positive]).mean() if positive.any() else query_logits.sum() * 0.0
    )
    if negative.any():
        probability = torch.sigmoid(query_logits.float())
        focal = (probability.square() * F.binary_cross_entropy_with_logits(
            query_logits.float(), torch.zeros_like(query_logits, dtype=torch.float32), reduction="none"
        )).mean(dim=(-2, -1))
        false_area = probability.mean(dim=(-2, -1))
        pieces["background_query"] = (focal[negative] + 0.1 * false_area[negative]).mean()
    else:
        pieces["background_query"] = query_logits.sum() * 0.0
    if candidate_valid.any():
        pieces["objectness"] = F.cross_entropy(
            object_logits[candidate_valid].float(),
            semantic_labels[candidate_valid].long(),
            weight=None if class_weights is None else class_weights.float(),
        )
    else:
        pieces["objectness"] = object_logits.sum() * 0.0
    pieces["survival"] = component_survival_loss(
        raw_gates.float(), component_ids, semantic_labels.bool(), candidate_valid
    )
    with torch.no_grad():
        probability = torch.sigmoid(query_logits.float())
        intersection = (probability * query_targets.float()).sum(dim=(-2, -1))
        union = (probability + query_targets.float() - probability * query_targets.float()).sum(dim=(-2, -1))
        iou_target = torch.where(semantic_labels.bool(), intersection / union.clamp_min(1e-6), 0.0)
    pieces["iou"] = (
        F.smooth_l1_loss(iou_predictions[candidate_valid].float(), iou_target[candidate_valid])
        if candidate_valid.any()
        else iou_predictions.sum() * 0.0
    )
    if variant == "f2_gate_token":
        if candidate_tokens is None:
            raise ValueError("F2 requires candidate_tokens")
        pieces["token"] = (
            candidate_tokens[candidate_valid].float().square().sum(dim=-1).mean()
            if candidate_valid.any()
            else candidate_tokens.sum() * 0.0
        )
    total = (
        full
        + 0.25 * covered
        + 0.50 * pieces["positive_query"]
        + 0.25 * pieces["background_query"]
        + 0.50 * pieces["objectness"]
        + 0.20 * pieces["survival"]
        + 0.10 * pieces["iou"]
        + (1e-4 * pieces["token"] if variant == "f2_gate_token" else zero)
    )
    pieces["total"] = total
    return pieces
