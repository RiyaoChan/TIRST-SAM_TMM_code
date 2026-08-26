"""Core utilities for candidate-isolated MicroQuery decoding and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from .prompt_proposal import PromptProposal


@dataclass(frozen=True)
class MicroQueryDecodeOutput:
    """Per-candidate mask logits and decoder-predicted quality."""

    mask_logits: torch.Tensor  # [B,Q,H,W]
    quality: torch.Tensor  # [B,Q]


@dataclass(frozen=True)
class MicroQueryHeadOutput:
    """Candidate-level target/no-object and mask-quality predictions."""

    object_logits: torch.Tensor  # [B,Q,2]
    quality_logits: torch.Tensor  # [B,Q]


def extract_candidate_roi_features(
    shallow_features: torch.Tensor,
    neck_features: torch.Tensor,
    candidate_xy: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    input_h: int,
    input_w: int,
    shallow_radius: float = 14.0,
    neck_radius: float = 6.0,
) -> torch.Tensor:
    """Extract frozen local candidate descriptors without using GT.

    The descriptor contains mean-pooled shallow 7x7 and neck 3x3 ROIAlign
    features, normalized x/y coordinates, and the frozen proposal score.
    Invalid candidates are represented by all-zero descriptors.
    """

    if shallow_features.ndim != 4 or neck_features.ndim != 4:
        raise ValueError("feature maps must have shape [B,C,H,W]")
    if candidate_xy.ndim != 3 or candidate_xy.shape[-1] != 2:
        raise ValueError("candidate_xy must have shape [B,Q,2]")
    if candidate_scores.shape != candidate_xy.shape[:2]:
        raise ValueError("candidate_scores must have shape [B,Q]")
    if candidate_valid.shape != candidate_xy.shape[:2]:
        raise ValueError("candidate_valid must have shape [B,Q]")
    if shallow_features.shape[0] != candidate_xy.shape[0] or neck_features.shape[0] != candidate_xy.shape[0]:
        raise ValueError("feature and candidate batch sizes must match")
    if int(input_h) <= 0 or int(input_w) <= 0:
        raise ValueError("input dimensions must be positive")

    batch_size, query_count = candidate_xy.shape[:2]
    flat_xy = candidate_xy.reshape(-1, 2)
    batch_indices = (
        torch.arange(batch_size, device=candidate_xy.device, dtype=candidate_xy.dtype)
        .view(-1, 1)
        .expand(-1, query_count)
        .reshape(-1, 1)
    )

    def make_boxes(radius: float) -> torch.Tensor:
        x = flat_xy[:, 0].clamp(0.0, float(input_w - 1))
        y = flat_xy[:, 1].clamp(0.0, float(input_h - 1))
        x1 = (x - float(radius)).clamp(0.0, float(input_w - 1))
        y1 = (y - float(radius)).clamp(0.0, float(input_h - 1))
        x2 = (x + float(radius)).clamp(0.0, float(input_w - 1))
        y2 = (y + float(radius)).clamp(0.0, float(input_h - 1))
        return torch.cat((batch_indices, torch.stack((x1, y1, x2, y2), dim=1)), dim=1)

    shallow_roi = roi_align(
        shallow_features,
        make_boxes(shallow_radius),
        output_size=(7, 7),
        spatial_scale=float(shallow_features.shape[-1]) / float(input_w),
        sampling_ratio=2,
        aligned=True,
    ).mean(dim=(-2, -1))
    neck_roi = roi_align(
        neck_features,
        make_boxes(neck_radius),
        output_size=(3, 3),
        spatial_scale=float(neck_features.shape[-1]) / float(input_w),
        sampling_ratio=2,
        aligned=True,
    ).mean(dim=(-2, -1))
    normalized_xy = torch.stack(
        (
            flat_xy[:, 0] / float(max(1, input_w - 1)),
            flat_xy[:, 1] / float(max(1, input_h - 1)),
        ),
        dim=1,
    )
    descriptor = torch.cat(
        (shallow_roi, neck_roi, normalized_xy, candidate_scores.reshape(-1, 1)), dim=1
    ).reshape(batch_size, query_count, -1)
    return torch.where(candidate_valid.unsqueeze(-1), descriptor, torch.zeros_like(descriptor))


class CandidateROIEncoder(nn.Module):
    """Lightweight projection from frozen ROI descriptors to candidate tokens."""

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

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.input_dim:
            raise ValueError(f"descriptors must have shape [B,Q,{self.input_dim}]")
        return self.projection(descriptors)


class MicroQueryHead(nn.Module):
    """Minimal M2-S1 candidate objectness and quality heads."""

    def __init__(self, input_dim: int = 451, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.roi_encoder = CandidateROIEncoder(input_dim, hidden_dim, dropout)
        self.object_head = nn.Linear(int(hidden_dim), 2)
        self.quality_head = nn.Linear(int(hidden_dim), 1)

    def forward(
        self, descriptors: torch.Tensor, candidate_valid: torch.Tensor
    ) -> MicroQueryHeadOutput:
        if candidate_valid.shape != descriptors.shape[:2]:
            raise ValueError("candidate_valid must have shape [B,Q]")
        tokens = self.roi_encoder(descriptors)
        object_logits = self.object_head(tokens)
        quality_logits = self.quality_head(tokens).squeeze(-1)
        # Invalid entries never become deployable queries.  Finite logits keep
        # losses and serialized outputs robust for entirely empty candidate sets.
        invalid_object = object_logits.new_tensor([0.0, -30.0])
        object_logits = torch.where(
            candidate_valid.unsqueeze(-1), object_logits, invalid_object
        )
        quality_logits = torch.where(
            candidate_valid, quality_logits, quality_logits.new_full(quality_logits.shape, -30.0)
        )
        return MicroQueryHeadOutput(object_logits, quality_logits)


def proposal_to_point_queries(
    proposal: PromptProposal,
    budget: int,
    *,
    independent: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a frozen proposal to one mixed query or Q isolated queries."""

    proposal.validate()
    budget = int(budget)
    if budget <= 0 or budget > proposal.candidate_xy.shape[1]:
        raise ValueError("budget must be in [1, proposal candidate count]")
    coords = proposal.candidate_xy[:, :budget]
    valid = proposal.candidate_valid[:, :budget]
    labels = torch.where(
        valid,
        torch.ones_like(valid, dtype=torch.int64),
        -torch.ones_like(valid, dtype=torch.int64),
    )
    coords = torch.where(valid.unsqueeze(-1), coords, torch.zeros_like(coords))
    if independent:
        return coords.unsqueeze(2), labels.unsqueeze(2)
    return coords.unsqueeze(1), labels.unsqueeze(1)


def _rescale_boxes(model, boxes: torch.Tensor, input_h: int, input_w: int) -> torch.Tensor:
    scale = boxes.new_tensor(
        [
            model.prompt_encoder.input_image_size[1] / max(1, input_w),
            model.prompt_encoder.input_image_size[0] / max(1, input_h),
            model.prompt_encoder.input_image_size[1] / max(1, input_w),
            model.prompt_encoder.input_image_size[0] / max(1, input_h),
        ]
    )
    return boxes * scale.view(1, 1, 4)


def decode_prompt_queries(
    model,
    image_embeddings: torch.Tensor,
    interm_embeddings: torch.Tensor,
    *,
    input_h: int,
    input_w: int,
    output_h: int,
    output_w: int,
    points: Optional[torch.Tensor] = None,
    point_labels: Optional[torch.Tensor] = None,
    boxes: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    hq_token_only: bool = False,
    chunk_size: int = 0,
) -> MicroQueryDecodeOutput:
    """Decode arbitrary point/box/mask query groups while sharing image embeddings.

    Prompt tensors use shapes ``[B,Q,N,2]``, ``[B,Q,N]``, ``[B,Q,4]`` and
    ``[B,Q,1,Hm,Wm]``. At least one prompt tensor must define ``B,Q``. A
    zero-length point sequence is a valid null prompt and preserves batch size.
    """

    candidates = [item for item in (points, boxes, masks) if item is not None]
    if not candidates:
        raise ValueError("At least one of points, boxes, or masks is required")
    batch_size = int(candidates[0].shape[0])
    query_count = int(candidates[0].shape[1])
    if batch_size != int(image_embeddings.shape[0]):
        raise ValueError("Prompt batch size must match image embeddings")
    for item in candidates[1:]:
        if item.shape[:2] != (batch_size, query_count):
            raise ValueError("All prompt tensors must share [B,Q]")
    if points is not None:
        if point_labels is None or point_labels.shape != points.shape[:-1]:
            raise ValueError("point_labels must match points without the coordinate axis")
        points = model.get_rescaled_pts(points, input_h, input_w)
    elif point_labels is not None:
        raise ValueError("point_labels cannot be supplied without points")
    if boxes is not None:
        boxes = _rescale_boxes(model, boxes, input_h, input_w)

    chunk_size = query_count if int(chunk_size) <= 0 else min(int(chunk_size), query_count)
    output_masks = []
    output_quality = []
    dense_pe = model.prompt_encoder.get_dense_pe()
    for start in range(0, query_count, chunk_size):
        end = min(query_count, start + chunk_size)
        width = end - start
        flat_points = None
        flat_labels = None
        if points is not None:
            flat_points = points[:, start:end].reshape(batch_size * width, points.shape[2], 2)
            flat_labels = point_labels[:, start:end].reshape(batch_size * width, point_labels.shape[2])
        flat_boxes = (
            boxes[:, start:end].reshape(batch_size * width, 4) if boxes is not None else None
        )
        flat_masks = None
        if masks is not None:
            flat_masks = masks[:, start:end].reshape(
                batch_size * width, *masks.shape[2:]
            )
        sparse, dense = model.prompt_encoder(
            points=(flat_points, flat_labels) if flat_points is not None else None,
            boxes=flat_boxes,
            masks=flat_masks,
            text_embeds=None,
        )
        repeated_image = image_embeddings.repeat_interleave(width, dim=0)
        repeated_interm = interm_embeddings.repeat_interleave(width, dim=0)
        low_res, quality = model.mask_decoder(
            repeated_image,
            dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            hq_token_only=bool(hq_token_only),
            interm_embeddings=repeated_interm,
        )
        resized = F.interpolate(low_res, (output_h, output_w), mode="bicubic")
        output_masks.append(resized.reshape(batch_size, width, output_h, output_w))
        output_quality.append(quality.reshape(batch_size, width))
    return MicroQueryDecodeOutput(
        mask_logits=torch.cat(output_masks, dim=1),
        quality=torch.cat(output_quality, dim=1),
    )


def aggregate_query_probabilities(
    probabilities: torch.Tensor,
    valid: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None,
    top_n: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate independent masks with a weighted pixel-wise maximum."""

    if probabilities.ndim != 4:
        raise ValueError("probabilities must have shape [B,Q,H,W]")
    if valid.shape != probabilities.shape[:2]:
        raise ValueError("valid must have shape [B,Q]")
    if probabilities.shape[1] == 0:
        return probabilities.new_zeros(
            probabilities.shape[0], probabilities.shape[2], probabilities.shape[3]
        )
    if weights is None:
        weights = torch.ones_like(valid, dtype=probabilities.dtype)
    if weights.shape != valid.shape:
        raise ValueError("weights must have shape [B,Q]")
    weights = torch.where(valid, weights.to(probabilities.dtype).clamp(0.0, 1.0), 0.0)
    if top_n is not None:
        top_n = max(0, min(int(top_n), int(weights.shape[1])))
        keep = torch.zeros_like(valid)
        if top_n > 0:
            ranking = torch.argsort(weights, dim=1, descending=True, stable=True)
            keep.scatter_(1, ranking[:, :top_n], True)
        weights = torch.where(keep & valid, weights, torch.zeros_like(weights))
    weighted = probabilities * weights[:, :, None, None]
    return weighted.amax(dim=1)
