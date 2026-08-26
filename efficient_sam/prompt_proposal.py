"""Unified, image-only prompt proposal interface for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .PGAP import PhasePromptGenerator
from .self_prompting_head import build_dog_log_saliency


@dataclass
class PromptProposal:
    """A padded batch of image-only spatial prompt proposals.

    Coordinates always use ``(x, y)`` in the output map's pixel space.
    Invalid padded slots have ``candidate_valid=False`` and label ``-1`` when
    converted to SAM point prompts. No field is derived from a GT mask.
    """

    dense_logits: Optional[torch.Tensor]
    dense_probs: Optional[torch.Tensor]
    candidate_xy: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_valid: torch.Tensor
    candidate_source: Optional[list[list[str]]] = None
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)

    def validate(self) -> "PromptProposal":
        if self.candidate_xy.ndim != 3 or self.candidate_xy.shape[-1] != 2:
            raise ValueError("candidate_xy must have shape [B,K,2]")
        if self.candidate_scores.shape != self.candidate_xy.shape[:2]:
            raise ValueError("candidate_scores must have shape [B,K]")
        if self.candidate_valid.shape != self.candidate_xy.shape[:2]:
            raise ValueError("candidate_valid must have shape [B,K]")
        if self.candidate_valid.dtype != torch.bool:
            raise TypeError("candidate_valid must be boolean")
        if self.dense_probs is not None:
            if self.dense_probs.ndim != 4 or self.dense_probs.shape[1] != 1:
                raise ValueError("dense_probs must have shape [B,1,H,W]")
            if self.dense_probs.shape[0] != self.candidate_xy.shape[0]:
                raise ValueError("dense_probs batch size must match candidates")
        if self.dense_logits is not None and self.dense_logits.shape != self.dense_probs.shape:
            raise ValueError("dense_logits and dense_probs must have the same shape")
        return self

    def to_point_prompts(self) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.where(
            self.candidate_valid,
            torch.ones_like(self.candidate_scores, dtype=torch.int64),
            -torch.ones_like(self.candidate_scores, dtype=torch.int64),
        )
        coords = torch.where(
            self.candidate_valid.unsqueeze(-1),
            self.candidate_xy,
            torch.zeros_like(self.candidate_xy),
        )
        return coords.unsqueeze(1), labels.unsqueeze(1)


def _stable_score_order(scores: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    """Sort descending by score, then ascending by row-major pixel index."""
    index_order = torch.argsort(flat_indices, stable=True)
    ordered_scores = scores[index_order]
    score_order = torch.argsort(ordered_scores, descending=True, stable=True)
    return index_order[score_order]


@torch.no_grad()
def extract_local_maxima(
    dense_probs: torch.Tensor,
    candidate_k_raw: int = 32,
    nms_radius: float = 3.0,
    score_threshold: float = 0.1,
    source: str = "unknown",
    dense_logits: Optional[torch.Tensor] = None,
    auxiliary: Optional[dict[str, torch.Tensor]] = None,
) -> PromptProposal:
    """Extract deterministic local maxima without any forced fallback point."""
    if dense_probs.ndim != 4 or dense_probs.shape[1] != 1:
        raise ValueError("dense_probs must have shape [B,1,H,W]")
    if candidate_k_raw <= 0:
        raise ValueError("candidate_k_raw must be positive")
    batch, _, height, width = dense_probs.shape
    radius = max(0.0, float(nms_radius))
    pool_radius = max(1, int(round(radius)))
    kernel = 2 * pool_radius + 1
    pooled = F.max_pool2d(dense_probs, kernel_size=kernel, stride=1, padding=pool_radius)
    local = (dense_probs == pooled) & (dense_probs >= float(score_threshold))

    coords_out = dense_probs.new_zeros((batch, candidate_k_raw, 2))
    scores_out = dense_probs.new_zeros((batch, candidate_k_raw))
    valid_out = torch.zeros((batch, candidate_k_raw), dtype=torch.bool, device=dense_probs.device)
    sources: list[list[str]] = []
    for batch_index in range(batch):
        ys, xs = torch.where(local[batch_index, 0])
        if xs.numel() == 0:
            sources.append([source] * candidate_k_raw)
            continue
        scores = dense_probs[batch_index, 0, ys, xs]
        flat = ys * width + xs
        order = _stable_score_order(scores, flat)
        selected_xy = []
        selected_scores = []
        for index in order.tolist():
            xy = torch.stack((xs[index], ys[index])).to(dtype=dense_probs.dtype)
            if selected_xy:
                existing = torch.stack(selected_xy)
                if torch.linalg.vector_norm(existing - xy, dim=1).min().item() < radius:
                    continue
            selected_xy.append(xy)
            selected_scores.append(scores[index])
            if len(selected_xy) >= candidate_k_raw:
                break
        count = len(selected_xy)
        if count:
            coords_out[batch_index, :count] = torch.stack(selected_xy)
            scores_out[batch_index, :count] = torch.stack(selected_scores)
            valid_out[batch_index, :count] = True
        sources.append([source] * candidate_k_raw)

    return PromptProposal(
        dense_logits=dense_logits,
        dense_probs=dense_probs,
        candidate_xy=coords_out,
        candidate_scores=scores_out,
        candidate_valid=valid_out,
        candidate_source=sources,
        auxiliary=dict(auxiliary or {}),
    ).validate()


class PGAPProposalAdapter(nn.Module):
    """Expose PGAP saliency through the no-GT proposal contract."""

    def __init__(
        self,
        generator: Optional[PhasePromptGenerator] = None,
        candidate_k_raw: int = 32,
        nms_radius: float = 3.0,
        score_threshold: float = 0.1,
    ):
        super().__init__()
        self.generator = generator or PhasePromptGenerator(
            top_k=candidate_k_raw,
            min_dist=max(1, int(round(nms_radius))),
            saliency_thr=score_threshold,
            dynamic_top_k=True,
            min_top_k=0,
        )
        self.candidate_k_raw = int(candidate_k_raw)
        self.nms_radius = float(nms_radius)
        self.score_threshold = float(score_threshold)

    def forward(self, images: torch.Tensor) -> PromptProposal:
        saliency = self.generator.get_phase_saliency(images)
        return extract_local_maxima(
            saliency,
            candidate_k_raw=self.candidate_k_raw,
            nms_radius=self.nms_radius,
            score_threshold=self.score_threshold,
            source="pgap",
        )


class DoGLoGProposalAdapter(nn.Module):
    """Expose fixed multi-scale DoG/LoG saliency as image-only proposals."""

    def __init__(
        self,
        candidate_k_raw: int = 32,
        nms_radius: float = 3.0,
        score_threshold: float = 0.1,
        dog_sigmas=None,
        log_sigmas=None,
        truncate: float = 3.0,
    ):
        super().__init__()
        self.candidate_k_raw = int(candidate_k_raw)
        self.nms_radius = float(nms_radius)
        self.score_threshold = float(score_threshold)
        self.dog_sigmas = dog_sigmas
        self.log_sigmas = log_sigmas
        self.truncate = float(truncate)

    def forward(self, images: torch.Tensor) -> PromptProposal:
        saliency = build_dog_log_saliency(
            images,
            dog_sigmas=self.dog_sigmas,
            log_sigmas=self.log_sigmas,
            truncate=self.truncate,
        )
        return extract_local_maxima(
            saliency,
            candidate_k_raw=self.candidate_k_raw,
            nms_radius=self.nms_radius,
            score_threshold=self.score_threshold,
            source="doglog",
        )


class DenseHeadProposalAdapter(nn.Module):
    """Convert a learned spatial head's logits into no-GT point proposals."""

    def __init__(
        self,
        head: nn.Module,
        candidate_k_raw: int = 32,
        nms_radius: float = 3.0,
        score_threshold: float = 0.1,
    ):
        super().__init__()
        self.head = head
        self.candidate_k_raw = int(candidate_k_raw)
        self.nms_radius = float(nms_radius)
        self.score_threshold = float(score_threshold)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> PromptProposal:
        logits = self.head(features)
        if isinstance(logits, dict):
            logits = logits.get("foreground", next(iter(logits.values())))
        if logits.ndim != 4:
            raise ValueError("Learned dense head must return [B,C,H,W]")
        logits = logits[:, :1]
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        return extract_local_maxima(
            probs,
            candidate_k_raw=self.candidate_k_raw,
            nms_radius=self.nms_radius,
            score_threshold=self.score_threshold,
            source="learned_dense_head",
            dense_logits=logits,
        )
