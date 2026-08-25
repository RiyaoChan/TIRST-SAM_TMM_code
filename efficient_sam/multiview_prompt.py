"""Deterministic multi-view prompt fusion and image-only reliability gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .prompt_proposal import PromptProposal
from .self_prompting_head import build_dog_log_saliency


DEFAULT_VIEWS = ("identity", "hflip", "vflip", "local_contrast", "doglog")


def apply_view(images: torch.Tensor, view: str) -> torch.Tensor:
    """Apply one deterministic, size-preserving view to an image batch."""
    if view == "identity":
        return images
    if view == "hflip":
        return torch.flip(images, dims=(-1,))
    if view == "vflip":
        return torch.flip(images, dims=(-2,))
    if view == "local_contrast":
        local_mean = F.avg_pool2d(images, kernel_size=9, stride=1, padding=4)
        residual = images - local_mean
        scale = residual.flatten(2).abs().amax(dim=2, keepdim=True).clamp_min(1e-6)
        residual = residual / scale.unsqueeze(-1)
        return (0.5 * images + 0.5 * (residual + 1.0) / 2.0).clamp(0.0, 1.0)
    if view == "doglog":
        saliency = build_dog_log_saliency(images)
        if saliency.shape[1] != images.shape[1]:
            saliency = saliency.expand(-1, images.shape[1], -1, -1)
        return (0.5 * images + 0.5 * saliency).clamp(0.0, 1.0)
    raise ValueError(f"Unknown multi-view transform: {view}")


def inverse_warp_map(values: torch.Tensor, view: str) -> torch.Tensor:
    """Map a view-aligned spatial tensor back to identity coordinates."""
    if view in {"identity", "local_contrast", "doglog"}:
        return values
    if view == "hflip":
        return torch.flip(values, dims=(-1,))
    if view == "vflip":
        return torch.flip(values, dims=(-2,))
    raise ValueError(f"Unknown multi-view transform: {view}")


def inverse_warp_xy(
    xy: torch.Tensor,
    view: str,
    width: int,
    height: int,
) -> torch.Tensor:
    """Map ``(x,y)`` coordinates from a transformed view to the input image."""
    result = xy.clone()
    if view in {"identity", "local_contrast", "doglog"}:
        return result
    if view == "hflip":
        result[..., 0] = int(width) - 1 - result[..., 0]
        return result
    if view == "vflip":
        result[..., 1] = int(height) - 1 - result[..., 1]
        return result
    raise ValueError(f"Unknown multi-view transform: {view}")


def make_view_batch(
    images: torch.Tensor,
    views: Sequence[str] = DEFAULT_VIEWS,
) -> torch.Tensor:
    """Stack shared-weight views as ``[view0 batch, view1 batch, ...]``."""
    if not views:
        raise ValueError("At least one view is required")
    return torch.cat([apply_view(images, view) for view in views], dim=0)


@dataclass(frozen=True)
class CandidateObservation:
    xy: tuple[float, float]
    score: float
    view_index: int
    rank: int


@dataclass
class CandidateCluster:
    center_xy: tuple[float, float]
    mean_score: float
    max_score: float
    score_variance: float
    support_count: int
    support_fraction: float
    center_dispersion: float
    rank_mean: float
    rank_variance: float
    mean_map_value: float = 0.0
    variance_map_value: float = 0.0
    local_contrast: float = 0.0
    observations: tuple[CandidateObservation, ...] = ()


def _cluster_from_observations(
    observations: list[CandidateObservation],
    num_views: int,
    mean_map: torch.Tensor | None,
    variance_map: torch.Tensor | None,
) -> CandidateCluster:
    scores = torch.tensor([item.score for item in observations], dtype=torch.float64)
    coords = torch.tensor([item.xy for item in observations], dtype=torch.float64)
    ranks = torch.tensor([item.rank for item in observations], dtype=torch.float64)
    weights = scores.clamp_min(1e-12)
    center = (coords * weights[:, None]).sum(dim=0) / weights.sum()
    dispersion = torch.linalg.vector_norm(coords - center, dim=1).mean()
    center_x = int(round(float(center[0])))
    center_y = int(round(float(center[1])))

    mean_value = 0.0
    variance_value = 0.0
    local_contrast = 0.0
    if mean_map is not None:
        height, width = mean_map.shape[-2:]
        center_x = min(max(center_x, 0), width - 1)
        center_y = min(max(center_y, 0), height - 1)
        mean_value = float(mean_map[..., center_y, center_x].mean())
        y0, y1 = max(0, center_y - 2), min(height, center_y + 3)
        x0, x1 = max(0, center_x - 2), min(width, center_x + 3)
        local_contrast = mean_value - float(mean_map[..., y0:y1, x0:x1].mean())
    if variance_map is not None:
        height, width = variance_map.shape[-2:]
        center_x = min(max(center_x, 0), width - 1)
        center_y = min(max(center_y, 0), height - 1)
        variance_value = float(variance_map[..., center_y, center_x].mean())

    return CandidateCluster(
        center_xy=(float(center[0]), float(center[1])),
        mean_score=float(scores.mean()),
        max_score=float(scores.max()),
        score_variance=float(scores.var(unbiased=False)),
        support_count=len({item.view_index for item in observations}),
        support_fraction=len({item.view_index for item in observations}) / float(num_views),
        center_dispersion=float(dispersion),
        rank_mean=float(ranks.mean()),
        rank_variance=float(ranks.var(unbiased=False)),
        mean_map_value=mean_value,
        variance_map_value=variance_value,
        local_contrast=local_contrast,
        observations=tuple(observations),
    )


def cluster_candidates(
    candidate_xy: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_valid: torch.Tensor,
    radius: float = 3.0,
    mean_map: torch.Tensor | None = None,
    variance_map: torch.Tensor | None = None,
) -> list[CandidateCluster]:
    """Cluster ``[M,K]`` view candidates without GT or input-order effects."""
    if candidate_xy.ndim != 3 or candidate_xy.shape[-1] != 2:
        raise ValueError("candidate_xy must have shape [M,K,2]")
    if candidate_scores.shape != candidate_xy.shape[:2]:
        raise ValueError("candidate_scores must have shape [M,K]")
    if candidate_valid.shape != candidate_xy.shape[:2]:
        raise ValueError("candidate_valid must have shape [M,K]")
    num_views, max_candidates = candidate_scores.shape
    observations: list[CandidateObservation] = []
    for view_index in range(num_views):
        valid_indices = torch.where(candidate_valid[view_index])[0]
        view_items = []
        for candidate_index in valid_indices.tolist():
            x, y = candidate_xy[view_index, candidate_index].detach().cpu().tolist()
            score = float(candidate_scores[view_index, candidate_index].detach().cpu())
            view_items.append((score, float(x), float(y), candidate_index))
        view_items.sort(key=lambda item: (-item[0], item[2], item[1], item[3]))
        for rank, (score, x, y, _) in enumerate(view_items, start=1):
            observations.append(CandidateObservation((x, y), score, view_index, rank))
    observations.sort(
        key=lambda item: (-item.score, item.xy[1], item.xy[0], item.view_index, item.rank)
    )

    grouped: list[list[CandidateObservation]] = []
    for observation in observations:
        best_index = None
        best_distance = math.inf
        for group_index, group in enumerate(grouped):
            if observation.view_index in {item.view_index for item in group}:
                continue
            cluster = _cluster_from_observations(group, num_views, None, None)
            distance = math.dist(observation.xy, cluster.center_xy)
            if distance <= float(radius) and (distance, group_index) < (best_distance, best_index or 0):
                best_index = group_index
                best_distance = distance
        if best_index is None:
            grouped.append([observation])
        else:
            grouped[best_index].append(observation)

    clusters = [
        _cluster_from_observations(group, num_views, mean_map, variance_map)
        for group in grouped
    ]
    clusters.sort(
        key=lambda item: (
            -(0.5 * item.mean_score + 0.5 * item.max_score),
            item.center_xy[1],
            item.center_xy[0],
        )
    )
    return clusters


def rule_reliability(
    cluster: CandidateCluster,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> float:
    return float(
        cluster.mean_score
        * cluster.support_fraction ** float(alpha)
        * math.exp(-float(beta) * cluster.center_dispersion)
        * math.exp(-float(gamma) * cluster.score_variance)
    )


def clusters_to_proposal(
    clusters_per_image: Sequence[Sequence[CandidateCluster]],
    dense_mean: torch.Tensor | None,
    max_candidates: int = 32,
    gate: str = "none",
    min_support: int = 3,
    max_dispersion: float = 2.0,
    reliability_threshold: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> PromptProposal:
    """Create A2 mean/max or A3 abstaining proposals from clusters."""
    if gate not in {"none", "rule"}:
        raise ValueError("gate must be 'none' or 'rule'")
    batch = len(clusters_per_image)
    device = dense_mean.device if dense_mean is not None else torch.device("cpu")
    dtype = dense_mean.dtype if dense_mean is not None else torch.float32
    coords = torch.zeros((batch, max_candidates, 2), device=device, dtype=dtype)
    scores = torch.zeros((batch, max_candidates), device=device, dtype=dtype)
    valid = torch.zeros((batch, max_candidates), device=device, dtype=torch.bool)
    support = torch.zeros((batch, max_candidates), device=device, dtype=dtype)
    dispersion = torch.zeros((batch, max_candidates), device=device, dtype=dtype)
    sources: list[list[str]] = []

    for batch_index, clusters in enumerate(clusters_per_image):
        selected = []
        for cluster in clusters:
            reliability = rule_reliability(cluster, alpha=alpha, beta=beta, gamma=gamma)
            if gate == "rule" and (
                cluster.support_count < int(min_support)
                or cluster.center_dispersion > float(max_dispersion)
                or reliability < float(reliability_threshold)
            ):
                continue
            fused_score = (
                reliability
                if gate == "rule"
                else 0.5 * cluster.mean_score + 0.5 * cluster.max_score
            )
            selected.append((fused_score, cluster))
        selected.sort(key=lambda item: (-item[0], item[1].center_xy[1], item[1].center_xy[0]))
        selected = selected[:max_candidates]
        for output_index, (score, cluster) in enumerate(selected):
            coords[batch_index, output_index] = torch.tensor(cluster.center_xy, device=device, dtype=dtype)
            scores[batch_index, output_index] = score
            valid[batch_index, output_index] = True
            support[batch_index, output_index] = cluster.support_fraction
            dispersion[batch_index, output_index] = cluster.center_dispersion
        sources.append([f"multiview_{gate}"] * max_candidates)

    return PromptProposal(
        dense_logits=None,
        dense_probs=dense_mean,
        candidate_xy=coords,
        candidate_scores=scores,
        candidate_valid=valid,
        candidate_source=sources,
        auxiliary={"support_fraction": support, "center_dispersion": dispersion},
    ).validate()


@torch.no_grad()
def multiview_propose(
    images: torch.Tensor,
    generator: Callable[[torch.Tensor], PromptProposal],
    views: Sequence[str] = DEFAULT_VIEWS,
    cluster_radius: float = 3.0,
    max_candidates: int = 32,
    gate: str = "none",
    min_support: int = 3,
    max_dispersion: float = 2.0,
    reliability_threshold: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 1.0,
) -> tuple[PromptProposal, list[list[CandidateCluster]], torch.Tensor, torch.Tensor]:
    """Run one shared generator over stacked views and fuse in image space."""
    batch, _, height, width = images.shape
    view_batch = make_view_batch(images, views)
    raw = generator(view_batch).validate()
    expected_batch = batch * len(views)
    if raw.candidate_xy.shape[0] != expected_batch:
        raise ValueError("Generator output batch does not match stacked views")
    if raw.dense_probs is None:
        raise ValueError("Multi-view fusion requires generator dense_probs")

    candidates_xy = raw.candidate_xy.reshape(len(views), batch, -1, 2).clone()
    candidates_scores = raw.candidate_scores.reshape(len(views), batch, -1)
    candidates_valid = raw.candidate_valid.reshape(len(views), batch, -1)
    dense_by_view = raw.dense_probs.reshape(len(views), batch, 1, height, width)
    inverse_maps = []
    for view_index, view in enumerate(views):
        candidates_xy[view_index] = inverse_warp_xy(
            candidates_xy[view_index], view, width=width, height=height
        )
        inverse_maps.append(inverse_warp_map(dense_by_view[view_index], view))
    inverse_maps_tensor = torch.stack(inverse_maps, dim=0)
    mean_map = inverse_maps_tensor.mean(dim=0)
    variance_map = inverse_maps_tensor.var(dim=0, unbiased=False)

    clusters_per_image = []
    for batch_index in range(batch):
        clusters_per_image.append(
            cluster_candidates(
                candidates_xy[:, batch_index],
                candidates_scores[:, batch_index],
                candidates_valid[:, batch_index],
                radius=cluster_radius,
                mean_map=mean_map[batch_index],
                variance_map=variance_map[batch_index],
            )
        )
    proposal = clusters_to_proposal(
        clusters_per_image,
        dense_mean=mean_map,
        max_candidates=max_candidates,
        gate=gate,
        min_support=min_support,
        max_dispersion=max_dispersion,
        reliability_threshold=reliability_threshold,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    proposal.auxiliary["view_maps"] = inverse_maps_tensor
    proposal.auxiliary["variance_map"] = variance_map
    return proposal, clusters_per_image, mean_map, variance_map
