"""Deployable component-safe grouping and rejection for MicroQuery candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


GroupingMode = Literal["coordinate", "mask", "hybrid", "hybrid_feature"]


@dataclass(frozen=True)
class PairwiseCandidateRelations:
    """Pairwise deployable relations for one image."""

    coordinate_distance: np.ndarray
    soft_iou: np.ndarray
    mask_centroid_distance: np.ndarray
    feature_cosine: np.ndarray | None = None


@dataclass(frozen=True)
class GroupingConfig:
    mode: GroupingMode
    r_xy: float = 3.0
    tau_iou: float = 0.3
    r_near: float = 3.0
    r_far: float = 8.0
    r_mask: float = 5.0
    tau_feat: float = 0.8


@dataclass(frozen=True)
class GroupSelection:
    """Candidate acceptance and mask weights produced without GT."""

    accepted: np.ndarray
    weights: np.ndarray
    state: tuple[str, ...]


@dataclass(frozen=True)
class ComponentSafeHeadOutput:
    semantic_logits: torch.Tensor
    utility_logits: torch.Tensor


def semantic_label_from_roles(
    primary: np.ndarray, duplicate: np.ndarray, candidate_valid: np.ndarray
) -> np.ndarray:
    """Map both primary and duplicate target candidates to semantic positive."""

    primary = np.asarray(primary, dtype=bool)
    duplicate = np.asarray(duplicate, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    if primary.shape != duplicate.shape or primary.shape != valid.shape:
        raise ValueError("primary, duplicate, and candidate_valid must share shape")
    return (valid & (primary | duplicate)).astype(np.int64)


def soft_mask_iou(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return all-pairs soft IoU for probability maps shaped ``[Q,H,W]``."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("probabilities must have shape [Q,H,W]")
    count = values.shape[0]
    result = np.eye(count, dtype=np.float32)
    for first in range(count):
        for second in range(first + 1, count):
            intersection = float(np.minimum(values[first], values[second]).sum())
            union = float(np.maximum(values[first], values[second]).sum())
            value = intersection / (union + float(eps))
            result[first, second] = value
            result[second, first] = value
    return result


def probability_centroid(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return probability-weighted ``(x,y)`` centroids for ``[Q,H,W]`` maps."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("probabilities must have shape [Q,H,W]")
    _, height, width = values.shape
    x_grid = np.arange(width, dtype=np.float32)[None, None, :]
    y_grid = np.arange(height, dtype=np.float32)[None, :, None]
    mass = values.sum(axis=(1, 2))
    x_value = (values * x_grid).sum(axis=(1, 2)) / (mass + float(eps))
    y_value = (values * y_grid).sum(axis=(1, 2)) / (mass + float(eps))
    return np.stack((x_value, y_value), axis=1).astype(np.float32)


def pairwise_candidate_relations(
    candidate_xy: np.ndarray,
    probabilities: np.ndarray,
    descriptors: np.ndarray | None = None,
) -> PairwiseCandidateRelations:
    """Precompute pairwise relations used by every grouping grid point."""

    xy = np.asarray(candidate_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("candidate_xy must have shape [Q,2]")
    if probabilities.shape[0] != xy.shape[0]:
        raise ValueError("candidate and probability counts must match")
    coordinate_distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    mask_iou = soft_mask_iou(probabilities)
    centroids = probability_centroid(probabilities)
    centroid_distance = np.linalg.norm(
        centroids[:, None] - centroids[None, :], axis=-1
    )
    feature_cosine = None
    if descriptors is not None:
        features = np.asarray(descriptors, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != xy.shape[0]:
            raise ValueError("descriptors must have shape [Q,D]")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / np.maximum(norms, 1e-8)
        feature_cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return PairwiseCandidateRelations(
        coordinate_distance.astype(np.float32),
        mask_iou,
        centroid_distance.astype(np.float32),
        None if feature_cosine is None else feature_cosine.astype(np.float32),
    )


def build_candidate_graph(
    relations: PairwiseCandidateRelations,
    candidate_valid: np.ndarray,
    config: GroupingConfig,
) -> np.ndarray:
    """Build a symmetric adjacency matrix using deployable relations only."""

    valid = np.asarray(candidate_valid, dtype=bool)
    count = valid.shape[0]
    matrices = (
        relations.coordinate_distance,
        relations.soft_iou,
        relations.mask_centroid_distance,
    )
    if any(matrix.shape != (count, count) for matrix in matrices):
        raise ValueError("relation matrices and candidate_valid must agree")
    if config.mode == "coordinate":
        connected = relations.coordinate_distance <= float(config.r_xy)
    elif config.mode == "mask":
        connected = relations.soft_iou >= float(config.tau_iou)
    elif config.mode in {"hybrid", "hybrid_feature"}:
        near = relations.coordinate_distance <= float(config.r_near)
        supported_far = (
            (relations.coordinate_distance <= float(config.r_far))
            & (relations.soft_iou >= float(config.tau_iou))
            & (relations.mask_centroid_distance <= float(config.r_mask))
        )
        connected = near | supported_far
        if config.mode == "hybrid_feature":
            if relations.feature_cosine is None:
                raise ValueError("hybrid_feature grouping requires descriptors")
            connected = connected & (
                relations.feature_cosine >= float(config.tau_feat)
            )
    else:
        raise ValueError(f"Unknown grouping mode: {config.mode}")
    connected = connected & valid[:, None] & valid[None, :]
    np.fill_diagonal(connected, valid)
    return np.asarray(connected | connected.T, dtype=bool)


def connected_candidate_groups(
    adjacency: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_xy: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic graph components containing valid candidates only."""

    graph = np.asarray(adjacency, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    if graph.shape != (len(valid), len(valid)):
        raise ValueError("adjacency must have shape [Q,Q]")
    xy = None if candidate_xy is None else np.asarray(candidate_xy, dtype=np.float32)
    remaining = set(np.flatnonzero(valid).tolist())
    groups: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            neighbors = np.flatnonzero(graph[current] & valid).tolist()
            stack.extend(index for index in neighbors if index not in component)
        remaining.difference_update(component)
        if xy is None:
            ordered = tuple(sorted(component))
        else:
            ordered = tuple(
                sorted(component, key=lambda index: (xy[index, 0], xy[index, 1], index))
            )
        groups.append(ordered)
    if xy is None:
        groups.sort(key=lambda group: group)
    else:
        groups.sort(
            key=lambda group: (
                min(float(xy[index, 0]) for index in group),
                min(float(xy[index, 1]) for index in group),
                len(group),
            )
        )
    return tuple(groups)


def summarize_candidate_groups(
    groups: Sequence[Sequence[int]],
    candidate_xy: np.ndarray,
    objectness: np.ndarray,
    raw_score: np.ndarray,
    sam_quality: np.ndarray,
    soft_iou_matrix: np.ndarray,
) -> list[dict]:
    """Summarize deployable group statistics."""

    xy = np.asarray(candidate_xy, dtype=np.float32)
    objectness = np.asarray(objectness, dtype=np.float32)
    raw_score = np.asarray(raw_score, dtype=np.float32)
    sam_quality = np.asarray(sam_quality, dtype=np.float32)
    rows: list[dict] = []
    for group_id, group_values in enumerate(groups):
        group = np.asarray(group_values, dtype=np.int64)
        pair_values = [
            float(soft_iou_matrix[first, second])
            for offset, first in enumerate(group)
            for second in group[offset + 1 :]
        ]
        center = xy[group].mean(axis=0)
        dispersion = float(np.linalg.norm(xy[group] - center, axis=1).max())
        rows.append(
            {
                "group_id": group_id,
                "candidate_indices": tuple(int(index) for index in group),
                "group_size": int(len(group)),
                "max_objectness": float(objectness[group].max()),
                "mean_objectness": float(objectness[group].mean()),
                "max_raw_score": float(raw_score[group].max()),
                "mean_raw_score": float(raw_score[group].mean()),
                "max_sam_quality": float(sam_quality[group].max()),
                "mean_sam_quality": float(sam_quality[group].mean()),
                "coordinate_dispersion": dispersion,
                "mean_pairwise_soft_iou": (
                    float(np.mean(pair_values)) if pair_values else 1.0
                ),
                "highest_objectness_candidate": int(
                    group[np.argmax(objectness[group])]
                ),
                "highest_quality_candidate": int(
                    group[np.argmax(sam_quality[group])]
                ),
            }
        )
    return rows


def _champion(
    group: Sequence[int], scores: np.ndarray, candidate_valid: np.ndarray
) -> int | None:
    valid_indices = [int(index) for index in group if bool(candidate_valid[index])]
    if not valid_indices:
        return None
    return min(valid_indices, key=lambda index: (-float(scores[index]), index))


def select_group_champions(
    groups: Sequence[Sequence[int]],
    objectness: np.ndarray,
    candidate_valid: np.ndarray,
    *,
    tau_high: float,
    tau_group: float,
    champion_scores: np.ndarray | None = None,
) -> GroupSelection:
    """A3 selection: reject low-confidence groups and keep one champion."""

    objectness = np.asarray(objectness, dtype=np.float32)
    valid = np.asarray(candidate_valid, dtype=bool)
    scores = objectness if champion_scores is None else np.asarray(
        champion_scores, dtype=np.float32
    )
    accepted = np.zeros_like(valid)
    weights = np.zeros_like(objectness)
    states = ["INVALID" if not flag else "REJECT" for flag in valid]
    for group in groups:
        valid_group = [int(index) for index in group if valid[index]]
        if not valid_group:
            continue
        group_confidence = max(float(objectness[index]) for index in valid_group)
        if group_confidence < float(tau_group):
            continue
        high = [index for index in valid_group if objectness[index] >= float(tau_high)]
        pool = high if high else valid_group
        champion = _champion(pool, scores, valid)
        if champion is not None:
            accepted[champion] = True
            weights[champion] = 1.0
            states[champion] = "ACCEPT" if high else "RESCUE"
    return GroupSelection(accepted, weights, tuple(states))


def tri_state_group_rejection(
    groups: Sequence[Sequence[int]],
    objectness: np.ndarray,
    candidate_valid: np.ndarray,
    *,
    tau_high: float,
    tau_low: float,
    tau_rescue: float,
    uncertain_weight: float,
    champion_scores: np.ndarray | None = None,
) -> GroupSelection:
    """A4 tri-state group rejection with optional uncertain rescue."""

    if not (0.0 <= tau_low <= tau_high <= 1.0):
        raise ValueError("thresholds must satisfy 0 <= tau_low <= tau_high <= 1")
    objectness = np.asarray(objectness, dtype=np.float32)
    valid = np.asarray(candidate_valid, dtype=bool)
    scores = objectness if champion_scores is None else np.asarray(
        champion_scores, dtype=np.float32
    )
    accepted = np.zeros_like(valid)
    weights = np.zeros_like(objectness)
    states = ["INVALID" if not flag else "REJECT" for flag in valid]
    for group in groups:
        valid_group = [int(index) for index in group if valid[index]]
        if not valid_group:
            continue
        high = [index for index in valid_group if objectness[index] >= tau_high]
        uncertain = [
            index
            for index in valid_group
            if tau_low <= objectness[index] < tau_high
        ]
        if high:
            champion = _champion(high, scores, valid)
            weight = 1.0
            state = "ACCEPT"
        elif uncertain and max(float(objectness[index]) for index in uncertain) >= tau_rescue:
            champion = _champion(uncertain, scores, valid)
            weight = float(uncertain_weight)
            state = "RESCUE"
        else:
            champion = None
            weight = 0.0
            state = "REJECT"
        if champion is not None:
            accepted[champion] = True
            weights[champion] = weight
            states[champion] = state
    return GroupSelection(accepted, weights, tuple(states))


def global_top_l_rescue(
    objectness: np.ndarray,
    candidate_valid: np.ndarray,
    *,
    threshold: float,
    minimum_count: int,
) -> GroupSelection:
    """A2 control: hard gate followed by per-image Top-L rescue."""

    scores = np.asarray(objectness, dtype=np.float32)
    valid = np.asarray(candidate_valid, dtype=bool)
    accepted = valid & (scores >= float(threshold))
    needed = max(0, min(int(minimum_count), int(valid.sum())) - int(accepted.sum()))
    if needed:
        rejected = np.flatnonzero(valid & ~accepted)
        ranking = sorted(rejected.tolist(), key=lambda index: (-float(scores[index]), index))
        accepted[ranking[:needed]] = True
    states = tuple(
        "INVALID" if not valid[index] else ("ACCEPT" if accepted[index] else "REJECT")
        for index in range(len(valid))
    )
    return GroupSelection(accepted, accepted.astype(np.float32), states)


class ComponentSafeMicroQueryHead(nn.Module):
    """Small semantic/utility head whose deployable forward has no GT inputs."""

    def __init__(self, input_dim: int = 451, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.semantic_head = nn.Linear(int(hidden_dim), 2)
        self.utility_head = nn.Linear(int(hidden_dim), 1)

    def forward(
        self, descriptors: torch.Tensor, candidate_valid: torch.Tensor
    ) -> ComponentSafeHeadOutput:
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.input_dim:
            raise ValueError(f"descriptors must have shape [B,Q,{self.input_dim}]")
        if candidate_valid.shape != descriptors.shape[:2]:
            raise ValueError("candidate_valid must have shape [B,Q]")
        token = self.shared(descriptors)
        semantic = self.semantic_head(token)
        utility = self.utility_head(token).squeeze(-1)
        semantic = torch.where(
            candidate_valid.unsqueeze(-1),
            semantic,
            semantic.new_tensor([0.0, -30.0]),
        )
        utility = torch.where(
            candidate_valid, utility, utility.new_full(utility.shape, -30.0)
        )
        return ComponentSafeHeadOutput(semantic, utility)


def component_survival_loss(
    semantic_logits: torch.Tensor,
    component_index: torch.Tensor,
    semantic_target: torch.Tensor,
    candidate_valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize covered training components whose candidates all score low."""

    if semantic_logits.shape != component_index.shape:
        raise ValueError("semantic_logits and component_index must share [B,Q]")
    if semantic_target.shape != component_index.shape or candidate_valid.shape != component_index.shape:
        raise ValueError("targets and validity must share [B,Q]")
    probabilities = torch.sigmoid(semantic_logits)
    losses = []
    for batch_index in range(component_index.shape[0]):
        ids = torch.unique(component_index[batch_index][semantic_target[batch_index] & candidate_valid[batch_index]])
        for component in ids:
            if int(component) < 0:
                continue
            selected = (
                (component_index[batch_index] == component)
                & semantic_target[batch_index]
                & candidate_valid[batch_index]
            )
            failure = torch.prod(1.0 - probabilities[batch_index][selected])
            survival = 1.0 - failure
            losses.append(-torch.log(survival.clamp_min(float(eps))))
    if not losses:
        return semantic_logits.sum() * 0.0
    return torch.stack(losses).mean()


def pairwise_representative_loss(
    utility_logits: torch.Tensor,
    query_iou: torch.Tensor,
    component_index: torch.Tensor,
    semantic_target: torch.Tensor,
    candidate_valid: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Rank the best-IoU representative above duplicates of the same component."""

    losses = []
    for batch_index in range(component_index.shape[0]):
        ids = torch.unique(component_index[batch_index][semantic_target[batch_index] & candidate_valid[batch_index]])
        for component in ids:
            if int(component) < 0:
                continue
            indices = torch.nonzero(
                (component_index[batch_index] == component)
                & semantic_target[batch_index]
                & candidate_valid[batch_index],
                as_tuple=False,
            ).flatten()
            if indices.numel() <= 1:
                continue
            ious = query_iou[batch_index, indices]
            best_offset = int(torch.argmax(ious))
            best_index = indices[best_offset]
            others = torch.cat((indices[:best_offset], indices[best_offset + 1 :]))
            losses.append(
                F.relu(
                    float(margin)
                    - utility_logits[batch_index, best_index]
                    + utility_logits[batch_index, others]
                ).mean()
            )
    if not losses:
        return utility_logits.sum() * 0.0
    return torch.stack(losses).mean()
