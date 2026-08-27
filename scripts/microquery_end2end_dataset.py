#!/usr/bin/env python3
"""Dataset boundary for full end-to-end MicroQuery training.

Cached candidates are deployable image-only inputs. Connected-component labels,
query masks and covered masks are returned in a separate ``supervision`` mapping
and are never accepted by the model forward path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from efficient_sam.microquery_end2end import (
    assign_candidates_to_components,
    build_covered_gt_mask,
    build_query_targets,
)
from sirst_dataset import SIRSTDataset


def load_candidate_cache(path: str | Path, expected_names: list[str], budget: int = 10) -> dict[str, np.ndarray]:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as values:
        required = {"image_names", "candidate_xy", "candidate_scores", "candidate_valid"}
        if set(values.files) != required:
            raise RuntimeError(
                f"candidate cache must contain exactly {sorted(required)}, got {sorted(values.files)}"
            )
        cache = {key: values[key].copy() for key in required}
    names = [str(value) for value in cache["image_names"].tolist()]
    if names != list(expected_names):
        mismatch = next(
            (index for index, pair in enumerate(zip(names, expected_names)) if pair[0] != pair[1]),
            min(len(names), len(expected_names)),
        )
        raise RuntimeError(
            f"candidate cache order differs from dataset at index {mismatch}: "
            f"cache={names[mismatch:mismatch + 1]} dataset={expected_names[mismatch:mismatch + 1]}"
        )
    if len(names) != len(expected_names):
        raise RuntimeError("candidate cache and split have different image counts")
    budget = int(budget)
    if budget <= 0 or cache["candidate_xy"].shape[1] < budget:
        raise ValueError("candidate budget is outside cached range")
    return {
        "image_names": cache["image_names"],
        "candidate_xy": cache["candidate_xy"][:, :budget].astype(np.float32, copy=False),
        "candidate_scores": cache["candidate_scores"][:, :budget].astype(np.float32, copy=False),
        "candidate_valid": cache["candidate_valid"][:, :budget].astype(bool, copy=False),
    }


def flip_image_mask_coordinates(
    image: torch.Tensor,
    mask: torch.Tensor,
    candidate_xy: torch.Tensor,
    *,
    horizontal: bool,
    vertical: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply exact synchronized flips in 256-space."""

    if image.ndim != 3 or mask.ndim != 2 or candidate_xy.ndim != 2:
        raise ValueError("expected image [C,H,W], mask [H,W], candidate_xy [K,2]")
    height, width = mask.shape
    output_image = image
    output_mask = mask
    output_xy = candidate_xy.clone()
    if horizontal:
        output_image = torch.flip(output_image, dims=(-1,))
        output_mask = torch.flip(output_mask, dims=(-1,))
        output_xy[:, 0] = float(width - 1) - output_xy[:, 0]
    if vertical:
        output_image = torch.flip(output_image, dims=(-2,))
        output_mask = torch.flip(output_mask, dims=(-2,))
        output_xy[:, 1] = float(height - 1) - output_xy[:, 1]
    return output_image.contiguous(), output_mask.contiguous(), output_xy


class MicroQueryEndToEndDataset(Dataset):
    """Fixed-resize IRSTD dataset with cached candidates and flip-only augmentation."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        split: str | Path,
        candidate_cache: str | Path,
        size: int = 256,
        budget: int = 10,
        augment: bool = False,
        seed: int = 20260825,
        mask_suffix: str = "",
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.split = Path(split).resolve()
        self.size = int(size)
        self.budget = int(budget)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.base = SIRSTDataset(
            root=str(self.data_root),
            split_txt=str(self.split),
            size=self.size,
            keep_ratio_pad=False,
            augment=False,
            skip_bg_only=False,
            mask_suffix=mask_suffix,
            sctransnet_preproc=True,
            sc_use_noise=False,
            sc_use_gamma=False,
            sc_dataset_name="IRSTD-1k",
            sc_eval_crop="resize",
            mllm_features_path=None,
        )
        expected_names = [Path(sample[0]).stem for sample in self.base.samples]
        self.candidates = load_candidate_cache(candidate_cache, expected_names, budget=self.budget)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def _flip_decisions(self, index: int) -> tuple[bool, bool]:
        if not self.augment:
            return False, False
        payload = f"{self.seed}:{self.epoch}:{int(index)}".encode("ascii")
        digest = hashlib.sha256(payload).digest()
        return digest[0] < 128, digest[1] < 128

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[int(index)]
        image = sample["image"].float()
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        mask = sample["mask"].float()
        xy = torch.from_numpy(self.candidates["candidate_xy"][index].copy()).float()
        scores = torch.from_numpy(self.candidates["candidate_scores"][index].copy()).float()
        valid = torch.from_numpy(self.candidates["candidate_valid"][index].copy()).bool()
        horizontal, vertical = self._flip_decisions(int(index))
        image, mask, xy = flip_image_mask_coordinates(
            image, mask, xy, horizontal=horizontal, vertical=vertical
        )
        assignment = assign_candidates_to_components(xy, valid, mask)
        query_targets = build_query_targets(assignment.component_map, assignment.component_ids).float()
        covered = build_covered_gt_mask(assignment.component_map, assignment.component_ids).float()
        return {
            "deployable": {
                "image": image,
                "candidate_xy": xy,
                "candidate_scores": scores,
                "candidate_valid": valid,
            },
            "supervision": {
                "full_mask": mask,
                "covered_mask": covered,
                "query_targets": query_targets,
                "semantic_labels": torch.from_numpy(assignment.semantic_labels.copy()).bool(),
                "component_ids": torch.from_numpy(assignment.component_ids.copy()).long(),
                "component_map": torch.from_numpy(assignment.component_map.copy()).long(),
                "multi_match_count": torch.tensor(assignment.multi_match_count, dtype=torch.long),
            },
            "meta": {
                "name": str(sample["name"]),
                "index": torch.tensor(index, dtype=torch.long),
                "horizontal_flip": torch.tensor(horizontal),
                "vertical_flip": torch.tensor(vertical),
            },
        }


def candidate_class_weights(dataset: MicroQueryEndToEndDataset) -> tuple[torch.Tensor, dict[str, int]]:
    """Count frozen train-split semantic labels once, without augmentation."""

    prior_augment = dataset.augment
    dataset.augment = False
    positive = 0
    negative = 0
    invalid = 0
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            valid = sample["deployable"]["candidate_valid"]
            semantic = sample["supervision"]["semantic_labels"]
            positive += int((valid & semantic).sum())
            negative += int((valid & ~semantic).sum())
            invalid += int((~valid).sum())
    finally:
        dataset.augment = prior_augment
    weights = torch.tensor([1.0, negative / max(1, positive)], dtype=torch.float32)
    return weights, {"positive": positive, "negative": negative, "invalid": invalid}
