"""Prompt-level metrics for image-only infrared small-target proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk
from sklearn.metrics import average_precision_score, roc_auc_score

from .prompt_proposal import PromptProposal


DEFAULT_BUDGETS = (1, 3, 5, 10, 20, 32)
AREA_BINS = ((1, 9, "1-9"), (10, 16, "10-16"), (17, 25, "17-25"), (26, None, ">25"))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _safe_nanmean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _safe_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.size == 0 or np.unique(labels_array).size < 2:
        return float("nan")
    return float(average_precision_score(labels_array, np.asarray(scores, dtype=np.float64)))


def _safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.size == 0 or np.unique(labels_array).size < 2:
        return float("nan")
    return float(roc_auc_score(labels_array, np.asarray(scores, dtype=np.float64)))


def area_bin(area: int) -> str:
    for lower, upper, name in AREA_BINS:
        if area >= lower and (upper is None or area <= upper):
            return name
    return "0"


@dataclass(frozen=True)
class Component:
    index: int
    area: int
    centroid_xy: tuple[float, float]
    mask: np.ndarray
    dilated_mask: np.ndarray


def extract_components(mask: np.ndarray, dilation_radius: int = 2) -> list[Component]:
    binary = np.asarray(mask) > 0
    labels = label(binary.astype(np.uint8), connectivity=2)
    footprint = disk(max(0, int(dilation_radius)))
    components = []
    for region in regionprops(labels):
        component_mask = labels == region.label
        dilated = dilation(component_mask, footprint=footprint)
        centroid_y, centroid_x = region.centroid
        components.append(
            Component(
                index=len(components),
                area=int(region.area),
                centroid_xy=(float(centroid_x), float(centroid_y)),
                mask=component_mask,
                dilated_mask=dilated,
            )
        )
    return components


def candidate_component_options(
    xy: np.ndarray,
    components: Sequence[Component],
    centroid_radius: float = 3.0,
) -> list[tuple[float, int]]:
    x = float(xy[0])
    y = float(xy[1])
    if not components:
        return []
    height, width = components[0].mask.shape
    xi = int(np.clip(round(x), 0, width - 1))
    yi = int(np.clip(round(y), 0, height - 1))
    matches = []
    for component in components:
        cx, cy = component.centroid_xy
        distance = float(np.hypot(x - cx, y - cy))
        if bool(component.dilated_mask[yi, xi]) or distance <= float(centroid_radius):
            matches.append((distance, component.index))
    return sorted(matches, key=lambda item: (item[0], item[1]))


def greedy_match_candidates(
    candidate_xy: np.ndarray,
    components: Sequence[Component],
    centroid_radius: float = 3.0,
) -> dict:
    """One-to-one deterministic matching in the supplied candidate order."""
    matched_components: set[int] = set()
    matched_candidates: set[int] = set()
    positive_candidates: set[int] = set()
    duplicate_candidates: set[int] = set()
    assignments = []
    for candidate_index, xy in enumerate(np.asarray(candidate_xy)):
        options = candidate_component_options(xy, components, centroid_radius=centroid_radius)
        if options:
            positive_candidates.add(candidate_index)
        available = [item for item in options if item[1] not in matched_components]
        if available:
            distance, component_index = available[0]
            matched_components.add(component_index)
            matched_candidates.add(candidate_index)
            assignments.append((candidate_index, component_index, distance))
        elif options:
            duplicate_candidates.add(candidate_index)
    return {
        "matched_components": matched_components,
        "matched_candidates": matched_candidates,
        "positive_candidates": positive_candidates,
        "duplicate_candidates": duplicate_candidates,
        "assignments": assignments,
    }


class PromptMetricAccumulator:
    def __init__(
        self,
        budgets: Iterable[int] = DEFAULT_BUDGETS,
        dilation_radius: int = 2,
        centroid_radius: float = 3.0,
    ):
        self.budgets = tuple(sorted({int(value) for value in budgets if int(value) > 0}))
        if not self.budgets:
            raise ValueError("At least one positive candidate budget is required")
        self.dilation_radius = int(dilation_radius)
        self.centroid_radius = float(centroid_radius)
        self.per_image_rows: list[dict] = []
        self.per_component_rows: list[dict] = []
        self.candidate_scores: list[float] = []
        self.candidate_labels: list[int] = []
        self.dense_scores: list[np.ndarray] = []
        self.dense_labels: list[np.ndarray] = []
        self._image_count = 0

    def update(self, proposal: PromptProposal, gt_masks: torch.Tensor, names: Sequence[str]) -> None:
        proposal.validate()
        if gt_masks.ndim == 4:
            gt_masks = gt_masks[:, 0]
        if gt_masks.ndim != 3:
            raise ValueError("gt_masks must have shape [B,H,W] or [B,1,H,W]")
        if len(names) != gt_masks.shape[0] or proposal.candidate_xy.shape[0] != gt_masks.shape[0]:
            raise ValueError("Batch sizes for proposal, masks, and names must match")

        coords_batch = proposal.candidate_xy.detach().cpu().numpy()
        scores_batch = proposal.candidate_scores.detach().cpu().numpy()
        valid_batch = proposal.candidate_valid.detach().cpu().numpy()
        masks_batch = (gt_masks.detach().cpu().numpy() > 0.5)
        dense_batch = (
            proposal.dense_probs.detach().float().cpu().numpy()[:, 0]
            if proposal.dense_probs is not None
            else None
        )

        for batch_index, name in enumerate(names):
            mask = masks_batch[batch_index]
            components = extract_components(mask, dilation_radius=self.dilation_radius)
            valid_indices = np.flatnonzero(valid_batch[batch_index])
            coords = coords_batch[batch_index, valid_indices]
            scores = scores_batch[batch_index, valid_indices]
            if scores.size:
                order = np.argsort(-scores, kind="stable")
                coords = coords[order]
                scores = scores[order]

            candidate_positive = []
            for xy, score in zip(coords, scores):
                positive = bool(candidate_component_options(xy, components, self.centroid_radius))
                candidate_positive.append(int(positive))
                self.candidate_labels.append(int(positive))
                self.candidate_scores.append(float(score))

            dense = dense_batch[batch_index] if dense_batch is not None else None
            if dense is not None:
                self.dense_scores.append(dense.reshape(-1).astype(np.float32))
                self.dense_labels.append(mask.reshape(-1).astype(np.uint8))

            component_rows = []
            for component in components:
                max_response = float(dense[component.mask].max()) if dense is not None else float("nan")
                row = {
                    "name": str(name),
                    "component_index": component.index,
                    "area": component.area,
                    "area_bin": area_bin(component.area),
                    "centroid_x": component.centroid_xy[0],
                    "centroid_y": component.centroid_xy[1],
                    "max_response": max_response,
                }
                component_rows.append(row)

            image_row = {
                "name": str(name),
                "components": len(components),
                "candidate_count": int(len(coords)),
                "zero_prompt": int(len(coords) == 0),
                "pixels": int(mask.size),
                "target_present": int(bool(components)),
            }
            if dense is not None:
                component_maxima = [float(dense[item.mask].max()) for item in components]
                background_values = dense[~mask]
                background_mean = float(background_values.mean()) if background_values.size else 0.0
                background_std = float(background_values.std()) if background_values.size else 0.0
                mean_component_max = float(np.mean(component_maxima)) if component_maxima else float("nan")
                image_row.update(
                    {
                        "mean_component_max_response": mean_component_max,
                        "background_mean_response": background_mean,
                        "peak_to_background_contrast": (
                            (mean_component_max - background_mean) / (background_std + 1e-6)
                            if component_maxima
                            else float("nan")
                        ),
                    }
                )

            for budget in self.budgets:
                selected = coords[:budget]
                matched = greedy_match_candidates(
                    selected,
                    components,
                    centroid_radius=self.centroid_radius,
                )
                hit_components = matched["matched_components"]
                matched_count = len(hit_components)
                selected_count = len(selected)
                false_count = selected_count - len(matched["matched_candidates"])
                image_row.update(
                    {
                        f"matched_components_at_{budget}": matched_count,
                        f"component_recall_at_{budget}": _safe_ratio(matched_count, len(components)),
                        f"prompt_precision_at_{budget}": _safe_ratio(
                            len(matched["matched_candidates"]), selected_count
                        ),
                        f"false_prompts_at_{budget}": false_count,
                        f"duplicates_at_{budget}": len(matched["duplicate_candidates"]),
                        f"center_hit_at_{budget}": int(bool(hit_components)),
                    }
                )
                for component_row in component_rows:
                    component_row[f"hit_at_{budget}"] = int(
                        component_row["component_index"] in hit_components
                    )
            self.per_image_rows.append(image_row)
            self.per_component_rows.extend(component_rows)
            self._image_count += 1

    def finalize(self) -> dict:
        if not self.per_image_rows:
            raise RuntimeError("No prompt batches were accumulated")
        image_count = len(self.per_image_rows)
        total_pixels = sum(int(row["pixels"]) for row in self.per_image_rows)
        total_components = sum(int(row["components"]) for row in self.per_image_rows)
        target_images = sum(int(row["target_present"]) for row in self.per_image_rows)

        budget_rows = []
        for budget in self.budgets:
            matched = sum(int(row[f"matched_components_at_{budget}"]) for row in self.per_image_rows)
            selected = sum(min(int(row["candidate_count"]), budget) for row in self.per_image_rows)
            false_prompts = sum(int(row[f"false_prompts_at_{budget}"]) for row in self.per_image_rows)
            duplicates = sum(int(row[f"duplicates_at_{budget}"]) for row in self.per_image_rows)
            center_hits = sum(
                int(row[f"center_hit_at_{budget}"])
                for row in self.per_image_rows
                if int(row["target_present"])
            )
            budget_rows.append(
                {
                    "budget": budget,
                    "center_hit": _safe_ratio(center_hits, target_images),
                    "component_recall": _safe_ratio(matched, total_components),
                    "prompt_precision": _safe_ratio(matched, selected),
                    "false_prompts_per_million_pixels": _safe_ratio(false_prompts * 1e6, total_pixels),
                    "mean_candidates_per_image": _safe_ratio(selected, image_count),
                    "zero_prompt_fraction": _safe_ratio(
                        sum(int(row["candidate_count"] == 0) for row in self.per_image_rows),
                        image_count,
                    ),
                    "duplicate_prompts_per_component": _safe_ratio(duplicates, total_components),
                    "matched_components": matched,
                    "total_components": total_components,
                    "false_prompts": false_prompts,
                }
            )

        area_rows = []
        for _, _, bin_name in AREA_BINS:
            rows = [row for row in self.per_component_rows if row["area_bin"] == bin_name]
            for budget in self.budgets:
                hits = sum(int(row[f"hit_at_{budget}"]) for row in rows)
                false_prompts = sum(
                    int(row[f"false_prompts_at_{budget}"])
                    for row in self.per_image_rows
                )
                area_rows.append(
                    {
                        "area_bin": bin_name,
                        "budget": budget,
                        "components": len(rows),
                        "component_recall": _safe_ratio(hits, len(rows)),
                        "mean_maximum_response": (
                            _safe_nanmean([row["max_response"] for row in rows])
                            if rows
                            else float("nan")
                        ),
                        "false_prompt_adjusted_recall": _safe_ratio(
                            hits,
                            len(rows) + false_prompts,
                        ),
                    }
                )

        if self.dense_scores:
            dense_scores = np.concatenate(self.dense_scores)
            dense_labels = np.concatenate(self.dense_labels)
            dense_auprc = _safe_average_precision(dense_labels, dense_scores)
            dense_auroc = _safe_auroc(dense_labels, dense_scores)
        else:
            dense_auprc = dense_auroc = float("nan")

        summary = {
            "images": image_count,
            "target_images": target_images,
            "components": total_components,
            "candidate_score_auprc": _safe_average_precision(
                self.candidate_labels,
                self.candidate_scores,
            ),
            "dense_prompt_auprc": dense_auprc,
            "dense_prompt_auroc": dense_auroc,
            "mean_peak_to_background_contrast": _safe_nanmean(
                [row.get("peak_to_background_contrast", float("nan")) for row in self.per_image_rows]
            ),
            "mean_gt_component_maximum_response": _safe_nanmean(
                [row.get("max_response", float("nan")) for row in self.per_component_rows]
            ),
            "budgets": list(self.budgets),
            "matching": {
                "component_dilation_radius_px": self.dilation_radius,
                "centroid_radius_px": self.centroid_radius,
                "assignment": "score-ordered deterministic greedy one-to-one",
            },
        }
        return {
            "summary": summary,
            "budget_rows": budget_rows,
            "area_rows": area_rows,
            "per_image_rows": self.per_image_rows,
            "per_component_rows": self.per_component_rows,
        }
