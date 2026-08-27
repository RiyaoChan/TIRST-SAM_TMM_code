"""Candidate-conditional metrics for MicroQuery-SAM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage.measure import label, regionprops
from sklearn.metrics import average_precision_score, roc_auc_score

from .prompt_metrics import Component, area_bin, candidate_component_options, extract_components


ASSIGNMENT_PRIMARY = "primary"
ASSIGNMENT_DUPLICATE = "duplicate"
ASSIGNMENT_BACKGROUND = "background"


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _safe_metric(function, labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.size == 0 or np.unique(labels_array).size < 2:
        return float("nan")
    return float(function(labels_array, np.asarray(scores, dtype=np.float64)))


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float:
    labels_array = np.asarray(labels, dtype=np.float64)
    scores_array = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if labels_array.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(labels_array.size)
    value = 0.0
    for index in range(int(bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (scores_array >= lower) & (
            scores_array <= upper if index == int(bins) - 1 else scores_array < upper
        )
        if selected.any():
            value += float(selected.sum()) / total * abs(
                float(labels_array[selected].mean()) - float(scores_array[selected].mean())
            )
    return value


@dataclass(frozen=True)
class CandidateAssignment:
    semantic_target: np.ndarray
    assignment: tuple[str, ...]
    component_index: np.ndarray
    covered_components: frozenset[int]
    best_rank_by_component: dict[int, int]


def assign_candidates(
    candidate_xy: np.ndarray,
    candidate_valid: np.ndarray,
    components: Sequence[Component],
    *,
    budget: int,
    centroid_radius: float = 3.0,
) -> CandidateAssignment:
    """Create semantic and deterministic one-to-one labels in candidate order."""

    xy = np.asarray(candidate_xy, dtype=np.float32)
    valid = np.asarray(candidate_valid, dtype=bool)
    count = min(int(budget), int(xy.shape[0]))
    semantic = np.zeros(count, dtype=bool)
    component_index = np.full(count, -1, dtype=np.int64)
    assignment: list[str] = []
    used: set[int] = set()
    best_rank: dict[int, int] = {}
    for index in range(count):
        if not valid[index]:
            assignment.append(ASSIGNMENT_BACKGROUND)
            continue
        options = candidate_component_options(
            xy[index], components, centroid_radius=centroid_radius
        )
        semantic[index] = bool(options)
        for _, candidate_component in options:
            best_rank.setdefault(candidate_component, index + 1)
        available = [item for item in options if item[1] not in used]
        if available:
            _, matched_component = available[0]
            used.add(matched_component)
            component_index[index] = matched_component
            assignment.append(ASSIGNMENT_PRIMARY)
        elif options:
            component_index[index] = options[0][1]
            assignment.append(ASSIGNMENT_DUPLICATE)
        else:
            assignment.append(ASSIGNMENT_BACKGROUND)
    return CandidateAssignment(
        semantic_target=semantic,
        assignment=tuple(assignment),
        component_index=component_index,
        covered_components=frozenset(used),
        best_rank_by_component=best_rank,
    )


def _match_prediction_components(
    prediction: np.ndarray,
    components: Sequence[Component],
    *,
    centroid_radius: float = 3.0,
) -> tuple[set[int], int]:
    predicted_regions = list(regionprops(label(np.asarray(prediction) > 0, connectivity=2)))
    available = set(range(len(predicted_regions)))
    detected: set[int] = set()
    for component in components:
        cx, cy = component.centroid_xy
        choices = []
        for predicted_index in available:
            py, px = predicted_regions[predicted_index].centroid
            distance = float(np.hypot(px - cx, py - cy))
            if distance < float(centroid_radius):
                choices.append((distance, predicted_index))
        if choices:
            _, predicted_index = min(choices)
            available.remove(predicted_index)
            detected.add(component.index)
    false_pixels = sum(int(predicted_regions[index].area) for index in available)
    return detected, false_pixels


def detected_component_indices(
    prediction: np.ndarray,
    components: Sequence[Component],
    *,
    centroid_radius: float = 3.0,
) -> set[int]:
    detected, _ = _match_prediction_components(
        prediction, components, centroid_radius=centroid_radius
    )
    return detected


def _binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return float(intersection / union) if union else 1.0


class MicroQueryMetricAccumulator:
    """Accumulate coverage, recovery, rejection, leakage, and mask metrics."""

    def __init__(self, budget: int, centroid_radius: float = 3.0):
        self.budget = int(budget)
        self.centroid_radius = float(centroid_radius)
        self.per_image_rows: list[dict] = []
        self.per_component_rows: list[dict] = []
        self.per_query_rows: list[dict] = []
        self.semantic_labels: list[int] = []
        self.candidate_scores: list[float] = []
        self.object_scores: list[float] = []
        self.total_intersection = 0
        self.total_union = 0
        self.total_target_components = 0
        self.total_detected_components = 0
        self.total_false_pixels = 0
        self.total_pixels = 0

    def update(
        self,
        *,
        name: str,
        gt_mask: np.ndarray,
        candidate_xy: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_valid: np.ndarray,
        query_probabilities: np.ndarray,
        final_probability: np.ndarray,
        accepted: np.ndarray,
        object_scores: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> None:
        gt = np.asarray(gt_mask) > 0
        final_prediction = np.asarray(final_probability) >= float(threshold)
        query_probabilities = np.asarray(query_probabilities, dtype=np.float32)[: self.budget]
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)[: self.budget]
        candidate_valid = np.asarray(candidate_valid, dtype=bool)[: self.budget]
        accepted = np.asarray(accepted, dtype=bool)[: self.budget] & candidate_valid
        if object_scores is None:
            object_scores = candidate_scores
        object_scores = np.asarray(object_scores, dtype=np.float32)[: self.budget]
        components = extract_components(gt)
        assignment = assign_candidates(
            candidate_xy,
            candidate_valid,
            components,
            budget=self.budget,
            centroid_radius=self.centroid_radius,
        )
        detected, final_false_pixels = _match_prediction_components(
            final_prediction, components, centroid_radius=self.centroid_radius
        )
        covered = set(assignment.covered_components)
        uncovered = set(range(len(components))) - covered
        covered_detected = covered & detected
        uncovered_detected = uncovered & detected
        retained_components: set[int] = set()
        for index, is_accepted in enumerate(accepted):
            if not is_accepted:
                continue
            options = candidate_component_options(
                np.asarray(candidate_xy)[index], components, centroid_radius=self.centroid_radius
            )
            retained_components.update(component_index for _, component_index in options)

        intersection = int(np.logical_and(final_prediction, gt).sum())
        union = int(np.logical_or(final_prediction, gt).sum())
        self.total_intersection += intersection
        self.total_union += union
        self.total_pixels += int(gt.size)
        self.total_target_components += len(components)
        self.total_detected_components += len(detected)
        self.total_false_pixels += final_false_pixels

        component_best_ious: dict[int, float] = {component.index: 0.0 for component in components}
        component_overlap: dict[int, int] = {component.index: 0 for component in components}
        false_query_pixels = 0
        background_queries = 0
        accepted_background = 0
        duplicate_queries = 0
        accepted_duplicates = 0
        for index in range(min(self.budget, len(candidate_valid))):
            if not candidate_valid[index]:
                continue
            semantic = int(assignment.semantic_target[index])
            assignment_name = assignment.assignment[index]
            query_prediction = query_probabilities[index] >= float(threshold)
            best_iou = 0.0
            overlap_pixels = 0
            options = candidate_component_options(
                np.asarray(candidate_xy)[index], components, centroid_radius=self.centroid_radius
            )
            for _, component_index in options:
                component = components[component_index]
                iou = _binary_iou(query_prediction, component.mask)
                overlap = int(np.logical_and(query_prediction, component.mask).sum())
                if iou > best_iou:
                    best_iou = iou
                    overlap_pixels = overlap
                component_best_ious[component_index] = max(
                    component_best_ious[component_index], iou
                )
                component_overlap[component_index] = max(
                    component_overlap[component_index], overlap
                )
            false_pixels = int(query_prediction.sum()) if not semantic else 0
            if not semantic:
                background_queries += 1
                false_query_pixels += false_pixels
                accepted_background += int(accepted[index])
            if assignment_name == ASSIGNMENT_DUPLICATE:
                duplicate_queries += 1
                accepted_duplicates += int(accepted[index])
            self.semantic_labels.append(semantic)
            self.candidate_scores.append(float(candidate_scores[index]))
            self.object_scores.append(float(object_scores[index]))
            self.per_query_rows.append(
                {
                    "image": str(name),
                    "candidate_rank": index + 1,
                    "semantic_target": semantic,
                    "assignment": assignment_name,
                    "component_index": int(assignment.component_index[index]),
                    "candidate_score": float(candidate_scores[index]),
                    "object_score": float(object_scores[index]),
                    "accepted": int(accepted[index]),
                    "best_query_iou": best_iou,
                    "overlap_pixels": overlap_pixels,
                    "false_mask_pixels": false_pixels,
                }
            )

        for component in components:
            self.per_component_rows.append(
                {
                    "image": str(name),
                    "component_index": component.index,
                    "area": component.area,
                    "area_bin": area_bin(component.area),
                    "covered": int(component.index in covered),
                    "best_candidate_rank": assignment.best_rank_by_component.get(
                        component.index, -1
                    ),
                    "final_detected": int(component.index in detected),
                    "retained": int(component.index in retained_components),
                    "best_query_iou": component_best_ious[component.index],
                    "best_query_overlap_pixels": component_overlap[component.index],
                }
            )
        image_iou = float(intersection / union) if union else 1.0
        f1_denominator = int(final_prediction.sum()) + int(gt.sum())
        image_f1 = float(2 * intersection / f1_denominator) if f1_denominator else 1.0
        self.per_image_rows.append(
            {
                "image": str(name),
                "components": len(components),
                "covered_components": len(covered),
                "detected_components": len(detected),
                "covered_detected": len(covered_detected),
                "uncovered_detected": len(uncovered_detected),
                "retained_covered_components": len(covered & retained_components),
                "background_queries": background_queries,
                "rejected_background_queries": background_queries - accepted_background,
                "duplicate_queries": duplicate_queries,
                "rejected_duplicate_queries": duplicate_queries - accepted_duplicates,
                "false_query_mask_pixels": false_query_pixels,
                "final_false_pixels": final_false_pixels,
                "intersection_pixels": intersection,
                "union_pixels": union,
                "iou": image_iou,
                "f1": image_f1,
                "pixels": int(gt.size),
            }
        )

    def finalize(self) -> dict:
        covered = sum(int(row["covered_components"]) for row in self.per_image_rows)
        covered_detected = sum(int(row["covered_detected"]) for row in self.per_image_rows)
        components = sum(int(row["components"]) for row in self.per_image_rows)
        uncovered = components - covered
        uncovered_detected = sum(int(row["uncovered_detected"]) for row in self.per_image_rows)
        retained = sum(
            int(row["retained_covered_components"]) for row in self.per_image_rows
        )
        background_queries = sum(int(row["background_queries"]) for row in self.per_image_rows)
        rejected_background = sum(
            int(row["rejected_background_queries"]) for row in self.per_image_rows
        )
        duplicate_queries = sum(int(row["duplicate_queries"]) for row in self.per_image_rows)
        rejected_duplicates = sum(
            int(row["rejected_duplicate_queries"]) for row in self.per_image_rows
        )
        covered_rows = [row for row in self.per_component_rows if int(row["covered"])]
        summary = {
            "images": len(self.per_image_rows),
            "candidate_budget": self.budget,
            "components": components,
            "candidate_coverage": _safe_ratio(covered, components),
            "all_target_image_coverage": _safe_ratio(
                sum(
                    int(row["components"] > 0 and row["covered_components"] == row["components"])
                    for row in self.per_image_rows
                ),
                sum(int(row["components"] > 0) for row in self.per_image_rows),
            ),
            "covered_target_recovery": _safe_ratio(covered_detected, covered),
            "uncovered_target_incidental_detection": _safe_ratio(
                uncovered_detected, uncovered
            ),
            "target_candidate_retention": _safe_ratio(retained, covered),
            "false_candidate_rejection": _safe_ratio(
                rejected_background, background_queries
            ),
            "duplicate_suppression": _safe_ratio(rejected_duplicates, duplicate_queries),
            "duplicate_queries_per_component": _safe_ratio(duplicate_queries, components),
            "best_query_mask_iou": float(
                np.mean([row["best_query_iou"] for row in covered_rows])
            )
            if covered_rows
            else float("nan"),
            "qmsr_at_0_1": _safe_ratio(
                sum(float(row["best_query_iou"]) >= 0.1 for row in covered_rows), len(covered_rows)
            ),
            "qmsr_at_0_3": _safe_ratio(
                sum(float(row["best_query_iou"]) >= 0.3 for row in covered_rows), len(covered_rows)
            ),
            "qmsr_at_0_5": _safe_ratio(
                sum(float(row["best_query_iou"]) >= 0.5 for row in covered_rows), len(covered_rows)
            ),
            "mean_false_mask_pixels_per_background_query": _safe_ratio(
                sum(int(row["false_query_mask_pixels"]) for row in self.per_image_rows),
                background_queries,
            ),
            "candidate_score_auprc": _safe_metric(
                average_precision_score, self.semantic_labels, self.candidate_scores
            ),
            "objectness_auprc": _safe_metric(
                average_precision_score, self.semantic_labels, self.object_scores
            ),
            "objectness_auroc": _safe_metric(
                roc_auc_score, self.semantic_labels, self.object_scores
            ),
            "objectness_brier": float(
                np.mean(
                    (
                        np.asarray(self.object_scores, dtype=np.float64)
                        - np.asarray(self.semantic_labels, dtype=np.float64)
                    )
                    ** 2
                )
            )
            if self.object_scores
            else float("nan"),
            "objectness_ece": expected_calibration_error(
                self.semantic_labels, self.object_scores
            ),
            "global_iou": _safe_ratio(self.total_intersection, self.total_union),
            "mean_niou": float(np.mean([row["iou"] for row in self.per_image_rows])),
            "f1": float(np.mean([row["f1"] for row in self.per_image_rows])),
            "pd": _safe_ratio(self.total_detected_components, self.total_target_components),
            "fa": _safe_ratio(self.total_false_pixels, self.total_pixels),
        }
        return {
            "summary": summary,
            "per_image_rows": self.per_image_rows,
            "per_component_rows": self.per_component_rows,
            "per_query_rows": self.per_query_rows,
        }
