#!/usr/bin/env python3
"""Streaming validation metrics for full MicroQuery training."""

from __future__ import annotations

import numpy as np
import torch
from skimage import measure
from sklearn.metrics import average_precision_score, roc_auc_score

from efficient_sam.microquery_metrics import expected_calibration_error


class MaskHistogramAUPRC:
    def __init__(self, bins: int = 2048):
        self.bins = int(bins)
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        probability = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0).reshape(-1)
        target = np.asarray(target, dtype=bool).reshape(-1)
        index = np.minimum((probability * self.bins).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(index[target], minlength=self.bins)
        self.negative += np.bincount(index[~target], minlength=self.bins)

    def finalize(self) -> float:
        total_positive = int(self.positive.sum())
        if total_positive == 0:
            return float("nan")
        tp = np.cumsum(self.positive[::-1], dtype=np.float64)
        fp = np.cumsum(self.negative[::-1], dtype=np.float64)
        recall = tp / float(total_positive)
        precision = tp / np.maximum(1.0, tp + fp)
        previous = np.concatenate(([0.0], recall[:-1]))
        return float(np.sum((recall - previous) * precision))


def _match_components(prediction: np.ndarray, target: np.ndarray, radius: float = 3.0):
    predicted = list(measure.regionprops(measure.label(prediction, connectivity=2)))
    targets = list(measure.regionprops(measure.label(target, connectivity=2)))
    available = set(range(len(predicted)))
    matches = 0
    for region in targets:
        center = np.asarray(region.centroid)
        choices = []
        for index in available:
            distance = float(np.linalg.norm(np.asarray(predicted[index].centroid) - center))
            if distance < float(radius):
                choices.append((distance, index))
        if choices:
            _, index = min(choices)
            available.remove(index)
            matches += 1
    false_pixels = sum(int(predicted[index].area) for index in available)
    return len(targets), matches, false_pixels, len(predicted)


class FullMaskMetricAccumulator:
    def __init__(self, threshold: float = 0.5, distance_threshold: float = 3.0):
        self.threshold = float(threshold)
        self.distance_threshold = float(distance_threshold)
        self.intersection = 0
        self.union = 0
        self.targets = 0
        self.matches = 0
        self.false_pixels = 0
        self.pixels = 0
        self.predicted_pixels = 0
        self.predicted_components = 0
        self.per_image: list[dict] = []
        self.mask_auprc = MaskHistogramAUPRC()
        self.semantic_labels: list[int] = []
        self.object_scores: list[float] = []
        self.raw_scores: list[float] = []

    def update(
        self,
        *,
        names: list[str],
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_scores: torch.Tensor,
        semantic_labels: torch.Tensor,
        object_scores: torch.Tensor | None,
    ) -> None:
        probs = probabilities.detach().float().cpu().numpy()
        gt_rows = targets.detach().cpu().numpy() > 0.5
        valid_rows = candidate_valid.detach().cpu().numpy().astype(bool)
        raw_rows = candidate_scores.detach().float().cpu().numpy()
        semantic_rows = semantic_labels.detach().cpu().numpy().astype(bool)
        object_rows = raw_rows if object_scores is None else object_scores.detach().float().cpu().numpy()
        for index, name in enumerate(names):
            gt = gt_rows[index]
            probability = probs[index]
            prediction = probability >= self.threshold
            intersection = int(np.logical_and(prediction, gt).sum())
            union = int(np.logical_or(prediction, gt).sum())
            denominator = int(prediction.sum()) + int(gt.sum())
            target_count, matches, false_pixels, prediction_count = _match_components(
                prediction, gt, self.distance_threshold
            )
            self.intersection += intersection
            self.union += union
            self.targets += target_count
            self.matches += matches
            self.false_pixels += false_pixels
            self.pixels += int(gt.size)
            self.predicted_pixels += int(prediction.sum())
            self.predicted_components += prediction_count
            self.mask_auprc.update(probability, gt)
            image_iou = intersection / union if union else 1.0
            image_f1 = (2.0 * intersection / denominator) if denominator else 1.0
            self.per_image.append(
                {
                    "image": str(name),
                    "intersection_pixels": intersection,
                    "union_pixels": union,
                    "iou": float(image_iou),
                    "f1": float(image_f1),
                    "target_components": target_count,
                    "detected_components": matches,
                    "false_pixels": false_pixels,
                    "predicted_pixels": int(prediction.sum()),
                    "predicted_components": prediction_count,
                    "pixels": int(gt.size),
                }
            )
            selected = valid_rows[index]
            self.semantic_labels.extend(semantic_rows[index][selected].astype(np.int64).tolist())
            self.object_scores.extend(object_rows[index][selected].astype(float).tolist())
            self.raw_scores.extend(raw_rows[index][selected].astype(float).tolist())

    @staticmethod
    def _safe_classifier_metric(function, labels, scores):
        labels = np.asarray(labels, dtype=np.int64)
        if labels.size == 0 or np.unique(labels).size < 2:
            return float("nan")
        return float(function(labels, np.asarray(scores, dtype=np.float64)))

    def finalize(self) -> dict:
        labels = np.asarray(self.semantic_labels, dtype=np.int64)
        scores = np.asarray(self.object_scores, dtype=np.float64)
        return {
            "images": len(self.per_image),
            "global_iou": self.intersection / self.union if self.union else 1.0,
            "mean_niou": float(np.mean([row["iou"] for row in self.per_image])),
            "f1": float(np.mean([row["f1"] for row in self.per_image])),
            "pd": self.matches / max(1, self.targets),
            "fa": self.false_pixels / max(1, self.pixels),
            "fa_per_million": self.false_pixels / max(1, self.pixels) * 1e6,
            "mask_auprc": self.mask_auprc.finalize(),
            "predicted_foreground_pixels": self.predicted_pixels,
            "connected_components_per_image": self.predicted_components / max(1, len(self.per_image)),
            "objectness_auprc": self._safe_classifier_metric(average_precision_score, labels, scores),
            "objectness_auroc": self._safe_classifier_metric(roc_auc_score, labels, scores),
            "objectness_ece": expected_calibration_error(labels, scores),
            "objectness_brier": float(np.mean((scores - labels) ** 2)) if labels.size else float("nan"),
            "raw_candidate_auprc": self._safe_classifier_metric(
                average_precision_score, labels, self.raw_scores
            ),
        }
