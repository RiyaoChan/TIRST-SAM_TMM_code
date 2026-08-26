#!/usr/bin/env python3
"""Train and evaluate the minimal frozen-candidate MicroQuery M2-2 sanity head."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery import MicroQueryHead
from efficient_sam.microquery_metrics import (
    MicroQueryMetricAccumulator,
    expected_calibration_error,
)
from scripts.eval_prompt_quality import sha256_file
from scripts.train_experiment1_single_view import set_deterministic


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item) for item in value.split(",") if item.strip())
    if not thresholds or min(thresholds) < 0.0 or max(thresholds) > 1.0:
        raise argparse.ArgumentTypeError("thresholds must lie in [0,1]")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--train_targets", required=True)
    parser.add_argument("--val_features", required=True)
    parser.add_argument("--val_targets", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="IRSTD-1k")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--quality_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05)),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def safe_binary_metric(function, labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(function(labels, scores))


def load_cache(feature_path: Path, target_path: Path, *, require_probs: bool) -> dict:
    features = np.load(feature_path, allow_pickle=False)
    targets = np.load(target_path, allow_pickle=False)
    feature_names = [str(name) for name in features["image_names"]]
    target_names = [str(name) for name in targets["image_names"]]
    if feature_names != target_names:
        raise RuntimeError("Feature and target caches do not have identical image order")
    required = {
        "roi_descriptors",
        "candidate_xy",
        "candidate_scores",
        "candidate_valid",
        "sam_decoder_quality",
    }
    if require_probs:
        required.add("query_probabilities")
    missing = sorted(required - set(features.files))
    if missing:
        raise RuntimeError(f"Feature cache is missing: {missing}")
    return {
        "names": feature_names,
        "descriptors": features["roi_descriptors"].astype(np.float32),
        "xy": features["candidate_xy"].astype(np.float32),
        "candidate_scores": features["candidate_scores"].astype(np.float32),
        "valid": features["candidate_valid"].astype(bool),
        "sam_quality": features["sam_decoder_quality"].astype(np.float32),
        "query_probabilities": (
            features["query_probabilities"].astype(np.float32) if require_probs else None
        ),
        "semantic": targets["semantic_target"].astype(bool),
        "primary": targets["primary_target"].astype(bool),
        "duplicate": targets["duplicate_target"].astype(bool),
        "component_index": targets["component_index"].astype(np.int64),
        "query_iou": targets["query_mask_iou"].astype(np.float32),
        "gt_masks": targets["gt_masks"].astype(bool),
    }


def predict_head(
    model: MicroQueryHead, descriptors: np.ndarray, valid: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    object_batches: list[np.ndarray] = []
    quality_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(descriptors), 128):
            end = min(len(descriptors), start + 128)
            x = torch.from_numpy(descriptors[start:end]).to(device)
            v = torch.from_numpy(valid[start:end]).to(device)
            output = model(x, v)
            object_batches.append(
                torch.softmax(output.object_logits, dim=-1)[..., 1].cpu().numpy()
            )
            quality_batches.append(torch.sigmoid(output.quality_logits).cpu().numpy())
    return np.concatenate(object_batches), np.concatenate(quality_batches)


def classification_metrics(data: dict, object_scores: np.ndarray) -> dict:
    valid = data["valid"]
    semantic = data["semantic"][valid].astype(np.int64)
    primary = data["primary"][valid].astype(np.int64)
    scores = object_scores[valid]
    raw = data["candidate_scores"][valid]
    return {
        "semantic_auprc": safe_binary_metric(average_precision_score, semantic, scores),
        "semantic_auroc": safe_binary_metric(roc_auc_score, semantic, scores),
        "semantic_ece": expected_calibration_error(semantic, scores),
        "semantic_brier": float(np.mean((scores - semantic) ** 2)),
        "primary_auprc": safe_binary_metric(average_precision_score, primary, scores),
        "primary_auroc": safe_binary_metric(roc_auc_score, primary, scores),
        "primary_ece": expected_calibration_error(primary, scores),
        "primary_brier": float(np.mean((scores - primary) ** 2)),
        "raw_semantic_auprc": safe_binary_metric(average_precision_score, semantic, raw),
        "raw_primary_auprc": safe_binary_metric(average_precision_score, primary, raw),
    }


def candidate_rejection_metrics(
    data: dict, object_scores: np.ndarray, threshold: float
) -> dict:
    valid = data["valid"]
    accepted = valid & (object_scores >= float(threshold))
    primary = data["primary"] & valid
    background = valid & ~data["semantic"]
    duplicate = data["duplicate"] & valid
    covered_ids: set[tuple[int, int]] = set()
    retained_ids: set[tuple[int, int]] = set()
    for image_index in range(len(data["names"])):
        for query_index in range(valid.shape[1]):
            component = int(data["component_index"][image_index, query_index])
            if primary[image_index, query_index] and component >= 0:
                covered_ids.add((image_index, component))
            if accepted[image_index, query_index] and component >= 0:
                retained_ids.add((image_index, component))
    true_positive = int((accepted & primary).sum())
    false_positive = int((accepted & ~primary & valid).sum())
    false_negative = int((~accepted & primary).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    no_object = valid & ~primary
    return {
        "object_threshold": float(threshold),
        "accepted_queries": int(accepted.sum()),
        "query_coverage": float(accepted.sum() / max(1, valid.sum())),
        "classification_risk": float(1.0 - precision),
        "primary_precision": float(precision),
        "primary_recall": float(recall),
        "primary_f1": float(f1),
        "no_object_accuracy": float((~accepted & no_object).sum() / max(1, no_object.sum())),
        "target_candidate_retention": float(
            len(covered_ids & retained_ids) / max(1, len(covered_ids))
        ),
        "false_candidate_rejection": float(
            (~accepted & background).sum() / max(1, background.sum())
        ),
        "duplicate_suppression": float(
            (~accepted & duplicate).sum() / max(1, duplicate.sum())
        ),
    }


def aggregate_hard_gate(
    query_probabilities: np.ndarray, accepted: np.ndarray
) -> np.ndarray:
    weighted = np.where(accepted[..., None, None], query_probabilities, 0.0)
    return weighted.max(axis=1)


def aggregate_weighted(
    query_probabilities: np.ndarray, valid: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    clipped = np.where(valid, np.clip(weights, 0.0, 1.0), 0.0)
    return (query_probabilities * clipped[..., None, None]).max(axis=1)


def evaluate_mask_condition(
    data: dict,
    final_probabilities: np.ndarray,
    accepted: np.ndarray,
    object_scores: np.ndarray,
    mask_threshold: float,
) -> dict:
    budget = int(data["valid"].shape[1])
    accumulator = MicroQueryMetricAccumulator(budget)
    for image_index, name in enumerate(data["names"]):
        accumulator.update(
            name=name,
            gt_mask=data["gt_masks"][image_index],
            candidate_xy=data["xy"][image_index],
            candidate_scores=data["candidate_scores"][image_index],
            candidate_valid=data["valid"][image_index],
            query_probabilities=data["query_probabilities"][image_index],
            final_probability=final_probabilities[image_index],
            accepted=accepted[image_index],
            object_scores=object_scores[image_index],
            threshold=float(mask_threshold),
        )
    return accumulator.finalize()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_feature_path = Path(args.train_features).resolve()
    train_target_path = Path(args.train_targets).resolve()
    val_feature_path = Path(args.val_features).resolve()
    val_target_path = Path(args.val_targets).resolve()
    train = load_cache(train_feature_path, train_target_path, require_probs=False)
    val = load_cache(val_feature_path, val_target_path, require_probs=True)
    if train["descriptors"].shape[1:] != val["descriptors"].shape[1:]:
        raise RuntimeError("Train and validation feature shapes do not match")

    model = MicroQueryHead(
        input_dim=int(train["descriptors"].shape[-1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    descriptors_tensor = torch.from_numpy(train["descriptors"])
    valid_tensor = torch.from_numpy(train["valid"])
    primary_tensor = torch.from_numpy(train["primary"].astype(np.int64))
    quality_tensor = torch.from_numpy(train["query_iou"])
    dataset = TensorDataset(
        descriptors_tensor, valid_tensor, primary_tensor, quality_tensor
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    positive_count = int((train["primary"] & train["valid"]).sum())
    negative_count = int((~train["primary"] & train["valid"]).sum())
    class_weights = torch.tensor(
        [1.0, negative_count / max(1, positive_count)], device=device
    )
    history: list[dict] = []
    best_metric = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        object_total = 0.0
        quality_total = 0.0
        batch_count = 0
        for descriptors, valid, primary, quality_target in loader:
            descriptors = descriptors.to(device)
            valid = valid.to(device)
            primary = primary.to(device)
            quality_target = quality_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(descriptors, valid)
            object_loss = F.cross_entropy(
                output.object_logits[valid], primary[valid], weight=class_weights
            )
            quality_loss = F.smooth_l1_loss(
                torch.sigmoid(output.quality_logits[valid]), quality_target[valid]
            )
            loss = object_loss + float(args.quality_weight) * quality_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())
            object_total += float(object_loss.detach())
            quality_total += float(quality_loss.detach())
            batch_count += 1
        val_object, val_quality = predict_head(
            model, val["descriptors"], val["valid"], device
        )
        metrics = classification_metrics(val, val_object)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, batch_count),
            "train_object_loss": object_total / max(1, batch_count),
            "train_quality_loss": quality_total / max(1, batch_count),
            **metrics,
        }
        history.append(row)
        selection_metric = float(metrics["primary_auprc"])
        if math.isfinite(selection_metric) and selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(json.dumps(row, ensure_ascii=False))
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - start_time
    val_object, val_quality = predict_head(
        model, val["descriptors"], val["valid"], device
    )
    val_metrics = classification_metrics(val, val_object)
    train_object, train_quality = predict_head(
        model, train["descriptors"], train["valid"], device
    )
    train_metrics = classification_metrics(train, train_object)
    benchmark_descriptors = torch.from_numpy(val["descriptors"]).to(device)
    benchmark_valid = torch.from_numpy(val["valid"]).to(device)
    benchmark_repetitions = 200
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(10):
            model(benchmark_descriptors, benchmark_valid)
        if device.type == "cuda":
            torch.cuda.synchronize()
        benchmark_start = time.perf_counter()
        for _ in range(benchmark_repetitions):
            model(benchmark_descriptors, benchmark_valid)
        if device.type == "cuda":
            torch.cuda.synchronize()
    head_ms_per_image = (
        (time.perf_counter() - benchmark_start)
        * 1000.0
        / (benchmark_repetitions * len(val["names"]))
    )
    head_peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )

    risk_rows = [
        candidate_rejection_metrics(val, val_object, threshold)
        for threshold in args.thresholds
    ]
    eligible = [
        row
        for row in risk_rows
        if float(row["target_candidate_retention"]) >= 0.995
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["false_candidate_rejection"]),
                float(row["duplicate_suppression"]),
                float(row["object_threshold"]),
            ),
        )
    else:
        selected = max(
            risk_rows,
            key=lambda row: (
                float(row["target_candidate_retention"]),
                float(row["false_candidate_rejection"]),
            ),
        )
    selected_object_threshold = float(selected["object_threshold"])
    valid = val["valid"]
    accepted_all = valid.copy()
    accepted_correct = valid & (val_object >= selected_object_threshold)
    baseline_probability = aggregate_hard_gate(
        val["query_probabilities"], accepted_all
    )
    correct_probability = aggregate_hard_gate(
        val["query_probabilities"], accepted_correct
    )
    weighted_probability = aggregate_weighted(
        val["query_probabilities"], valid, val_object
    )
    quality_weighted_probability = aggregate_weighted(
        val["query_probabilities"], valid, val_object * val_quality
    )
    raw_weighted_probability = aggregate_weighted(
        val["query_probabilities"], valid, val["candidate_scores"]
    )

    mask_curve_rows: list[dict] = []
    fixed_results: dict[str, dict] = {}
    main_conditions = {
        "M2-1-independent-all": (baseline_probability, accepted_all, val["candidate_scores"]),
        "M2-2-object-hard-gate": (correct_probability, accepted_correct, val_object),
        "M2-2-object-weighted": (weighted_probability, valid, val_object),
        "M2-2-object-quality-weighted": (
            quality_weighted_probability,
            valid,
            val_object * val_quality,
        ),
        "Raw-candidate-weighted": (
            raw_weighted_probability,
            valid,
            val["candidate_scores"],
        ),
    }
    fixed_mask_threshold = min(
        args.thresholds, key=lambda value: abs(float(value) - 0.5)
    )
    for condition, (final_probability, accepted, scores) in main_conditions.items():
        for mask_threshold in args.thresholds:
            result = evaluate_mask_condition(
                val, final_probability, accepted, scores, mask_threshold
            )
            mask_curve_rows.append(
                {
                    "condition": condition,
                    "object_threshold": (
                        selected_object_threshold
                        if condition == "M2-2-object-hard-gate"
                        else float("nan")
                    ),
                    "mask_threshold": float(mask_threshold),
                    **result["summary"],
                }
            )
            if math.isclose(float(mask_threshold), fixed_mask_threshold):
                fixed_results[condition] = result

    # Counterfactuals hold coordinates, query masks, and capacity fixed; only
    # the candidate descriptor/objectness pairing is changed.
    zero_descriptors = np.zeros_like(val["descriptors"])
    zero_object, _ = predict_head(model, zero_descriptors, valid, device)
    shuffled_object = np.roll(val_object, shift=1, axis=0)
    wrong_object = 1.0 - val_object
    counterfactual_scores = {
        "CF-objectness-all-accept": np.ones_like(val_object),
        "CF-descriptor-zero": zero_object,
        "CF-descriptor-batch-shuffle": shuffled_object,
        "CF-objectness-wrong-inverted": wrong_object,
        "CF-objectness-all-reject": np.zeros_like(val_object),
    }
    counterfactual_rows: list[dict] = []
    for condition, scores in counterfactual_scores.items():
        accepted = valid & (scores >= selected_object_threshold)
        probability = aggregate_hard_gate(val["query_probabilities"], accepted)
        result = evaluate_mask_condition(
            val, probability, accepted, scores, fixed_mask_threshold
        )
        counterfactual_rows.append(
            {
                "condition": condition,
                "object_threshold": selected_object_threshold,
                "mask_threshold": fixed_mask_threshold,
                **result["summary"],
            }
        )
        fixed_results[condition] = result

    baseline = fixed_results["M2-1-independent-all"]["summary"]
    candidate = fixed_results["M2-2-object-hard-gate"]["summary"]
    object_auprc_delta = (
        float(val_metrics["semantic_auprc"])
        - float(val_metrics["raw_semantic_auprc"])
    )
    fa_reduction = (
        (float(baseline["fa"]) - float(candidate["fa"])) / float(baseline["fa"])
        if float(baseline["fa"]) > 0
        else float("nan")
    )
    ctr_delta = float(candidate["covered_target_recovery"]) - float(
        baseline["covered_target_recovery"]
    )
    tcr_delta = float(candidate["target_candidate_retention"]) - float(
        baseline["target_candidate_retention"]
    )
    gate_passed = bool(
        object_auprc_delta >= 0.03
        and fa_reduction >= 0.10
        and ctr_delta >= -0.005
        and tcr_delta >= -0.005
    )

    checkpoint_payload = {
        "model_state": best_state,
        "input_dim": int(train["descriptors"].shape[-1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "selected_object_threshold": selected_object_threshold,
        "seed": args.seed,
    }
    checkpoint_path = output_dir / "best_objectness.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    write_csv(output_dir / "training_history.csv", history)
    write_csv(output_dir / "risk_coverage.csv", risk_rows)
    write_csv(output_dir / "threshold_curve.csv", mask_curve_rows)
    write_csv(
        output_dir / "fixed_0_5_summary.csv",
        [
            {"condition": condition, **result["summary"]}
            for condition, result in fixed_results.items()
        ],
    )
    write_csv(output_dir / "counterfactuals.csv", counterfactual_rows)
    detail_dir = output_dir / "fixed_0_5_details"
    detail_dir.mkdir(exist_ok=True)
    for condition, result in fixed_results.items():
        write_csv(detail_dir / f"{condition}_per_image.csv", result["per_image_rows"])
        write_csv(detail_dir / f"{condition}_per_component.csv", result["per_component_rows"])
        write_csv(detail_dir / f"{condition}_per_query.csv", result["per_query_rows"])

    manifest = {
        "schema_version": 1,
        "experiment": "MicroQuery M2-S1 minimal objectness and quality sanity",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "seed": args.seed,
        "budget": int(val["valid"].shape[1]),
        "train_images": len(train["names"]),
        "val_images": len(val["names"]),
        "trainable_parameters": trainable_parameters,
        "frozen_candidate_generator": True,
        "frozen_image_encoder": True,
        "frozen_prompt_encoder": True,
        "frozen_mask_decoder": True,
        "best_epoch": best_epoch,
        "selected_object_threshold": selected_object_threshold,
        "fixed_mask_threshold": fixed_mask_threshold,
        "training_seconds": training_seconds,
        "head_ms_per_image_batch80": head_ms_per_image,
        "head_peak_gpu_memory_bytes_batch80": head_peak_gpu_memory_bytes,
        "train_feature_sha256": sha256_file(train_feature_path),
        "train_target_sha256": sha256_file(train_target_path),
        "val_feature_sha256": sha256_file(val_feature_path),
        "val_target_sha256": sha256_file(val_target_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "git_commit": current_git_commit(),
        "gt_boundary": (
            "MicroQueryHead.forward accepts only deployable ROI descriptors and candidate_valid. "
            "GT-derived primary and quality targets are loaded only by this training/evaluation script."
        ),
    }
    summary = {
        "manifest": manifest,
        "train_classification": train_metrics,
        "val_classification": val_metrics,
        "selected_rejection_operating_point": selected,
        "fixed_0_5": {
            condition: result["summary"] for condition, result in fixed_results.items()
        },
        "gate": {
            "objectness_auprc_delta_vs_raw_candidate": object_auprc_delta,
            "fa_relative_reduction_vs_M2_1": fa_reduction,
            "ctr_absolute_delta_vs_M2_1": ctr_delta,
            "tcr_absolute_delta_vs_M2_1": tcr_delta,
            "m2_2_gate_passed": gate_passed,
            "instruction": (
                "Proceed to M2-S2 100 epochs only if true"
                if gate_passed
                else "Stop M2 long training; retain diagnostic result only"
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
