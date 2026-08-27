#!/usr/bin/env python3
"""Train the sequential B1--B4 MicroQuery component-safe semantic/utility head."""

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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery_component_safe import (
    ComponentSafeMicroQueryHead,
    GroupingConfig,
    build_candidate_graph,
    component_survival_loss,
    connected_candidate_groups,
    pairwise_candidate_relations,
    pairwise_representative_loss,
    semantic_label_from_roles,
)
from efficient_sam.microquery_metrics import expected_calibration_error
from scripts.eval_microquery_component_safe_cache import (
    bootstrap_delta,
    candidate_metrics,
    evaluate_pattern,
    gate_passed,
    load_data,
    predict_objectness,
    randomize_groups,
    write_csv,
)
from scripts.eval_prompt_quality import sha256_file
from scripts.train_experiment1_single_view import set_deterministic
from scripts.train_microquery_m2 import load_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--train_targets", required=True)
    parser.add_argument("--val_features", required=True)
    parser.add_argument("--val_targets", required=True)
    parser.add_argument("--old_objectness_checkpoint", required=True)
    parser.add_argument("--one_query_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage", choices=("b1_b2", "b3", "b4"), default="b1_b2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--quality_weight", type=float, default=0.5)
    parser.add_argument("--rank_weight", type=float, default=0.2)
    parser.add_argument("--coverage_weight", type=float, default=0.5)
    parser.add_argument("--brier_weight", type=float, default=0.1)
    parser.add_argument("--rank_margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--dataset", default="IRSTD-1k")
    return parser.parse_args()


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def safe_metric(function, labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(function(labels.astype(np.int64), scores.astype(np.float64)))


def classification_metrics(data: dict, semantic: np.ndarray) -> dict:
    valid = data["valid"]
    labels = data["semantic"][valid].astype(np.int64)
    scores = semantic[valid]
    raw = data["raw"][valid]
    return {
        "semantic_auprc": safe_metric(average_precision_score, labels, scores),
        "semantic_auroc": safe_metric(roc_auc_score, labels, scores),
        "semantic_brier": float(np.mean((scores - labels) ** 2)),
        "semantic_ece": expected_calibration_error(labels, scores),
        "raw_semantic_auprc": safe_metric(average_precision_score, labels, raw),
    }


def predict(
    model: ComponentSafeMicroQueryHead,
    descriptors: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    semantic_rows: list[np.ndarray] = []
    utility_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(descriptors), 128):
            end = min(len(descriptors), start + 128)
            output = model(
                torch.from_numpy(descriptors[start:end]).to(device),
                torch.from_numpy(valid[start:end]).to(device),
            )
            semantic_rows.append(
                torch.softmax(output.semantic_logits, dim=-1)[..., 1].cpu().numpy()
            )
            utility_rows.append(torch.sigmoid(output.utility_logits).cpu().numpy())
    return np.concatenate(semantic_rows), np.concatenate(utility_rows)


def build_groups(data: dict) -> list[tuple[tuple[int, ...], ...]]:
    config = GroupingConfig(
        "hybrid", r_near=2.0, r_far=8.0, tau_iou=0.2, r_mask=5.0
    )
    rows = []
    for index in range(len(data["names"])):
        relations = pairwise_candidate_relations(
            data["xy"][index], data["query"][index], data["descriptors"][index]
        )
        graph = build_candidate_graph(relations, data["valid"][index], config)
        rows.append(connected_candidate_groups(graph, data["valid"][index]))
    return rows


def select_candidates(
    semantic: np.ndarray,
    utility: np.ndarray,
    valid: np.ndarray,
    groups: list[tuple[tuple[int, ...], ...]],
    threshold: float,
    *,
    group_safe: bool,
    use_utility: bool,
) -> np.ndarray:
    if not group_safe:
        return valid & (semantic >= float(threshold))
    accepted = np.zeros_like(valid, dtype=bool)
    champion_score = semantic * utility if use_utility else semantic
    for image_index, image_groups in enumerate(groups):
        for group in image_groups:
            members = [int(value) for value in group if valid[image_index, value]]
            if not members or max(float(semantic[image_index, value]) for value in members) < threshold:
                continue
            champion = max(
                members,
                key=lambda value: (
                    float(champion_score[image_index, value]),
                    float(semantic[image_index, value]),
                    -value,
                ),
            )
            accepted[image_index, champion] = True
    return accepted


def selection_key(row: dict) -> tuple:
    return (
        -int(row["fully_lost_covered_components"]),
        float(row["target_candidate_retention"]),
        int(float(row["false_candidate_rejection"]) >= 0.70),
        -float(row["fa"]),
        float(row["covered_target_recovery"]),
        float(row["mean_niou"]),
    )


def evaluate_thresholds(
    data: dict,
    semantic: np.ndarray,
    utility: np.ndarray,
    groups: list[tuple[tuple[int, ...], ...]],
    thresholds: tuple[float, ...],
    mask_threshold: float,
    *,
    use_utility: bool,
) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    results: dict[tuple[str, float], dict] = {}
    for condition, group_safe in (("B1-no-group", False), ("B2-group-safe", True)):
        for threshold in thresholds:
            accepted = select_candidates(
                semantic,
                utility,
                data["valid"],
                groups,
                threshold,
                group_safe=group_safe,
                use_utility=use_utility,
            )
            result = evaluate_pattern(
                data, semantic, accepted, accepted.astype(np.float32), mask_threshold
            )
            row = {
                "condition": condition,
                "semantic_threshold": float(threshold),
                **result["summary"],
            }
            rows.append(row)
            results[(condition, float(threshold))] = result
    b2_rows = [row for row in rows if row["condition"] == "B2-group-safe"]
    selected = max(b2_rows, key=selection_key)
    result = results[(selected["condition"], float(selected["semantic_threshold"]))]
    return rows, selected, result


def checkpoint_payload(
    model: ComponentSafeMicroQueryHead, args: argparse.Namespace, epoch: int, **extra
) -> dict:
    return {
        "model_state": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "input_dim": model.input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "stage": args.stage,
        "epoch": epoch,
        "seed": args.seed,
        **extra,
    }


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_feature_path = Path(args.train_features).resolve()
    train_target_path = Path(args.train_targets).resolve()
    val_feature_path = Path(args.val_features).resolve()
    val_target_path = Path(args.val_targets).resolve()
    old_checkpoint_path = Path(args.old_objectness_checkpoint).resolve()
    train = load_cache(train_feature_path, train_target_path, require_probs=False)
    val = load_data(val_feature_path, val_target_path)
    groups = build_groups(val)
    model = ComponentSafeMicroQueryHead(
        input_dim=int(train["descriptors"].shape[-1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters > 500_000:
        raise RuntimeError(f"Head has {parameters} parameters; plan limit is 500000")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    semantic_target = semantic_label_from_roles(
        train["primary"], train["duplicate"], train["valid"]
    )
    if not np.array_equal(semantic_target.astype(bool), train["semantic"] & train["valid"]):
        raise RuntimeError("Cached semantic labels disagree with primary|duplicate roles")
    dataset = TensorDataset(
        torch.from_numpy(train["descriptors"]),
        torch.from_numpy(train["valid"]),
        torch.from_numpy(semantic_target),
        torch.from_numpy(train["component_index"]),
        torch.from_numpy(train["query_iou"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    positive_count = int((train["semantic"] & train["valid"]).sum())
    negative_count = int((~train["semantic"] & train["valid"]).sum())
    class_weights = torch.tensor(
        [1.0, negative_count / max(1, positive_count)], device=device
    )
    use_utility = args.stage in {"b3", "b4"}
    use_coverage = args.stage == "b4"
    history: list[dict] = []
    best_semantic_metric = -math.inf
    best_component_key: tuple | None = None
    best_semantic_state: dict | None = None
    best_component_state: dict | None = None
    coarse_thresholds = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "semantic": 0.0, "quality": 0.0, "rank": 0.0, "coverage": 0.0, "brier": 0.0}
        count = 0
        for descriptors, valid, target, component, query_iou in loader:
            descriptors = descriptors.to(device)
            valid = valid.to(device)
            target = target.to(device)
            component = component.to(device)
            query_iou = query_iou.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(descriptors, valid)
            semantic_loss = F.cross_entropy(
                output.semantic_logits[valid], target[valid], weight=class_weights
            )
            probability = torch.softmax(output.semantic_logits, dim=-1)[..., 1]
            brier = ((probability[valid] - target[valid].float()) ** 2).mean()
            semantic_bool = target.bool()
            if use_utility:
                target_valid = valid & semantic_bool
                quality_loss = F.smooth_l1_loss(
                    torch.sigmoid(output.utility_logits[target_valid]),
                    query_iou[target_valid],
                )
                rank_loss = pairwise_representative_loss(
                    output.utility_logits,
                    query_iou,
                    component,
                    semantic_bool,
                    valid,
                    margin=args.rank_margin,
                )
            else:
                quality_loss = output.utility_logits.sum() * 0.0
                rank_loss = output.utility_logits.sum() * 0.0
            if use_coverage:
                coverage_loss = component_survival_loss(
                    output.semantic_logits[..., 1] - output.semantic_logits[..., 0],
                    component,
                    semantic_bool,
                    valid,
                )
            else:
                coverage_loss = output.semantic_logits.sum() * 0.0
            loss = semantic_loss + args.brier_weight * brier
            if use_utility:
                loss = loss + args.quality_weight * quality_loss + args.rank_weight * rank_loss
            if use_coverage:
                loss = loss + args.coverage_weight * coverage_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            for name, value in (
                ("loss", loss), ("semantic", semantic_loss), ("quality", quality_loss),
                ("rank", rank_loss), ("coverage", coverage_loss), ("brier", brier),
            ):
                totals[name] += float(value.detach())
            count += 1
        semantic, utility = predict(model, val["descriptors"], val["valid"], device)
        classification = classification_metrics(val, semantic)
        _, selected, _ = evaluate_thresholds(
            val,
            semantic,
            utility,
            groups,
            coarse_thresholds,
            args.mask_threshold,
            use_utility=use_utility,
        )
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / max(1, count) for name, value in totals.items()},
            **classification,
            **{f"selected_{name}": value for name, value in selected.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if classification["semantic_auprc"] > best_semantic_metric:
            best_semantic_metric = classification["semantic_auprc"]
            best_semantic_state = checkpoint_payload(model, args, epoch)
        key = selection_key(selected)
        if best_component_key is None or key > best_component_key:
            best_component_key = key
            best_component_state = checkpoint_payload(
                model,
                args,
                epoch,
                selected_semantic_threshold=float(selected["semantic_threshold"]),
                selection_key=list(key),
            )
        torch.save(checkpoint_payload(model, args, epoch), output_dir / "last.pt")
    if best_semantic_state is None or best_component_state is None:
        raise RuntimeError("training did not produce a selectable checkpoint")
    torch.save(best_semantic_state, output_dir / "best_semantic_auprc.pt")
    torch.save(best_component_state, output_dir / "best_component_safe.pt")
    model.load_state_dict(best_component_state["model_state"], strict=True)
    semantic, utility = predict(model, val["descriptors"], val["valid"], device)
    thresholds = tuple(round(value, 2) for value in np.arange(0.01, 1.0, 0.01))
    threshold_rows, selected, selected_result = evaluate_thresholds(
        val,
        semantic,
        utility,
        groups,
        thresholds,
        args.mask_threshold,
        use_utility=use_utility,
    )
    selected_threshold = float(selected["semantic_threshold"])

    def run_condition(
        name: str,
        scores: np.ndarray,
        utilities: np.ndarray,
        condition_groups: list[tuple[tuple[int, ...], ...]],
        group_safe: bool = True,
        utility_enabled: bool = use_utility,
    ) -> tuple[dict, dict]:
        accepted = select_candidates(
            scores,
            utilities,
            val["valid"],
            condition_groups,
            selected_threshold,
            group_safe=group_safe,
            use_utility=utility_enabled,
        )
        result = evaluate_pattern(
            val, scores, accepted, accepted.astype(np.float32), args.mask_threshold
        )
        return {"condition": name, **result["summary"]}, result

    counterfactual_rows: list[dict] = []
    results: dict[str, dict] = {}
    conditions = [
        ("Correct", semantic, utility, groups, True, use_utility),
        ("Shuffled-descriptor", np.roll(semantic, 1, axis=0), np.roll(utility, 1, axis=0), groups, True, use_utility),
        ("Inverted-semantic", 1.0 - semantic, utility, groups, True, use_utility),
        ("Random-groups", semantic, utility, randomize_groups(groups, val["valid"], args.seed), True, use_utility),
        ("No-grouping", semantic, utility, groups, False, use_utility),
        ("Semantic-only", semantic, np.ones_like(utility), groups, True, False),
    ]
    zero_semantic, zero_utility = predict(
        model, np.zeros_like(val["descriptors"]), val["valid"], device
    )
    conditions.append(("Zero-descriptor", zero_semantic, zero_utility, groups, True, use_utility))
    if use_utility:
        conditions.extend(
            [
                ("Shuffled-utility", semantic, np.roll(utility, 1, axis=0), groups, True, True),
                ("Zero-utility", semantic, np.ones_like(utility), groups, True, True),
            ]
        )
    for values in conditions:
        row, result = run_condition(*values)
        counterfactual_rows.append(row)
        results[row["condition"]] = result

    old_scores, _ = predict_objectness(val["descriptors"], val["valid"], old_checkpoint_path)
    old_accepted = val["valid"] & (old_scores >= 0.15)
    old_result = evaluate_pattern(
        val, old_scores, old_accepted, old_accepted.astype(np.float32), args.mask_threshold
    )
    independent_accepted = val["valid"].copy()
    independent_result = evaluate_pattern(
        val,
        val["raw"],
        independent_accepted,
        independent_accepted.astype(np.float32),
        args.mask_threshold,
    )
    bootstrap_rows = bootstrap_delta(
        old_result["per_image_rows"],
        selected_result["per_image_rows"],
        args.bootstrap_samples,
        args.seed,
    )
    with Path(args.one_query_summary).open(encoding="utf-8") as handle:
        one_rows = list(csv.DictReader(handle))
    one_row = next(
        row for row in one_rows if row["budget"] == "10" and row["condition"] == "M0-One"
    )
    classification = classification_metrics(val, semantic)
    semantic_delta = classification["semantic_auprc"] - classification["raw_semantic_auprc"]
    cache_gate = gate_passed(selected, independent_result["summary"], one_row)
    mechanism_gate = bool(
        semantic_delta >= 0.03
        and float(results["Correct"]["summary"]["global_iou"])
        > float(results["Inverted-semantic"]["summary"]["global_iou"])
        and float(results["Correct"]["summary"]["global_iou"])
        > float(results["Shuffled-descriptor"]["summary"]["global_iou"])
    )
    write_csv(output_dir / "training_history.csv", history)
    write_csv(output_dir / "threshold_curve.csv", threshold_rows)
    write_csv(output_dir / "counterfactuals.csv", counterfactual_rows)
    write_csv(output_dir / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(output_dir / "per_image.csv", selected_result["per_image_rows"])
    write_csv(output_dir / "per_component.csv", selected_result["per_component_rows"])
    write_csv(output_dir / "per_candidate.csv", selected_result["per_query_rows"])
    manifest = {
        "schema_version": 1,
        "experiment": f"MicroQuery M2-S1b {args.stage}",
        "dataset": args.dataset,
        "stage": args.stage,
        "epochs": args.epochs,
        "seed": args.seed,
        "trainable_parameters": parameters,
        "grouping": "G2_rn2_rf8_iou0.2_rm5",
        "best_epoch": int(best_component_state["epoch"]),
        "selected_semantic_threshold": selected_threshold,
        "train_feature_sha256": sha256_file(train_feature_path),
        "train_target_sha256": sha256_file(train_target_path),
        "val_feature_sha256": sha256_file(val_feature_path),
        "val_target_sha256": sha256_file(val_target_path),
        "old_objectness_checkpoint_sha256": sha256_file(old_checkpoint_path),
        "best_component_safe_sha256": sha256_file(output_dir / "best_component_safe.pt"),
        "git_commit": current_git_commit(),
        "training_seconds": time.perf_counter() - start_time,
        "frozen_encoder_probe_sam": True,
        "gt_boundary": "GT component membership and query IoU are training/evaluation targets only. Head.forward and grouping consume deployable descriptors, validity, coordinates, and query masks only.",
    }
    summary = {
        "manifest": manifest,
        "classification": classification,
        "semantic_auprc_delta_vs_raw": semantic_delta,
        "selected": selected,
        "independent_all": independent_result["summary"],
        "old_hard_0_15": old_result["summary"],
        "counterfactuals": counterfactual_rows,
        "cache_gate_before_hard_negative": cache_gate,
        "mechanism_counterfactual_gate": mechanism_gate,
        "decision": (
            "Proceed to hard-negative validation; do not start a longer run yet."
            if cache_gate and mechanism_gate
            else ("Proceed to B3." if args.stage == "b1_b2" else "Proceed to B4 or stop according to the sequential gate.")
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
