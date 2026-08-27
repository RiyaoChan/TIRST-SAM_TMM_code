#!/usr/bin/env python3
"""Causal gate/token/coordinate counterfactuals for trained F1/F2 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_microquery_end2end import build_from_checkpoint, write_csv
from scripts.microquery_end2end_dataset import MicroQueryEndToEndDataset
from scripts.microquery_end2end_metrics import FullMaskMetricAccumulator
from scripts.microquery_end2end_runtime import forward_deployable
from scripts.train_microquery_end2end import (
    DEFAULT_A1,
    DEFAULT_DATA_ROOT,
    DEFAULT_VAL_CACHE,
    DEFAULT_WEIGHTS,
    autocast_context,
    json_safe,
    resolve_repo_path,
    set_deterministic,
    sha256_file,
)


COUNTERFACTUAL_EFFECT_MARGIN = 0.005
COUNTERFACTUAL_METRICS = ("selected_global_iou", "selected_mean_niou", "selected_mask_auprc")


def counterfactual_effect(correct: dict, intervention: dict, margin: float = COUNTERFACTUAL_EFFECT_MARGIN) -> dict:
    """Require a visible segmentation effect, not a floating-point ordering accident."""

    deltas = {
        metric.removeprefix("selected_"): float(correct[metric]) - float(intervention[metric])
        for metric in COUNTERFACTUAL_METRICS
    }
    ordering_pass = all(value > 0.0 for value in deltas.values())
    return {
        "condition": intervention["condition"],
        "deltas_correct_minus_intervention": deltas,
        "max_segmentation_delta": max(deltas.values()),
        "ordering_pass": ordering_pass,
        "effect_margin": margin,
        "meaningful_effect_pass": ordering_pass and max(deltas.values()) >= margin,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--main_evaluation_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val_split", default="splits/experiment1_seed20260825/val.txt")
    parser.add_argument("--val_candidate_cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--a1_checkpoint", default=DEFAULT_A1)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--query_chunk", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bfloat16", "off"), default="bfloat16")
    return parser.parse_args()


def random_background_coordinates(
    xy: torch.Tensor, targets: torch.Tensor, seed: int, image_indices: torch.Tensor
) -> torch.Tensor:
    """Create a GT-informed diagnostic intervention outside the deployable forward."""

    output = xy.detach().cpu().clone()
    masks = targets.detach().cpu().numpy() > 0.5
    for batch_index in range(output.shape[0]):
        rng = np.random.default_rng(int(seed) + int(image_indices[batch_index]) * 7919)
        background_yx = np.argwhere(~masks[batch_index])
        chosen = background_yx[rng.integers(0, len(background_yx), size=output.shape[1])]
        output[batch_index, :, 0] = torch.from_numpy(chosen[:, 1].astype(np.float32))
        output[batch_index, :, 1] = torch.from_numpy(chosen[:, 0].astype(np.float32))
    return output.to(xy.device, dtype=xy.dtype)


def conditions_for_variant(variant: str) -> list[dict]:
    conditions = [{"family": "baseline", "condition": "correct"}]
    if variant == "f1_soft_gate":
        conditions.extend(
            {"family": "gate", "condition": name}
            for name in ("all_one", "zero", "batch_shuffled", "candidate_shuffled", "inverted")
        )
    if variant == "f2_gate_token":
        conditions.extend(
            {"family": "token", "condition": name}
            for name in ("zero", "batch_shuffled", "candidate_shuffled", "random", "coordinate_only")
        )
    conditions.extend(
        {"family": "coordinate", "condition": name}
        for name in ("candidate_shuffled", "invalid", "random_background")
    )
    return conditions


@torch.no_grad()
def evaluate_condition(
    model, head, variant, epoch, loader, device, args, spec, threshold
) -> tuple[dict, list[dict]]:
    selected_accumulator = FullMaskMetricAccumulator(threshold, 3.0)
    fixed_accumulator = FullMaskMetricAccumulator(0.5, 3.0)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + sum(ord(value) for value in spec["family"] + spec["condition"])
    )
    for batch in loader:
        deployable = {key: value.to(device) for key, value in batch["deployable"].items()}
        supervision = {key: value.to(device) for key, value in batch["supervision"].items()}
        gate_condition = spec["condition"] if spec["family"] == "gate" else "correct"
        token_condition = spec["condition"] if spec["family"] == "token" else "correct"
        coordinate_condition = spec["condition"] if spec["family"] == "coordinate" else "correct"
        coordinate_override = None
        if coordinate_condition == "random_background":
            coordinate_override = random_background_coordinates(
                deployable["candidate_xy"],
                supervision["full_mask"],
                args.seed,
                batch["meta"]["index"],
            )
            coordinate_condition = "correct"
        with autocast_context(device, args.amp_dtype):
            output = forward_deployable(
                model,
                head,
                deployable,
                variant=variant,
                epoch=epoch,
                query_chunk=args.query_chunk,
                gate_condition=gate_condition,
                token_condition=token_condition,
                coordinate_condition=coordinate_condition,
                coordinate_override=coordinate_override,
                generator=generator,
            )
        object_scores = output.raw_gates if output.object_logits is not None else None
        update = dict(
            names=list(batch["meta"]["name"]),
            probabilities=output.final_probability,
            targets=supervision["full_mask"],
            candidate_valid=deployable["candidate_valid"],
            candidate_scores=deployable["candidate_scores"],
            semantic_labels=supervision["semantic_labels"],
            object_scores=object_scores,
        )
        selected_accumulator.update(**update)
        fixed_accumulator.update(**update)
    selected = selected_accumulator.finalize()
    fixed = fixed_accumulator.finalize()
    rows = []
    for row in selected_accumulator.per_image:
        rows.append({"family": spec["family"], "condition": spec["condition"], **row})
    return {
        "family": spec["family"],
        "condition": spec["condition"],
        "selected_threshold": threshold,
        **{f"selected_{key}": value for key, value in selected.items()},
        **{f"fixed05_{key}": value for key, value in fixed.items()},
    }, rows


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    summary_path = resolve_repo_path(args.main_evaluation_summary)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_root = resolve_repo_path(args.data_root)
    val_split = Path(args.val_split).resolve() if Path(args.val_split).is_absolute() else data_root / args.val_split
    val_cache = resolve_repo_path(args.val_candidate_cache)
    checkpoint, variant, model, head = build_from_checkpoint(
        checkpoint_path,
        resolve_repo_path(args.a1_checkpoint),
        resolve_repo_path(args.weights),
        device,
    )
    if variant not in {"f1_soft_gate", "f2_gate_token"}:
        raise ValueError("counterfactual evaluation is defined for F1 or F2")
    with summary_path.open(encoding="utf-8") as handle:
        main_summary = json.load(handle)
    threshold = float(main_summary["selected_threshold"])
    dataset = MicroQueryEndToEndDataset(
        data_root=data_root, split=val_split, candidate_cache=val_cache,
        augment=False, budget=10, seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metric_rows = []
    per_image_rows = []
    for spec in conditions_for_variant(variant):
        metrics, per_image = evaluate_condition(
            model, head, variant, int(checkpoint["epoch"]), loader, device, args, spec, threshold
        )
        metric_rows.append(metrics)
        per_image_rows.extend(per_image)
        print(json.dumps(json_safe(metrics), ensure_ascii=False), flush=True)
    lookup = {(row["family"], row["condition"]): row for row in metric_rows}
    correct = lookup[("baseline", "correct")]
    if variant == "f1_soft_gate":
        mechanism_rows = [
            counterfactual_effect(correct, lookup[("gate", name)])
            for name in ("all_one", "batch_shuffled", "candidate_shuffled", "inverted")
        ]
    else:
        mechanism_rows = [
            counterfactual_effect(correct, lookup[("token", name)])
            for name in ("zero", "batch_shuffled", "candidate_shuffled")
        ]
    coordinate_rows = [
        counterfactual_effect(correct, lookup[("coordinate", name)])
        for name in ("candidate_shuffled", "invalid", "random_background")
    ]
    mechanism_ordering = all(row["ordering_pass"] for row in mechanism_rows)
    mechanism = all(row["meaningful_effect_pass"] for row in mechanism_rows)
    coordinate_ordering = all(row["ordering_pass"] for row in coordinate_rows)
    coordinate_mechanism = all(row["meaningful_effect_pass"] for row in coordinate_rows)
    summary = {
        "schema_version": 1,
        "variant": variant,
        "split_role": "validation",
        "test_split_read": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "main_selected_threshold": threshold,
        "counterfactual_effect_margin": COUNTERFACTUAL_EFFECT_MARGIN,
        "counterfactual_effect_rule": "For every required intervention, correct must improve all of global IoU, mean nIoU, and mask AUPRC, with at least one improvement >=0.005 at the shared validation-selected threshold.",
        "mechanism_counterfactual_ordering_pass": mechanism_ordering,
        "mechanism_counterfactual_pass": mechanism,
        "mechanism_counterfactual_effects": mechanism_rows,
        "coordinate_counterfactual_ordering_pass": coordinate_ordering,
        "coordinate_counterfactual_pass": coordinate_mechanism,
        "coordinate_counterfactual_effects": coordinate_rows,
        "random_background_note": "GT is used only by this offline diagnostic to construct background-coordinate interventions; forward receives coordinates only.",
        "conditions": json_safe(metric_rows),
    }
    write_csv(output_dir / f"{variant}_counterfactual_metrics.csv", metric_rows)
    write_csv(output_dir / f"{variant}_counterfactual_per_image.csv", per_image_rows)
    (output_dir / f"{variant}_counterfactual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
