#!/usr/bin/env python3
"""Validation threshold sweep, diagnostics and efficiency for full MicroQuery."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery import extract_candidate_roi_features
from efficient_sam.microquery_end2end import EndToEndMicroQueryHead
from efficient_sam.microquery_gate_deployment import GateDeploymentConfig
from efficient_sam.microquery_metrics import MicroQueryMetricAccumulator
from scripts.microquery_end2end_dataset import MicroQueryEndToEndDataset
from scripts.microquery_end2end_metrics import FullMaskMetricAccumulator
from scripts.microquery_end2end_runtime import (
    _decode_chunks,
    build_full_sam,
    encode_frozen_image,
    forward_deployable,
    load_checkpoint_state,
    trainable_parameter_counts,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val_split", default="splits/experiment1_seed20260825/val.txt")
    parser.add_argument("--val_candidate_cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--a1_checkpoint", default=DEFAULT_A1)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--query_chunk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bfloat16", "off"), default="bfloat16")
    parser.add_argument("--efficiency_repeats", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])


def build_from_checkpoint(checkpoint_path: Path, a1: Path, weights: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = str(checkpoint["variant"])
    model, _ = build_full_sam(a1, weights, device)
    head = None
    if variant != "c0_one_query":
        config = checkpoint["resolved_config"]
        shared = torch.load(Path(config["shared_head_init"]), map_location="cpu", weights_only=False)
        head = EndToEndMicroQueryHead(**shared["head_config"]).to(device)
    load_checkpoint_state(model, head, checkpoint["state"])
    model.eval()
    if head is not None:
        head.eval()
    return checkpoint, variant, model, head


@torch.no_grad()
def collect_predictions(model, head, variant, checkpoint_epoch, loader, device, args):
    rows = {
        "names": [], "probability": [], "query_probability": [], "gt": [], "xy": [],
        "raw": [], "valid": [], "semantic": [], "component_ids": [], "gates": [],
    }
    for batch in loader:
        deployable = {key: value.to(device) for key, value in batch["deployable"].items()}
        supervision = {key: value.to(device) for key, value in batch["supervision"].items()}
        with autocast_context(device, args.amp_dtype):
            output = forward_deployable(
                model,
                head,
                deployable,
                variant=variant,
                gate_deployment_config=(
                    GateDeploymentConfig("all_one")
                    if variant == "c1_independent_aux"
                    else GateDeploymentConfig("legacy_checkpoint_schedule")
                ),
                checkpoint_epoch=checkpoint_epoch,
                query_chunk=args.query_chunk,
            )
        rows["names"].extend(list(batch["meta"]["name"]))
        # Keep the final probability in float32 so a value close to the fixed
        # 0.5 checkpoint-selection threshold cannot cross it during caching.
        rows["probability"].append(output.final_probability.detach().float().cpu().numpy())
        rows["query_probability"].append(torch.sigmoid(output.query_logits).detach().float().cpu().numpy().astype(np.float16))
        rows["gt"].append(supervision["full_mask"].cpu().numpy().astype(np.uint8))
        rows["xy"].append(deployable["candidate_xy"].float().cpu().numpy())
        rows["raw"].append(deployable["candidate_scores"].float().cpu().numpy())
        rows["valid"].append(deployable["candidate_valid"].cpu().numpy().astype(bool))
        rows["semantic"].append(supervision["semantic_labels"].cpu().numpy().astype(bool))
        rows["component_ids"].append(supervision["component_ids"].cpu().numpy())
        if output.object_logits is None:
            rows["gates"].append(deployable["candidate_scores"].float().cpu().numpy())
        else:
            rows["gates"].append(output.raw_gates.detach().float().cpu().numpy())
    return {
        key: (np.asarray(value) if key == "names" else np.concatenate(value, axis=0))
        for key, value in rows.items()
    }


def evaluate_threshold(cache: dict, threshold: float) -> tuple[dict, list[dict]]:
    accumulator = FullMaskMetricAccumulator(threshold, 3.0)
    accumulator.update(
        names=cache["names"].tolist(),
        probabilities=torch.from_numpy(cache["probability"].astype(np.float32)),
        targets=torch.from_numpy(cache["gt"].astype(np.float32)),
        candidate_valid=torch.from_numpy(cache["valid"]),
        candidate_scores=torch.from_numpy(cache["raw"]),
        semantic_labels=torch.from_numpy(cache["semantic"]),
        object_scores=torch.from_numpy(cache["gates"]),
    )
    return accumulator.finalize(), accumulator.per_image


def candidate_diagnostics(cache: dict, variant: str, threshold: float):
    accumulator = MicroQueryMetricAccumulator(10)
    for index, name in enumerate(cache["names"]):
        query = cache["query_probability"][index].astype(np.float32)
        if variant == "c0_one_query":
            query = np.repeat(query[:1], 10, axis=0)
        accumulator.update(
            name=str(name),
            gt_mask=cache["gt"][index],
            candidate_xy=cache["xy"][index],
            candidate_scores=cache["raw"][index],
            candidate_valid=cache["valid"][index],
            query_probabilities=query,
            final_probability=cache["probability"][index].astype(np.float32),
            accepted=cache["valid"][index],
            object_scores=cache["gates"][index],
            threshold=threshold,
        )
    result = accumulator.finalize()
    image_index = {str(name): index for index, name in enumerate(cache["names"])}
    primary_by_component: dict[tuple[str, int], int] = {}
    for row in result["per_query_rows"]:
        if str(row["assignment"]) == "primary":
            primary_by_component[(str(row["image"]), int(row["component_index"]))] = int(
                row["candidate_rank"]
            ) - 1
    duplicate_overlap = []
    for row in result["per_query_rows"]:
        row["duplicate_primary_mask_iou"] = None
        if str(row["assignment"]) != "duplicate":
            continue
        key = (str(row["image"]), int(row["component_index"]))
        if key not in primary_by_component:
            continue
        index = image_index[key[0]]
        duplicate_index = int(row["candidate_rank"]) - 1
        primary_index = primary_by_component[key]
        query = cache["query_probability"][index].astype(np.float32)
        if variant == "c0_one_query":
            query = np.repeat(query[:1], 10, axis=0)
        first = query[duplicate_index] >= float(threshold)
        second = query[primary_index] >= float(threshold)
        union = int(np.logical_or(first, second).sum())
        overlap = int(np.logical_and(first, second).sum()) / union if union else 1.0
        row["duplicate_primary_mask_iou"] = float(overlap)
        duplicate_overlap.append(float(overlap))
    semantic = cache["semantic"] & cache["valid"]
    background = (~cache["semantic"]) & cache["valid"]
    gate = cache["gates"]
    result["summary"].update(
        {
            "covered_component_count": int(
                sum(int(row["covered_components"]) for row in result["per_image_rows"])
            ),
            "covered_detected_count": int(
                sum(int(row["covered_detected"]) for row in result["per_image_rows"])
            ),
            "uncovered_incidental_detection_count": int(
                sum(int(row["uncovered_detected"]) for row in result["per_image_rows"])
            ),
            "background_query_false_mask_pixels": int(
                sum(int(row["false_query_mask_pixels"]) for row in result["per_image_rows"])
            ),
            "gate_tcr_at_0_3": float((gate[semantic] >= 0.3).mean()) if semantic.any() else float("nan"),
            "gate_tcr_at_0_5": float((gate[semantic] >= 0.5).mean()) if semantic.any() else float("nan"),
            "gate_fcrr_at_0_3": float((gate[background] < 0.3).mean()) if background.any() else float("nan"),
            "gate_fcrr_at_0_5": float((gate[background] < 0.5).mean()) if background.any() else float("nan"),
            "c0_query_metric_note": (
                "not applicable: repeated shared one-query mask for file-shape compatibility"
                if variant == "c0_one_query" else None
            ),
            "duplicate_query_mask_overlap": (
                float(np.mean(duplicate_overlap)) if duplicate_overlap else float("nan")
            ),
        }
    )
    return result


def add_per_image_auprc(rows: list[dict], cache: dict) -> None:
    for index, row in enumerate(rows):
        labels = cache["gt"][index].reshape(-1).astype(np.uint8)
        scores = cache["probability"][index].reshape(-1).astype(np.float32)
        row["mask_auprc"] = float(average_precision_score(labels, scores)) if labels.any() else float("nan")


def area_bin_rows(component_rows: list[dict]) -> list[dict]:
    output = []
    bins = sorted({str(row["area_bin"]) for row in component_rows})
    for label in bins:
        rows = [row for row in component_rows if str(row["area_bin"]) == label]
        output.append(
            {
                "area_bin": label,
                "components": len(rows),
                "coverage": float(np.mean([int(row["covered"]) for row in rows])),
                "pd": float(np.mean([int(row["final_detected"]) for row in rows])),
                "best_query_iou": float(np.mean([float(row["best_query_iou"]) for row in rows])),
            }
        )
    return output


def benchmark(model, head, variant, dataset, device, args, epoch: int) -> dict:
    sample = dataset[0]
    deployable = {key: value.unsqueeze(0).to(device) for key, value in sample["deployable"].items()}
    images = deployable["image"]
    xy = deployable["candidate_xy"].to(images.dtype)
    valid = deployable["candidate_valid"]
    scores = deployable["candidate_scores"].to(images.dtype)
    labels = torch.where(valid, torch.ones_like(valid, dtype=torch.long), -torch.ones_like(valid, dtype=torch.long))
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    def elapsed(function, repeats):
        for _ in range(3):
            function()
        synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        synchronize()
        return (time.perf_counter() - start) * 1000.0 / repeats

    with torch.inference_mode(), autocast_context(device, args.amp_dtype):
        encoder_ms = elapsed(lambda: encode_frozen_image(model, images), args.efficiency_repeats)
        neck, interm, shallow = encode_frozen_image(model, images)
        if head is None:
            roi_ms = 0.0
        else:
            roi_ms = elapsed(
                lambda: head(
                    extract_candidate_roi_features(
                        shallow, neck, xy, scores, valid, input_h=256, input_w=256
                    ), valid
                ),
                args.efficiency_repeats,
            )
        if variant == "c0_one_query":
            points = torch.where(valid.unsqueeze(-1), xy, torch.zeros_like(xy)).unsqueeze(1)
            point_labels = labels.unsqueeze(1)
            tokens = None
        else:
            points = torch.where(valid.unsqueeze(-1), xy, torch.zeros_like(xy)).unsqueeze(2)
            point_labels = labels.unsqueeze(2)
            tokens = None
            if variant == "f2_gate_token":
                descriptor = extract_candidate_roi_features(
                    shallow, neck, xy, scores, valid, input_h=256, input_w=256
                )
                tokens = head(descriptor, valid).candidate_token
        decoder_ms = elapsed(
            lambda: _decode_chunks(
                model, neck, interm, points, point_labels, output_h=256, output_w=256,
                query_chunk=(1 if variant == "c0_one_query" else args.query_chunk),
                candidate_tokens=tokens,
            ),
            max(5, args.efficiency_repeats // 2),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        end_to_end_ms = elapsed(
            lambda: forward_deployable(
                model,
                head,
                deployable,
                variant=variant,
                gate_deployment_config=(
                    GateDeploymentConfig("all_one")
                    if variant == "c1_independent_aux"
                    else GateDeploymentConfig("legacy_checkpoint_schedule")
                ),
                checkpoint_epoch=epoch,
                query_chunk=args.query_chunk,
            ),
            max(5, args.efficiency_repeats // 2),
        )
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    counts = trainable_parameter_counts(model, head)
    return {
        "total_parameters": counts["total_model"] + counts["head"],
        "trainable_parameters": counts["head"] + counts["prompt_encoder"] + counts["mask_decoder"],
        "encoder_ms_per_image": encoder_ms,
        "probe_ms_per_image": 0.0,
        "probe_note": "coordinates loaded from frozen cache; no online probe call",
        "roi_head_ms_per_image": roi_ms,
        "prompt_encoder_decoder_ms_per_image": decoder_ms,
        "decoder_calls_per_image": 1 if variant == "c0_one_query" else 10,
        "peak_gpu_memory_bytes": int(peak),
        "end_to_end_latency_ms_per_image": end_to_end_ms,
        "repeats": args.efficiency_repeats,
        "batch_size": 1,
    }


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    device = torch.device(args.device)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_repo_path(args.data_root)
    val_split = Path(args.val_split).resolve() if Path(args.val_split).is_absolute() else data_root / args.val_split
    val_cache = resolve_repo_path(args.val_candidate_cache)
    a1 = resolve_repo_path(args.a1_checkpoint)
    weights = resolve_repo_path(args.weights)
    checkpoint, variant, model, head = build_from_checkpoint(checkpoint_path, a1, weights, device)
    dataset = MicroQueryEndToEndDataset(
        data_root=data_root, split=val_split, candidate_cache=val_cache, augment=False,
        budget=10, seed=args.seed,
    )
    if len(dataset) != 80:
        raise RuntimeError("validation split must contain exactly 80 images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    cache = collect_predictions(model, head, variant, int(checkpoint["epoch"]), loader, device, args)
    np.savez_compressed(
        output_dir / "evaluation_cache.npz",
        **cache,
        variant=np.asarray(variant),
        checkpoint_epoch=np.asarray(int(checkpoint["epoch"])),
    )
    thresholds = [round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05)]
    curve = []
    per_image_by_threshold = {}
    for threshold in thresholds:
        metrics, per_image = evaluate_threshold(cache, threshold)
        curve.append({"threshold": threshold, **metrics})
        per_image_by_threshold[threshold] = per_image
    fixed = next(row for row in curve if abs(float(row["threshold"]) - 0.5) < 1e-8)
    selected = max(curve, key=lambda row: (float(row["global_iou"]), float(row["f1"]), -abs(float(row["threshold"]) - 0.5)))
    selected_threshold = float(selected["threshold"])
    per_image = per_image_by_threshold[selected_threshold]
    add_per_image_auprc(per_image, cache)
    diagnostic = candidate_diagnostics(cache, variant, selected_threshold)
    diagnostic_by_image = {
        str(row["image"]): row for row in diagnostic["per_image_rows"]
    }
    for row in per_image:
        candidate_row = diagnostic_by_image.get(str(row["image"]), {})
        for key in (
            "components",
            "covered_components",
            "covered_detected",
            "uncovered_detected",
            "background_queries",
            "false_query_mask_pixels",
        ):
            if key in candidate_row:
                row[key] = candidate_row[key]
    summary = {
        "schema_version": 1,
        "variant": variant,
        "split_role": "validation",
        "test_split_read": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "fixed_0_5": json_safe(fixed),
        "validation_selected": json_safe(selected),
        "selected_threshold": selected_threshold,
        "candidate_conditional": json_safe(diagnostic["summary"]),
    }
    efficiency = benchmark(model, head, variant, dataset, device, args, int(checkpoint["epoch"]))
    write_csv(output_dir / "threshold_curve.csv", curve)
    write_csv(output_dir / "pd_fa_curve.csv", [
        {key: row[key] for key in ("threshold", "pd", "fa", "fa_per_million", "global_iou", "f1")}
        for row in curve
    ])
    write_csv(output_dir / "per_image.csv", per_image)
    write_csv(output_dir / "per_component.csv", diagnostic["per_component_rows"])
    write_csv(output_dir / "per_query.csv", diagnostic["per_query_rows"])
    write_csv(output_dir / "area_bin_metrics.csv", area_bin_rows(diagnostic["per_component_rows"]))
    (output_dir / "fixed05_metrics.json").write_text(
        json.dumps(json_safe(fixed), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "efficiency.json").write_text(
        json.dumps(json_safe(efficiency), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**summary, "efficiency": efficiency}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
