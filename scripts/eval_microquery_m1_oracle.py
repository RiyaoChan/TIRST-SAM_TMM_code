#!/usr/bin/env python3
"""Evaluate NOT-DEPLOYABLE oracle point/box/micro-mask prompt upper bounds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.microquery import aggregate_query_probabilities, decode_prompt_queries
from efficient_sam.microquery_metrics import MicroQueryMetricAccumulator
from efficient_sam.prompt_metrics import extract_components
from scripts.eval_prompt_quality import sha256_file
from scripts.train_experiment1_single_view import set_deterministic
from sirst_dataset import make_loader


CONDITIONS = (
    "M1-Null",
    "M1-P-OneQuery",
    "M1-P",
    "M1-PN",
    "M1-B",
    "M1-M",
    "M1-PM",
    "M1-BM",
)


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


def clean_background_points(
    component_mask: np.ndarray, full_gt: np.ndarray, count: int = 4
) -> list[tuple[float, float]]:
    ys, xs = np.nonzero(component_mask)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    cx = float(xs.mean())
    cy = float(ys.mean())
    starts = [
        (xmin - 2, cy, -1, 0),
        (xmax + 2, cy, 1, 0),
        (cx, ymin - 2, 0, -1),
        (cx, ymax + 2, 0, 1),
    ]
    height, width = full_gt.shape
    result = []
    for x_start, y_start, dx, dy in starts[:count]:
        x, y = float(x_start), float(y_start)
        for _ in range(max(height, width)):
            xi = int(np.clip(round(x), 0, width - 1))
            yi = int(np.clip(round(y), 0, height - 1))
            if not bool(full_gt[yi, xi]):
                result.append((float(xi), float(yi)))
                break
            x += dx
            y += dy
        else:
            result.append((float(np.clip(round(x_start), 0, width - 1)), float(np.clip(round(y_start), 0, height - 1))))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="IRSTD-1k")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--query_chunk", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--save_probability_maps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    thresholds = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    checkpoint_path = Path(args.checkpoint).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(weights_path),
        use_adapter=False,
        return_encoder_multi_scale=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    loader = make_loader(
        str(data_root),
        str(split_path),
        batch_size=1,
        size=args.size,
        augment=False,
        keep_ratio_pad=False,
        workers=args.workers,
        shuffle=False,
        mask_suffix="",
        sctransnet_preproc=True,
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_eval_crop="resize",
        mllm_features_path=None,
    )
    accumulators = {
        (condition, threshold): MicroQueryMetricAccumulator(args.max_queries)
        for condition in CONDITIONS
        for threshold in thresholds
    }
    final_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    query_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    names: list[str] = []
    gt_maps: list[np.ndarray] = []
    condition_seconds = defaultdict(float)
    encoder_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="microquery-M1")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            image = batch["image"].to(device)
            name = str(batch["name"][0])
            gt = (batch["mask"][0].detach().cpu().numpy() > 0)
            components = extract_components(gt)
            query_count = len(components)
            if query_count <= 0:
                continue
            if query_count > args.max_queries:
                raise RuntimeError(
                    f"{name} has {query_count} components, exceeding --max_queries"
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            encoded = model.get_image_embeddings(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            encoder_seconds += time.perf_counter() - start

            centers = torch.tensor(
                [[component.centroid_xy for component in components]],
                device=device,
                dtype=image.dtype,
            )
            points = centers.unsqueeze(2)
            point_labels = torch.ones((1, query_count, 1), device=device, dtype=torch.int64)
            pn_coords = []
            for component in components:
                pn_coords.append(
                    [component.centroid_xy, *clean_background_points(component.mask, gt)]
                )
            pn_points = torch.tensor([pn_coords], device=device, dtype=image.dtype)
            pn_labels = torch.tensor(
                [[[1, 0, 0, 0, 0] for _ in components]], device=device, dtype=torch.int64
            )
            boxes_np = []
            component_masks = []
            for component in components:
                ys, xs = np.nonzero(component.mask)
                boxes_np.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
                component_masks.append(component.mask.astype(np.float32))
            boxes = torch.tensor([boxes_np], device=device, dtype=image.dtype)
            masks = torch.from_numpy(np.stack(component_masks)[None, :, None]).to(
                device=device, dtype=image.dtype
            )
            masks = F.interpolate(
                masks.reshape(query_count, 1, args.size, args.size),
                size=model.prompt_encoder.mask_input_size,
                mode="nearest",
            ).reshape(1, query_count, 1, *model.prompt_encoder.mask_input_size)
            null_points = torch.empty((1, query_count, 0, 2), device=device, dtype=image.dtype)
            null_labels = torch.empty((1, query_count, 0), device=device, dtype=torch.int64)
            one_query_points = centers.unsqueeze(1)
            one_query_labels = torch.ones(
                (1, 1, query_count), device=device, dtype=torch.int64
            )
            prompt_inputs = {
                "M1-Null": {"points": null_points, "point_labels": null_labels},
                "M1-P-OneQuery": {
                    "points": one_query_points,
                    "point_labels": one_query_labels,
                },
                "M1-P": {"points": points, "point_labels": point_labels},
                "M1-PN": {"points": pn_points, "point_labels": pn_labels},
                "M1-B": {"boxes": boxes},
                "M1-M": {"masks": masks},
                "M1-PM": {
                    "points": points,
                    "point_labels": point_labels,
                    "masks": masks,
                },
                "M1-BM": {"boxes": boxes, "masks": masks},
            }
            for condition, prompt_kwargs in prompt_inputs.items():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                decoded = decode_prompt_queries(
                    model,
                    encoded[0],
                    encoded[1],
                    input_h=args.size,
                    input_w=args.size,
                    output_h=args.size,
                    output_w=args.size,
                    hq_token_only=False,
                    chunk_size=args.query_chunk,
                    **prompt_kwargs,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                condition_seconds[condition] += time.perf_counter() - start
                probabilities = torch.sigmoid(decoded.mask_logits)
                if condition == "M1-P-OneQuery":
                    final_probability = probabilities[:, 0]
                    per_query = probabilities[:, :1].expand(-1, query_count, -1, -1)
                else:
                    valid_tensor = torch.ones(
                        (1, query_count), device=device, dtype=torch.bool
                    )
                    final_probability = aggregate_query_probabilities(
                        probabilities, valid_tensor
                    )
                    per_query = probabilities
                padded_coords = np.zeros((args.max_queries, 2), dtype=np.float32)
                padded_scores = np.zeros(args.max_queries, dtype=np.float32)
                padded_valid = np.zeros(args.max_queries, dtype=bool)
                padded_query = np.zeros(
                    (args.max_queries, args.size, args.size), dtype=np.float32
                )
                padded_coords[:query_count] = centers[0].detach().cpu().numpy()
                padded_scores[:query_count] = 1.0
                padded_valid[:query_count] = True
                padded_query[:query_count] = per_query[0].detach().cpu().numpy()
                final_np = final_probability[0].detach().cpu().numpy()
                for threshold in thresholds:
                    accumulators[(condition, threshold)].update(
                        name=name,
                        gt_mask=gt,
                        candidate_xy=padded_coords,
                        candidate_scores=padded_scores,
                        candidate_valid=padded_valid,
                        query_probabilities=padded_query,
                        final_probability=final_np,
                        accepted=padded_valid,
                        object_scores=padded_scores,
                        threshold=threshold,
                    )
                final_maps[condition].append(final_np.astype(np.float16))
                if args.save_probability_maps:
                    query_maps[condition].append(padded_query.astype(np.float16))
            names.append(name)
            gt_maps.append(gt.astype(np.uint8))

    labels = np.stack(gt_maps).reshape(-1)
    fixed_threshold = 0.5
    fixed_results = {}
    threshold_rows = []
    for condition in CONDITIONS:
        mask_scores = np.stack(final_maps[condition]).astype(np.float32).reshape(-1)
        mask_auprc = float(average_precision_score(labels, mask_scores))
        for threshold in thresholds:
            result = accumulators[(condition, threshold)].finalize()
            result["summary"]["mask_auprc"] = mask_auprc
            threshold_rows.append(
                {"condition": condition, "threshold": threshold, **result["summary"]}
            )
            if math.isclose(threshold, fixed_threshold):
                fixed_results[condition] = result
    summary_rows = [
        {"condition": condition, **fixed_results[condition]["summary"]}
        for condition in CONDITIONS
    ]
    write_csv(output_dir / "threshold_curve.csv", threshold_rows)
    write_csv(output_dir / "fixed_0_5_summary.csv", summary_rows)
    details = output_dir / "fixed_0_5_details"
    details.mkdir(exist_ok=True)
    for condition, result in fixed_results.items():
        write_csv(details / f"{condition}_per_image.csv", result["per_image_rows"])
        write_csv(details / f"{condition}_per_component.csv", result["per_component_rows"])
        write_csv(details / f"{condition}_per_query.csv", result["per_query_rows"])

    area_rows = []
    for condition, result in fixed_results.items():
        for area_name in ("1-9", "10-16", "17-25", ">25"):
            rows = [
                row
                for row in result["per_component_rows"]
                if row["area_bin"] == area_name
            ]
            area_rows.append(
                {
                    "condition": condition,
                    "area_bin": area_name,
                    "components": len(rows),
                    "component_detection": sum(int(row["final_detected"]) for row in rows)
                    / max(1, len(rows)),
                    "best_query_mask_iou": float(
                        np.mean([float(row["best_query_iou"]) for row in rows])
                    )
                    if rows
                    else float("nan"),
                    "qmsr_at_0_3": sum(float(row["best_query_iou"]) >= 0.3 for row in rows)
                    / max(1, len(rows)),
                }
            )
    write_csv(output_dir / "area_bin_oracle_metrics.csv", area_rows)
    point_tiny = next(
        row for row in area_rows if row["condition"] == "M1-P" and row["area_bin"] == "1-9"
    )
    micro_candidates = [
        row
        for row in area_rows
        if row["condition"] in {"M1-M", "M1-PM"} and row["area_bin"] == "1-9"
    ]
    best_micro = max(micro_candidates, key=lambda row: float(row["best_query_mask_iou"]))
    point_summary = fixed_results["M1-P"]["summary"]
    best_micro_summary = fixed_results[best_micro["condition"]]["summary"]
    micro_mask_pass = bool(
        float(best_micro["best_query_mask_iou"])
        >= float(point_tiny["best_query_mask_iou"]) + 0.03
        or float(best_micro["component_detection"])
        >= float(point_tiny["component_detection"]) + 0.03
        or (
            float(best_micro_summary["mean_niou"])
            >= float(point_summary["mean_niou"]) + 0.005
            and float(best_micro_summary["fa"]) <= 1.05 * float(point_summary["fa"])
        )
    )
    decision = {
        "oracle_only": True,
        "micro_mask_gate_passed": micro_mask_pass,
        "best_micro_condition": best_micro["condition"],
        "tiny_best_query_iou_delta_vs_point": float(best_micro["best_query_mask_iou"])
        - float(point_tiny["best_query_mask_iou"]),
        "tiny_component_detection_delta_vs_point": float(best_micro["component_detection"])
        - float(point_tiny["component_detection"]),
        "mean_niou_delta_vs_point": float(best_micro_summary["mean_niou"])
        - float(point_summary["mean_niou"]),
        "instruction": (
            "M2 may include a micro-mask ablation"
            if micro_mask_pass
            else "Do not implement the M2 micro-mask head"
        ),
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.save_probability_maps:
        map_dir = output_dir / "probability_maps"
        map_dir.mkdir(exist_ok=True)
        for condition in CONDITIONS:
            np.savez_compressed(
                map_dir / f"{condition}.npz",
                image_names=np.asarray(names),
                final_probability=np.stack(final_maps[condition]),
                query_probability=np.stack(query_maps[condition]),
            )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    manifest = {
        "schema_version": 1,
        "experiment": "MicroQuery M1 oracle prompt representation ladder",
        "deployment_status": "NOT DEPLOYABLE",
        "dataset": args.dataset,
        "images": len(names),
        "split_sha256": sha256_file(split_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "weights_sha256": sha256_file(weights_path),
        "git_commit": current_git_commit(),
        "encoder_ms_per_image": encoder_seconds * 1000.0 / max(1, len(names)),
        "decoder_ms_per_image": {
            condition: seconds * 1000.0 / max(1, len(names))
            for condition, seconds in condition_seconds.items()
        },
        "peak_gpu_memory_bytes": peak_memory,
        "gt_boundary": "GT generates all M1 prompts; every result is oracle-only and NOT DEPLOYABLE",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"manifest": manifest, "fixed_0_5": summary_rows, "decision": decision},
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "fixed_0_5": summary_rows, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

