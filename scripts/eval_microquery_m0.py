#!/usr/bin/env python3
"""Zero-training one-query versus candidate-isolated MicroQuery diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.microquery import (
    aggregate_query_probabilities,
    decode_prompt_queries,
    proposal_to_point_queries,
)
from efficient_sam.microquery_metrics import MicroQueryMetricAccumulator, assign_candidates
from efficient_sam.prompt_metrics import extract_components
from efficient_sam.prompt_proposal import PromptProposal
from scripts.eval_prompt_quality import sha256_file
from scripts.train_experiment1_single_view import set_deterministic
from sirst_dataset import make_loader


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not result or min(result) <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return result


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


def _primary_filter(
    coords: np.ndarray, valid: np.ndarray, gt_mask: np.ndarray, budget: int
) -> np.ndarray:
    assignment = assign_candidates(
        coords, valid, extract_components(gt_mask), budget=budget
    )
    return np.asarray(
        [label_name == "primary" for label_name in assignment.assignment], dtype=bool
    )


def _bootstrap_delta(
    base_rows: list[dict], other_rows: list[dict], field: str, seed: int, samples: int
) -> dict:
    base_by_image = {row["image"]: row for row in base_rows}
    other_by_image = {row["image"]: row for row in other_rows}
    names = sorted(set(base_by_image) & set(other_by_image))
    if not names:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    def value(row: dict) -> float:
        if field == "ctr":
            return (
                float(row["covered_detected"]) / float(row["covered_components"])
                if int(row["covered_components"]) > 0
                else float("nan")
            )
        if field == "fa":
            return float(row["final_false_pixels"]) / max(1.0, float(row["pixels"]))
        return float(row[field])

    base = np.asarray([value(base_by_image[name]) for name in names], dtype=np.float64)
    other = np.asarray([value(other_by_image[name]) for name in names], dtype=np.float64)
    valid = np.isfinite(base) & np.isfinite(other)
    delta = other[valid] - base[valid]
    if delta.size == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boot = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        selected = rng.integers(0, delta.size, size=delta.size)
        boot[index] = float(delta[selected].mean())
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe_checkpoint", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="IRSTD-1k")
    parser.add_argument("--budgets", type=parse_ints, default=(5, 10, 20))
    parser.add_argument(
        "--thresholds",
        type=parse_floats,
        default=tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05)),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--query_chunk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--save_probability_maps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    checkpoint_path = Path(args.checkpoint).resolve()
    probe_path = Path(args.probe_checkpoint).resolve()
    weights_path = Path(args.weights).resolve()
    cache_path = Path(args.candidate_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(cache_path, allow_pickle=False)
    cached_names = [str(name) for name in cache["image_names"]]
    cached_lookup = {name: index for index, name in enumerate(cached_names)}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved = dict(checkpoint["resolved_args"])
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(weights_path),
        use_adapter=False,
        return_encoder_multi_scale=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    loader = make_loader(
        str(data_root),
        str(split_path),
        batch_size=args.batch_size,
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
    condition_names = (
        "M0-One",
        "M0-Micro-max",
        "M0-Micro-candidate",
        "M0-Micro-sam_iou",
        "M0-Micro-candidate_sam",
        "M0-Micro-top1",
        "M0-Micro-top3",
        "M0-Micro-top5",
        "CF-order-shuffle",
        "CF-coordinate-batch-shuffle",
        "CF-labels-invalid",
        "Oracle-filter-primary",
    )
    accumulators = {
        (budget, condition, threshold): MicroQueryMetricAccumulator(budget)
        for budget in args.budgets
        for condition in condition_names
        for threshold in args.thresholds
    }
    probability_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    correct_query_maps: list[np.ndarray] = []
    gt_map_batches: list[np.ndarray] = []
    probability_names: list[str] = []
    drop_rows: list[dict] = []
    total_encoder_seconds = 0.0
    total_one_decoder_seconds = defaultdict(float)
    total_micro_decoder_seconds = 0.0
    image_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="microquery-M0")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            names = [str(name) for name in batch["name"]]
            if any(name not in cached_lookup for name in names):
                raise RuntimeError("Candidate cache and split image names do not match")
            gt_masks = (batch["mask"].detach().cpu().numpy() > 0)
            gt_map_batches.append(gt_masks.astype(np.uint8))
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            encoded = model.get_image_embeddings(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_encoder_seconds += time.perf_counter() - start
            indices = [cached_lookup[name] for name in names]
            proposal = PromptProposal(
                dense_logits=None,
                dense_probs=None,
                candidate_xy=torch.from_numpy(cache["candidate_xy"][indices]).to(
                    device=device, dtype=images.dtype
                ),
                candidate_scores=torch.from_numpy(cache["candidate_scores"][indices]).to(
                    device=device, dtype=images.dtype
                ),
                candidate_valid=torch.from_numpy(cache["candidate_valid"][indices]).to(
                    device=device, dtype=torch.bool
                ),
                candidate_source=[
                    ["A1_neck_probe_cache"] * int(cache["candidate_xy"].shape[1])
                    for _ in names
                ],
            ).validate()

            max_budget = max(args.budgets)
            micro_points, micro_labels = proposal_to_point_queries(
                proposal, max_budget, independent=True
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            micro = decode_prompt_queries(
                model,
                encoded[0],
                encoded[1],
                input_h=args.size,
                input_w=args.size,
                output_h=args.size,
                output_w=args.size,
                points=micro_points,
                point_labels=micro_labels,
                hq_token_only=False,
                chunk_size=args.query_chunk,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_micro_decoder_seconds += time.perf_counter() - start
            micro_probs = torch.sigmoid(micro.mask_logits)
            micro_quality = micro.quality.clamp(0.0, 1.0)

            invalid_labels = -torch.ones_like(micro_labels)
            zero = decode_prompt_queries(
                model,
                encoded[0],
                encoded[1],
                input_h=args.size,
                input_w=args.size,
                output_h=args.size,
                output_w=args.size,
                points=micro_points,
                point_labels=invalid_labels,
                hq_token_only=False,
                chunk_size=args.query_chunk,
            )
            zero_probs = torch.sigmoid(zero.mask_logits)
            rolled_points = torch.roll(micro_points, shifts=1, dims=0)
            rolled_labels = torch.roll(micro_labels, shifts=1, dims=0)
            wrong = decode_prompt_queries(
                model,
                encoded[0],
                encoded[1],
                input_h=args.size,
                input_w=args.size,
                output_h=args.size,
                output_w=args.size,
                points=rolled_points,
                point_labels=rolled_labels,
                hq_token_only=False,
                chunk_size=args.query_chunk,
            )
            wrong_probs = torch.sigmoid(wrong.mask_logits)

            for budget in args.budgets:
                one_points, one_labels = proposal_to_point_queries(
                    proposal, budget, independent=False
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                one = decode_prompt_queries(
                    model,
                    encoded[0],
                    encoded[1],
                    input_h=args.size,
                    input_w=args.size,
                    output_h=args.size,
                    output_w=args.size,
                    points=one_points,
                    point_labels=one_labels,
                    hq_token_only=False,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_one_decoder_seconds[budget] += time.perf_counter() - start
                one_prob = torch.sigmoid(one.mask_logits[:, 0])
                valid = proposal.candidate_valid[:, :budget]
                scores = proposal.candidate_scores[:, :budget].clamp(0.0, 1.0)
                query_probs = micro_probs[:, :budget]
                quality = micro_quality[:, :budget]
                candidate_quality = (scores * quality).clamp(0.0, 1.0)
                conditions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
                conditions["M0-One"] = (
                    one_prob,
                    one_prob[:, None].expand(-1, budget, -1, -1),
                    valid,
                    scores,
                )
                conditions["M0-Micro-max"] = (
                    aggregate_query_probabilities(query_probs, valid),
                    query_probs,
                    valid,
                    scores,
                )
                conditions["M0-Micro-candidate"] = (
                    aggregate_query_probabilities(query_probs, valid, weights=scores),
                    query_probs,
                    valid,
                    scores,
                )
                conditions["M0-Micro-sam_iou"] = (
                    aggregate_query_probabilities(query_probs, valid, weights=quality),
                    query_probs,
                    valid,
                    quality,
                )
                conditions["M0-Micro-candidate_sam"] = (
                    aggregate_query_probabilities(
                        query_probs, valid, weights=candidate_quality
                    ),
                    query_probs,
                    valid,
                    candidate_quality,
                )
                for top_n in (1, 3, 5):
                    conditions[f"M0-Micro-top{top_n}"] = (
                        aggregate_query_probabilities(
                            query_probs, valid, weights=scores, top_n=top_n
                        ),
                        query_probs,
                        valid & (
                            scores
                            >= torch.topk(
                                torch.where(valid, scores, scores.new_full(scores.shape, -1.0)),
                                k=min(top_n, budget),
                                dim=1,
                            ).values[:, -1:]
                        ),
                        scores,
                    )
                order = torch.arange(budget - 1, -1, -1, device=device)
                conditions["CF-order-shuffle"] = (
                    aggregate_query_probabilities(
                        query_probs[:, order], valid[:, order], weights=scores[:, order]
                    ),
                    query_probs[:, order],
                    valid,
                    scores,
                )
                conditions["CF-coordinate-batch-shuffle"] = (
                    aggregate_query_probabilities(wrong_probs[:, :budget], valid),
                    wrong_probs[:, :budget],
                    valid,
                    scores,
                )
                conditions["CF-labels-invalid"] = (
                    aggregate_query_probabilities(zero_probs[:, :budget], valid),
                    zero_probs[:, :budget],
                    valid,
                    scores,
                )
                oracle_valid = []
                for image_index in range(len(names)):
                    oracle_valid.append(
                        _primary_filter(
                            proposal.candidate_xy[image_index].detach().cpu().numpy(),
                            proposal.candidate_valid[image_index].detach().cpu().numpy(),
                            gt_masks[image_index],
                            budget,
                        )
                    )
                oracle_valid_tensor = torch.from_numpy(np.stack(oracle_valid)).to(device)
                conditions["Oracle-filter-primary"] = (
                    aggregate_query_probabilities(query_probs, oracle_valid_tensor),
                    query_probs,
                    oracle_valid_tensor,
                    oracle_valid_tensor.to(scores.dtype),
                )

                correct_max = conditions["M0-Micro-max"][0]
                for image_index, name in enumerate(names):
                    for query_index in range(budget):
                        if not bool(valid[image_index, query_index]):
                            continue
                        drop_valid = valid[image_index].clone()
                        drop_valid[query_index] = False
                        dropped = aggregate_query_probabilities(
                            query_probs[image_index : image_index + 1],
                            drop_valid.unsqueeze(0),
                        )[0]
                        drop_rows.append(
                            {
                                "image": name,
                                "budget": budget,
                                "candidate_rank": query_index + 1,
                                "mean_abs_probability_delta": float(
                                    (correct_max[image_index] - dropped).abs().mean()
                                ),
                                "changed_pixels_at_0_5": int(
                                    (
                                        (correct_max[image_index] >= 0.5)
                                        != (dropped >= 0.5)
                                    ).sum()
                                ),
                            }
                        )

                for condition, (final_prob, per_query_prob, accepted, object_scores) in conditions.items():
                    probability_maps[f"K{budget}_{condition}"].append(
                        final_prob.detach().cpu().numpy().astype(np.float16)
                    )
                    for image_index, name in enumerate(names):
                        coords_np = proposal.candidate_xy[image_index, :budget].detach().cpu().numpy()
                        scores_np = proposal.candidate_scores[image_index, :budget].detach().cpu().numpy()
                        valid_np = proposal.candidate_valid[image_index, :budget].detach().cpu().numpy()
                        query_np = per_query_prob[image_index].detach().cpu().numpy()
                        final_np = final_prob[image_index].detach().cpu().numpy()
                        accepted_np = accepted[image_index].detach().cpu().numpy()
                        object_np = object_scores[image_index].detach().cpu().numpy()
                        for threshold in args.thresholds:
                            accumulators[(budget, condition, threshold)].update(
                                name=name,
                                gt_mask=gt_masks[image_index],
                                candidate_xy=coords_np,
                                candidate_scores=scores_np,
                                candidate_valid=valid_np,
                                query_probabilities=query_np,
                                final_probability=final_np,
                                accepted=accepted_np,
                                object_scores=object_np,
                                threshold=threshold,
                            )
                if args.save_probability_maps and budget == max_budget:
                    correct_query_maps.append(
                        micro_probs.detach().cpu().numpy().astype(np.float16)
                    )
            probability_names.extend(names)
            image_count += len(names)

    threshold_rows = []
    fixed_results: dict[str, dict] = {}
    fixed_threshold = min(args.thresholds, key=lambda value: abs(float(value) - 0.5))
    for budget in args.budgets:
        for condition in condition_names:
            for threshold in args.thresholds:
                result = accumulators[(budget, condition, threshold)].finalize()
                row = {
                    "budget": budget,
                    "condition": condition,
                    "threshold": threshold,
                    **result["summary"],
                }
                threshold_rows.append(row)
                if math.isclose(threshold, fixed_threshold):
                    fixed_results[f"K{budget}_{condition}"] = result

    mask_labels = np.concatenate(gt_map_batches, axis=0).reshape(-1)
    for key, batches in probability_maps.items():
        mask_scores = np.concatenate(batches, axis=0).astype(np.float32).reshape(-1)
        fixed_results[key]["summary"]["mask_auprc"] = float(
            average_precision_score(mask_labels, mask_scores)
        )

    summary_rows = []
    for key, result in fixed_results.items():
        budget_text, condition = key.split("_", 1)
        summary_rows.append(
            {"budget": int(budget_text[1:]), "condition": condition, **result["summary"]}
        )
    write_csv(output_dir / "threshold_curve.csv", threshold_rows)
    write_csv(output_dir / "fixed_0_5_summary.csv", summary_rows)
    write_csv(output_dir / "candidate_drop.csv", drop_rows)
    detail_dir = output_dir / "fixed_0_5_details"
    detail_dir.mkdir(exist_ok=True)
    for key, result in fixed_results.items():
        write_csv(detail_dir / f"{key}_per_image.csv", result["per_image_rows"])
        write_csv(detail_dir / f"{key}_per_component.csv", result["per_component_rows"])
        write_csv(detail_dir / f"{key}_per_query.csv", result["per_query_rows"])

    bootstrap_rows = []
    for budget in args.budgets:
        base = fixed_results[f"K{budget}_M0-One"]["per_image_rows"]
        for condition in condition_names:
            if condition == "M0-One":
                continue
            other = fixed_results[f"K{budget}_{condition}"]["per_image_rows"]
            for field in ("ctr", "iou", "fa"):
                interval = _bootstrap_delta(
                    base,
                    other,
                    field,
                    seed=args.seed + budget,
                    samples=args.bootstrap_samples,
                )
                bootstrap_rows.append(
                    {
                        "budget": budget,
                        "condition": condition,
                        "metric": field,
                        **interval,
                        "bootstrap_samples": args.bootstrap_samples,
                    }
                )
    write_csv(output_dir / "paired_bootstrap_vs_one.csv", bootstrap_rows)
    if args.save_probability_maps:
        map_dir = output_dir / "probability_maps"
        map_dir.mkdir(exist_ok=True)
        for key, batches in probability_maps.items():
            np.savez_compressed(
                map_dir / f"{key}.npz",
                image_names=np.asarray(probability_names),
                probability=np.concatenate(batches, axis=0),
            )
        np.savez_compressed(
            map_dir / f"independent_query_probs_K{max(args.budgets)}.npz",
            image_names=np.asarray(probability_names),
            probability=np.concatenate(correct_query_maps, axis=0),
        )

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    manifest = {
        "schema_version": 1,
        "experiment": "MicroQuery M0 zero-training candidate isolation",
        "dataset": args.dataset,
        "images": image_count,
        "budgets": list(args.budgets),
        "thresholds": list(args.thresholds),
        "fixed_threshold": fixed_threshold,
        "candidate_source": "A1 single-view neck SpatialProbeHead",
        "candidate_cache_sha256": sha256_file(cache_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "probe_checkpoint_sha256": sha256_file(probe_path),
        "split_sha256": sha256_file(split_path),
        "git_commit": current_git_commit(),
        "encoder_ms_per_image": total_encoder_seconds * 1000.0 / max(1, image_count),
        "micro_decoder_ms_per_image": total_micro_decoder_seconds * 1000.0 / max(1, image_count),
        "one_decoder_ms_per_image": {
            str(budget): total_one_decoder_seconds[budget] * 1000.0 / max(1, image_count)
            for budget in args.budgets
        },
        "peak_gpu_memory_bytes": peak_memory,
        "gt_boundary": (
            "GT is used only after frozen candidates and decoder masks are produced; "
            "Oracle-filter-primary is NOT DEPLOYABLE"
        ),
        "reliability_counterfactual": "not applicable to the default A1 candidate source",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            {"manifest": manifest, "fixed_0_5": summary_rows},
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
    print(json.dumps({"manifest": manifest, "fixed_0_5": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
