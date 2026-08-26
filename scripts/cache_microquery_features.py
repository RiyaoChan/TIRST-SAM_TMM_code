#!/usr/bin/env python3
"""Cache deployable frozen ROI descriptors and isolated GT analysis targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.microquery import (
    decode_prompt_queries,
    extract_candidate_roi_features,
)
from efficient_sam.microquery_metrics import (
    ASSIGNMENT_DUPLICATE,
    ASSIGNMENT_PRIMARY,
    assign_candidates,
)
from efficient_sam.prompt_metrics import extract_components
from scripts.eval_prompt_quality import sha256_file
from scripts.train_experiment1_single_view import set_deterministic
from sirst_dataset import make_loader


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="IRSTD-1k")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--query_chunk", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--save_query_probabilities", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    checkpoint_path = Path(args.checkpoint).resolve()
    candidate_cache_path = Path(args.candidate_cache).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_cache = np.load(candidate_cache_path, allow_pickle=False)
    cached_names = [str(name) for name in candidate_cache["image_names"]]
    if len(set(cached_names)) != len(cached_names):
        raise RuntimeError("Candidate cache contains duplicate image names")
    cache_lookup = {name: index for index, name in enumerate(cached_names)}
    if args.budget > int(candidate_cache["candidate_xy"].shape[1]):
        raise ValueError("budget exceeds cached candidate count")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(weights_path),
        use_adapter=False,
        return_encoder_multi_scale=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

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

    names_out: list[str] = []
    descriptors_out: list[np.ndarray] = []
    xy_out: list[np.ndarray] = []
    scores_out: list[np.ndarray] = []
    valid_out: list[np.ndarray] = []
    decoder_quality_out: list[np.ndarray] = []
    query_probs_out: list[np.ndarray] = []
    semantic_out: list[np.ndarray] = []
    primary_out: list[np.ndarray] = []
    duplicate_out: list[np.ndarray] = []
    component_out: list[np.ndarray] = []
    query_iou_out: list[np.ndarray] = []
    gt_masks_out: list[np.ndarray] = []
    encoder_seconds = 0.0
    decoder_seconds = 0.0
    image_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="microquery-feature-cache")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            names = [str(name) for name in batch["name"]]
            if any(name not in cache_lookup for name in names):
                raise RuntimeError("Candidate cache and split image names do not match")
            cache_indices = [cache_lookup[name] for name in names]
            xy_np = candidate_cache["candidate_xy"][cache_indices, : args.budget].astype(
                np.float32
            )
            scores_np = candidate_cache["candidate_scores"][
                cache_indices, : args.budget
            ].astype(np.float32)
            valid_np = candidate_cache["candidate_valid"][
                cache_indices, : args.budget
            ].astype(bool)
            xy = torch.from_numpy(xy_np).to(device=device, dtype=images.dtype)
            scores = torch.from_numpy(scores_np).to(device=device, dtype=images.dtype)
            valid = torch.from_numpy(valid_np).to(device=device, dtype=torch.bool)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            encoded = model.get_image_embeddings(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            encoder_seconds += time.perf_counter() - start
            if len(encoded) < 3 or not encoded[2]:
                raise RuntimeError("Encoder did not return multi-scale features")
            shallow = encoded[2][0]
            descriptors = extract_candidate_roi_features(
                shallow,
                encoded[0],
                xy,
                scores,
                valid,
                input_h=args.size,
                input_w=args.size,
            )
            point_labels = torch.where(
                valid,
                torch.ones_like(valid, dtype=torch.int64),
                -torch.ones_like(valid, dtype=torch.int64),
            )
            points = torch.where(valid.unsqueeze(-1), xy, torch.zeros_like(xy)).unsqueeze(2)
            point_labels = point_labels.unsqueeze(2)
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
                points=points,
                point_labels=point_labels,
                chunk_size=args.query_chunk,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            decoder_seconds += time.perf_counter() - start
            query_probs = torch.sigmoid(decoded.mask_logits).cpu().numpy().astype(np.float32)
            gt_masks = (batch["mask"].detach().cpu().numpy() > 0)

            for image_index, name in enumerate(names):
                components = extract_components(gt_masks[image_index])
                assignment = assign_candidates(
                    xy_np[image_index], valid_np[image_index], components, budget=args.budget
                )
                semantic = assignment.semantic_target.astype(bool)
                primary = np.asarray(
                    [value == ASSIGNMENT_PRIMARY for value in assignment.assignment], dtype=bool
                )
                duplicate = np.asarray(
                    [value == ASSIGNMENT_DUPLICATE for value in assignment.assignment], dtype=bool
                )
                target_iou = np.zeros(args.budget, dtype=np.float32)
                for query_index in range(args.budget):
                    component_index = int(assignment.component_index[query_index])
                    if component_index < 0:
                        continue
                    prediction = query_probs[image_index, query_index] >= 0.5
                    component_mask = components[component_index].mask
                    intersection = int(np.logical_and(prediction, component_mask).sum())
                    union = int(np.logical_or(prediction, component_mask).sum())
                    target_iou[query_index] = intersection / union if union else 1.0
                semantic_out.append(semantic)
                primary_out.append(primary)
                duplicate_out.append(duplicate)
                component_out.append(assignment.component_index.astype(np.int16))
                query_iou_out.append(target_iou)

            names_out.extend(names)
            descriptors_out.append(descriptors.cpu().numpy().astype(np.float16))
            xy_out.append(xy_np)
            scores_out.append(scores_np)
            valid_out.append(valid_np)
            decoder_quality_out.append(decoded.quality.cpu().numpy().astype(np.float16))
            if args.save_query_probabilities:
                query_probs_out.append(query_probs.astype(np.float16))
            gt_masks_out.append(gt_masks.astype(np.uint8))
            image_count += len(names)

    if not names_out:
        raise RuntimeError("No feature rows were cached")
    feature_path = output_dir / "features.npz"
    feature_payload = {
        "image_names": np.asarray(names_out),
        "roi_descriptors": np.concatenate(descriptors_out, axis=0),
        "candidate_xy": np.concatenate(xy_out, axis=0),
        "candidate_scores": np.concatenate(scores_out, axis=0),
        "candidate_valid": np.concatenate(valid_out, axis=0),
        "sam_decoder_quality": np.concatenate(decoder_quality_out, axis=0),
    }
    if args.save_query_probabilities:
        feature_payload["query_probabilities"] = np.concatenate(query_probs_out, axis=0)
    np.savez_compressed(feature_path, **feature_payload)

    analysis_path = output_dir / "analysis_targets.npz"
    np.savez_compressed(
        analysis_path,
        image_names=np.asarray(names_out),
        semantic_target=np.stack(semantic_out),
        primary_target=np.stack(primary_out),
        duplicate_target=np.stack(duplicate_out),
        component_index=np.stack(component_out),
        query_mask_iou=np.stack(query_iou_out),
        gt_masks=np.concatenate(gt_masks_out, axis=0),
    )
    manifest = {
        "schema_version": 1,
        "experiment": "MicroQuery M2-S0 frozen ROI feature cache",
        "dataset": args.dataset,
        "images": image_count,
        "budget": args.budget,
        "descriptor_dim": int(feature_payload["roi_descriptors"].shape[-1]),
        "contains_query_probabilities": bool(args.save_query_probabilities),
        "candidate_cache_sha256": sha256_file(candidate_cache_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "weights_sha256": sha256_file(weights_path),
        "split_sha256": sha256_file(split_path),
        "feature_cache_sha256": sha256_file(feature_path),
        "analysis_targets_sha256": sha256_file(analysis_path),
        "git_commit": current_git_commit(),
        "encoder_ms_per_image": encoder_seconds * 1000.0 / max(1, image_count),
        "decoder_ms_per_image": decoder_seconds * 1000.0 / max(1, image_count),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "gt_boundary": (
            "features.npz is deployable and contains no GT. All GT-derived labels, "
            "query IoU targets, and masks are physically isolated in analysis_targets.npz."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
