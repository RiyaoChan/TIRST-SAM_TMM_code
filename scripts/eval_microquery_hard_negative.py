#!/usr/bin/env python3
"""Evaluate component-safe rejection on a prepared background feature cache."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery_component_safe import (
    ComponentSafeMicroQueryHead,
    GroupingConfig,
    build_candidate_graph,
    connected_candidate_groups,
    pairwise_candidate_relations,
)
from scripts.eval_microquery_component_safe_cache import predict_objectness, write_csv
from scripts.train_microquery_component_safe import predict, select_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--old_objectness_checkpoint", required=True)
    parser.add_argument("--component_safe_checkpoint", required=True)
    parser.add_argument("--semantic_threshold", type=float, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def summarize(
    names: list[str], query: np.ndarray, valid: np.ndarray, accepted: np.ndarray,
    scores: np.ndarray, elapsed: float, threshold: float,
) -> tuple[dict, list[dict]]:
    rows = []
    for index, name in enumerate(names):
        final = np.where(accepted[index, :, None, None], query[index], 0.0).max(axis=0)
        rows.append(
            {
                "image": name,
                "zero_query": int(not accepted[index].any()),
                "accepted_candidates": int(accepted[index].sum()),
                "false_mask_pixels": int((final >= threshold).sum()),
                "max_mask_probability": float(final.max()),
                "max_group_confidence": float(scores[index][valid[index]].max()) if valid[index].any() else 0.0,
            }
        )
    return {
        "images": len(rows),
        "zero_query_fraction": float(np.mean([row["zero_query"] for row in rows])),
        "accepted_candidates_per_image": float(np.mean([row["accepted_candidates"] for row in rows])),
        "false_mask_pixels_per_image": float(np.mean([row["false_mask_pixels"] for row in rows])),
        "max_mask_probability_mean": float(np.mean([row["max_mask_probability"] for row in rows])),
        "latency_ms_per_image": elapsed * 1000.0 / max(1, len(rows)),
    }, rows


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features).resolve()
    values = np.load(feature_path, allow_pickle=False)
    names = [str(value) for value in values["image_names"]]
    descriptors = values["roi_descriptors"].astype(np.float32)
    xy = values["candidate_xy"].astype(np.float32)
    valid = values["candidate_valid"].astype(bool)
    query = values["query_probabilities"].astype(np.float32)
    groups = []
    config = GroupingConfig("hybrid", r_near=2, r_far=8, tau_iou=0.2, r_mask=5)
    for index in range(len(names)):
        relations = pairwise_candidate_relations(xy[index], query[index], descriptors[index])
        groups.append(connected_candidate_groups(build_candidate_graph(relations, valid[index], config), valid[index]))
    checkpoint = torch.load(args.component_safe_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ComponentSafeMicroQueryHead(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    start = time.perf_counter()
    semantic, utility = predict(model, descriptors, valid, device)
    accepted = select_candidates(semantic, utility, valid, groups, args.semantic_threshold, group_safe=True, use_utility=checkpoint["stage"] in {"b3", "b4"})
    candidate_elapsed = time.perf_counter() - start
    correct, correct_rows = summarize(names, query, valid, accepted, semantic, candidate_elapsed, args.mask_threshold)
    old_start = time.perf_counter()
    old, _ = predict_objectness(descriptors, valid, Path(args.old_objectness_checkpoint))
    old_accepted = valid & (old >= 0.15)
    old_elapsed = time.perf_counter() - old_start
    baseline, baseline_rows = summarize(names, query, valid, old_accepted, old, old_elapsed, args.mask_threshold)
    false_ratio = correct["false_mask_pixels_per_image"] / max(1e-12, baseline["false_mask_pixels_per_image"])
    accepted_ratio = correct["accepted_candidates_per_image"] / max(1e-12, baseline["accepted_candidates_per_image"])
    gate = bool(false_ratio <= 1.10 and accepted_ratio <= 1.10)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "correct_per_image.csv", correct_rows)
    write_csv(output_dir / "baseline_per_image.csv", baseline_rows)
    summary = {"correct": correct, "old_hard_0_15": baseline, "false_mask_ratio": false_ratio, "accepted_candidate_ratio": accepted_ratio, "hard_negative_gate_passed": gate}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
