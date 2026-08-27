#!/usr/bin/env python3
"""Audit deployable candidate groups and evaluate cache-level S1b policies."""

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

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery import MicroQueryHead
from efficient_sam.microquery_component_safe import (
    GroupingConfig,
    build_candidate_graph,
    connected_candidate_groups,
    global_top_l_rescue,
    pairwise_candidate_relations,
    select_group_champions,
    summarize_candidate_groups,
    tri_state_group_rejection,
)
from efficient_sam.microquery_metrics import MicroQueryMetricAccumulator
from scripts.eval_prompt_quality import sha256_file
from scripts.train_experiment1_single_view import set_deterministic


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--objectness_checkpoint", required=True)
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--val_split", required=True)
    parser.add_argument("--a1_checkpoint", required=True)
    parser.add_argument("--probe_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--dataset", default="IRSTD-1k")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    return parser.parse_args()


def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def load_data(feature_path: Path, target_path: Path) -> dict:
    features = np.load(feature_path, allow_pickle=False)
    targets = np.load(target_path, allow_pickle=False)
    names = [str(value) for value in features["image_names"]]
    if names != [str(value) for value in targets["image_names"]]:
        raise RuntimeError("Feature and analysis caches have different image order")
    if "query_probabilities" not in features.files:
        raise RuntimeError("Feature cache must contain frozen query probabilities")
    return {
        "names": names,
        "descriptors": features["roi_descriptors"].astype(np.float32),
        "xy": features["candidate_xy"].astype(np.float32),
        "raw": features["candidate_scores"].astype(np.float32),
        "valid": features["candidate_valid"].astype(bool),
        "sam_quality_raw": features["sam_decoder_quality"].astype(np.float32),
        "sam_quality": sigmoid_numpy(features["sam_decoder_quality"]),
        "query": features["query_probabilities"].astype(np.float32),
        "semantic": targets["semantic_target"].astype(bool),
        "primary": targets["primary_target"].astype(bool),
        "duplicate": targets["duplicate_target"].astype(bool),
        "component": targets["component_index"].astype(np.int64),
        "query_iou": targets["query_mask_iou"].astype(np.float32),
        "gt": targets["gt_masks"].astype(bool),
    }


def predict_objectness(
    descriptors: np.ndarray,
    valid: np.ndarray,
    checkpoint_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MicroQueryHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(
            torch.from_numpy(descriptors).to(device),
            torch.from_numpy(valid).to(device),
        )
        objectness = torch.softmax(output.object_logits, dim=-1)[..., 1].cpu().numpy()
        quality = torch.sigmoid(output.quality_logits).cpu().numpy()
    return objectness.astype(np.float32), quality.astype(np.float32)


def grouping_configs() -> list[GroupingConfig]:
    configs: list[GroupingConfig] = []
    configs.extend(GroupingConfig("coordinate", r_xy=value) for value in (2, 3, 4, 5, 6))
    configs.extend(GroupingConfig("mask", tau_iou=value) for value in (0.1, 0.2, 0.3, 0.4, 0.5))
    for r_near in (2, 3, 4):
        for r_far in (6, 8, 10):
            for tau_iou in (0.2, 0.3, 0.4):
                for r_mask in (3, 5, 8):
                    configs.append(
                        GroupingConfig(
                            "hybrid",
                            r_near=r_near,
                            r_far=r_far,
                            tau_iou=tau_iou,
                            r_mask=r_mask,
                        )
                    )
    return configs


def config_name(config: GroupingConfig) -> str:
    if config.mode == "coordinate":
        return f"G0_xy_r{config.r_xy:g}"
    if config.mode == "mask":
        return f"G1_mask_iou{config.tau_iou:g}"
    suffix = (
        f"rn{config.r_near:g}_rf{config.r_far:g}_"
        f"iou{config.tau_iou:g}_rm{config.r_mask:g}"
    )
    if config.mode == "hybrid_feature":
        suffix += f"_feat{config.tau_feat:g}"
    return f"{'G3' if config.mode == 'hybrid_feature' else 'G2'}_{suffix}"


def audit_groups(
    data: dict,
    groups_by_image: list[tuple[tuple[int, ...], ...]],
    objectness: np.ndarray,
) -> tuple[dict, list[dict]]:
    collision_groups = 0
    target_groups = 0
    background_groups = 0
    surviving_target_groups = 0
    rejected_background_groups = 0
    background_in_target = 0
    target_group_candidates = 0
    duplicate_total = 0
    duplicate_with_primary = 0
    target_pairs = 0
    target_pairs_grouped = 0
    total_groups = 0
    collision_rows: list[dict] = []
    for image_index, groups in enumerate(groups_by_image):
        accepted_hard = data["valid"][image_index] & (objectness[image_index] >= 0.15)
        group_index: dict[int, int] = {}
        for local_group_id, group in enumerate(groups):
            total_groups += 1
            for candidate in group:
                group_index[int(candidate)] = local_group_id
            target_components = sorted(
                {
                    int(data["component"][image_index, candidate])
                    for candidate in group
                    if data["semantic"][image_index, candidate]
                    and int(data["component"][image_index, candidate]) >= 0
                }
            )
            background_count = sum(
                not bool(data["semantic"][image_index, candidate]) for candidate in group
            )
            if target_components:
                target_groups += 1
                target_group_candidates += len(group)
                background_in_target += background_count
                surviving_target_groups += int(any(accepted_hard[candidate] for candidate in group))
            else:
                background_groups += 1
                rejected_background_groups += int(
                    not any(accepted_hard[candidate] for candidate in group)
                )
            if len(target_components) >= 2:
                collision_groups += 1
                collision_rows.append(
                    {
                        "image": data["names"][image_index],
                        "group_id": local_group_id,
                        "candidate_indices": " ".join(str(value) for value in group),
                        "target_components": " ".join(str(value) for value in target_components),
                    }
                )
        valid_indices = np.flatnonzero(data["valid"][image_index])
        for offset, first in enumerate(valid_indices):
            if not data["semantic"][image_index, first]:
                continue
            for second in valid_indices[offset + 1 :]:
                if (
                    data["semantic"][image_index, second]
                    and data["component"][image_index, first]
                    == data["component"][image_index, second]
                ):
                    target_pairs += 1
                    target_pairs_grouped += int(group_index[first] == group_index[second])
        for duplicate in np.flatnonzero(data["duplicate"][image_index] & data["valid"][image_index]):
            duplicate_total += 1
            component = data["component"][image_index, duplicate]
            primary = np.flatnonzero(
                data["primary"][image_index]
                & data["valid"][image_index]
                & (data["component"][image_index] == component)
            )
            duplicate_with_primary += int(
                len(primary) > 0 and group_index[duplicate] == group_index[int(primary[0])]
            )
    summary = {
        "images": len(data["names"]),
        "groups": total_groups,
        "groups_per_image": total_groups / max(1, len(data["names"])),
        "collision_groups": collision_groups,
        "group_collision_rate": collision_groups / max(1, total_groups),
        "target_groups": target_groups,
        "background_groups": background_groups,
        "target_group_survival_at_hard_0_15": surviving_target_groups / max(1, target_groups),
        "background_group_rejection_at_hard_0_15": rejected_background_groups / max(1, background_groups),
        "target_pair_grouping_rate": target_pairs_grouped / max(1, target_pairs),
        "duplicate_with_primary_grouping_rate": duplicate_with_primary / max(1, duplicate_total),
        "background_contamination_in_target_groups": background_in_target / max(1, target_group_candidates),
        "target_pairs": target_pairs,
        "duplicates": duplicate_total,
    }
    return summary, collision_rows


def candidate_metrics(data: dict, accepted: np.ndarray) -> dict:
    accepted = np.asarray(accepted, dtype=bool) & data["valid"]
    covered: set[tuple[int, int]] = set()
    retained: set[tuple[int, int]] = set()
    for image_index in range(len(data["names"])):
        for candidate in np.flatnonzero(data["valid"][image_index]):
            component = int(data["component"][image_index, candidate])
            if data["primary"][image_index, candidate] and component >= 0:
                covered.add((image_index, component))
            if accepted[image_index, candidate] and component >= 0:
                retained.add((image_index, component))
    background = data["valid"] & ~data["semantic"]
    duplicate = data["valid"] & data["duplicate"]
    return {
        "covered_components": len(covered),
        "retained_covered_components": len(covered & retained),
        "fully_lost_covered_components": len(covered - retained),
        "target_candidate_retention": len(covered & retained) / max(1, len(covered)),
        "background_candidates": int(background.sum()),
        "rejected_background_candidates": int((background & ~accepted).sum()),
        "false_candidate_rejection": float((background & ~accepted).sum() / max(1, background.sum())),
        "duplicate_candidates": int(duplicate.sum()),
        "rejected_duplicate_candidates": int((duplicate & ~accepted).sum()),
        "duplicate_suppression": float((duplicate & ~accepted).sum() / max(1, duplicate.sum())),
        "accepted_candidates": int(accepted.sum()),
    }


def aggregate(query: np.ndarray, accepted: np.ndarray, weights: np.ndarray) -> np.ndarray:
    effective = np.where(accepted, weights, 0.0).astype(np.float32)
    return (query * effective[..., None, None]).max(axis=1)


def evaluate_pattern(
    data: dict,
    object_scores: np.ndarray,
    accepted: np.ndarray,
    weights: np.ndarray,
    mask_threshold: float,
) -> dict:
    final = aggregate(data["query"], accepted, weights)
    accumulator = MicroQueryMetricAccumulator(data["valid"].shape[1])
    for image_index, name in enumerate(data["names"]):
        accumulator.update(
            name=name,
            gt_mask=data["gt"][image_index],
            candidate_xy=data["xy"][image_index],
            candidate_scores=data["raw"][image_index],
            candidate_valid=data["valid"][image_index],
            query_probabilities=data["query"][image_index],
            final_probability=final[image_index],
            accepted=accepted[image_index],
            object_scores=object_scores[image_index],
            threshold=mask_threshold,
        )
    result = accumulator.finalize()
    result["summary"].update(candidate_metrics(data, accepted))
    result["final_probability"] = final
    return result


def pattern_hash(accepted: np.ndarray, weights: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(accepted, dtype=np.uint8).tobytes())
    digest.update(np.asarray(weights, dtype=np.float32).tobytes())
    return digest.hexdigest()


def apply_policy(
    spec: dict,
    data: dict,
    objectness: np.ndarray,
    groups_by_image: list[tuple[tuple[int, ...], ...]],
    quality: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, ...]]]:
    quality_values = data["sam_quality"] if quality is None else quality
    formula = spec.get("champion_formula", "object")
    if formula == "object_raw":
        champion = objectness * data["raw"]
    elif formula == "object_quality":
        champion = objectness * quality_values
    elif formula == "object_raw_quality":
        champion = objectness * data["raw"] * quality_values
    else:
        champion = objectness
    accepted_rows = []
    weight_rows = []
    state_rows = []
    for image_index in range(len(data["names"])):
        if spec["kind"] == "hard":
            accepted = data["valid"][image_index] & (
                objectness[image_index] >= float(spec["threshold"])
            )
            weights = accepted.astype(np.float32)
            state = tuple(
                "INVALID" if not data["valid"][image_index, candidate] else (
                    "ACCEPT" if accepted[candidate] else "REJECT"
                )
                for candidate in range(data["valid"].shape[1])
            )
            selection = (accepted, weights, state)
        elif spec["kind"] == "top_l":
            output = global_top_l_rescue(
                objectness[image_index],
                data["valid"][image_index],
                threshold=float(spec["threshold"]),
                minimum_count=int(spec["minimum_count"]),
            )
            selection = (output.accepted, output.weights, output.state)
        elif spec["kind"] == "a3":
            output = select_group_champions(
                groups_by_image[image_index],
                objectness[image_index],
                data["valid"][image_index],
                tau_high=float(spec["tau_high"]),
                tau_group=float(spec["tau_group"]),
                champion_scores=champion[image_index],
            )
            selection = (output.accepted, output.weights, output.state)
        elif spec["kind"] == "a4":
            output = tri_state_group_rejection(
                groups_by_image[image_index],
                objectness[image_index],
                data["valid"][image_index],
                tau_high=float(spec["tau_high"]),
                tau_low=float(spec["tau_low"]),
                tau_rescue=float(spec["tau_rescue"]),
                uncertain_weight=float(spec["uncertain_weight"]),
                champion_scores=champion[image_index],
            )
            selection = (output.accepted, output.weights, output.state)
        else:
            raise ValueError(f"Unknown policy: {spec['kind']}")
        accepted_rows.append(selection[0])
        weight_rows.append(selection[1])
        state_rows.append(selection[2])
    return np.stack(accepted_rows), np.stack(weight_rows), state_rows


def bootstrap_delta(
    base_rows: list[dict],
    candidate_rows: list[dict],
    samples: int,
    seed: int,
) -> list[dict]:
    if len(base_rows) != len(candidate_rows):
        raise ValueError("paired rows must have equal length")
    rng = np.random.default_rng(seed)
    count = len(base_rows)

    def summarize(rows: list[dict], indices: np.ndarray) -> dict:
        selected = [rows[int(index)] for index in indices]
        sums = lambda field: sum(float(row[field]) for row in selected)
        covered = sums("covered_components")
        background = sums("background_queries")
        duplicates = sums("duplicate_queries")
        components = sums("components")
        return {
            "global_iou": sums("intersection_pixels") / max(1.0, sums("union_pixels")),
            "mean_niou": float(np.mean([float(row["iou"]) for row in selected])),
            "pd": sums("detected_components") / max(1.0, components),
            "fa": sums("final_false_pixels") / max(1.0, sums("pixels")),
            "ctr": sums("covered_detected") / max(1.0, covered),
            "tcr": sums("retained_covered_components") / max(1.0, covered),
            "flcc": covered - sums("retained_covered_components"),
            "fcrr": sums("rejected_background_queries") / max(1.0, background),
            "dsr": sums("rejected_duplicate_queries") / max(1.0, duplicates),
        }

    base_full = summarize(base_rows, np.arange(count))
    candidate_full = summarize(candidate_rows, np.arange(count))
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(int(samples)):
        indices = rng.integers(0, count, size=count)
        base = summarize(base_rows, indices)
        candidate = summarize(candidate_rows, indices)
        for field in base:
            values[field].append(candidate[field] - base[field])
    rows = []
    for field, samples_values in values.items():
        rows.append(
            {
                "metric": field,
                "base": base_full[field],
                "candidate": candidate_full[field],
                "delta": candidate_full[field] - base_full[field],
                "ci_low": float(np.quantile(samples_values, 0.025)),
                "ci_high": float(np.quantile(samples_values, 0.975)),
                "bootstrap_samples": int(samples),
            }
        )
    return rows


def gate_passed(summary: dict, independent_summary: dict, one_summary: dict) -> bool:
    iou_improved = (
        float(summary["global_iou"]) >= float(one_summary["global_iou"]) + 0.005
        and float(summary["mean_niou"]) >= float(one_summary["mean_niou"]) - 0.005
    )
    niou_improved = (
        float(summary["mean_niou"]) >= float(one_summary["mean_niou"]) + 0.005
        and float(summary["global_iou"]) >= float(one_summary["global_iou"]) - 0.005
    )
    return bool(
        int(summary["fully_lost_covered_components"]) <= 1
        and float(summary["covered_target_recovery"])
        >= float(independent_summary["covered_target_recovery"]) - 0.005
        and float(summary["false_candidate_rejection"]) >= 0.70
        and float(summary["fa"]) <= 48.24e-6
        # The frozen reference is 104/117 = 88.888...%; compare against the
        # exact attainable ratio instead of the rounded display value 88.89%.
        and float(summary["pd"]) >= (104.0 / 117.0) - 1e-12
        and (iou_improved or niou_improved)
    )


def randomize_groups(
    groups_by_image: list[tuple[tuple[int, ...], ...]],
    valid: np.ndarray,
    seed: int,
) -> list[tuple[tuple[int, ...], ...]]:
    rng = np.random.default_rng(seed)
    result = []
    for image_index, groups in enumerate(groups_by_image):
        sizes = [len(group) for group in groups]
        indices = np.flatnonzero(valid[image_index]).copy()
        rng.shuffle(indices)
        offset = 0
        new_groups = []
        for size in sizes:
            new_groups.append(tuple(sorted(int(value) for value in indices[offset : offset + size])))
            offset += size
        result.append(tuple(new_groups))
    return result


def render_cases(
    data: dict,
    data_root: Path,
    groups_by_image: list[tuple[tuple[int, ...], ...]],
    baseline_accepted: np.ndarray,
    selected_accepted: np.ndarray,
    selected_final: np.ndarray,
    objectness: np.ndarray,
    output_dir: Path,
) -> None:
    success_dir = output_dir / "success_cases"
    failure_dir = output_dir / "failure_cases"
    success_dir.mkdir(exist_ok=True)
    failure_dir.mkdir(exist_ok=True)
    baseline_metrics = candidate_metrics(data, baseline_accepted)
    _ = baseline_metrics
    rendered_success = 0
    rendered_failure = 0
    for image_index, name in enumerate(data["names"]):
        components = {
            int(value)
            for value in data["component"][image_index]
            if int(value) >= 0
        }
        before_retained = {
            int(data["component"][image_index, index])
            for index in np.flatnonzero(baseline_accepted[image_index])
            if int(data["component"][image_index, index]) >= 0
        }
        after_retained = {
            int(data["component"][image_index, index])
            for index in np.flatnonzero(selected_accepted[image_index])
            if int(data["component"][image_index, index]) >= 0
        }
        is_success = bool((components - before_retained) & after_retained)
        selected_prediction = selected_final[image_index] >= 0.5
        false_pixels = int(np.logical_and(selected_prediction, ~data["gt"][image_index]).sum())
        is_failure = bool(components - after_retained) or false_pixels > 25
        if (is_success and rendered_success >= 5) or (is_failure and rendered_failure >= 5):
            continue
        if not is_success and not is_failure:
            continue
        image_path = data_root / "images" / f"{name}.png"
        if image_path.exists():
            original = Image.open(image_path).convert("L").resize((256, 256)).convert("RGB")
        else:
            original = Image.new("RGB", (256, 256), "black")
        gt_panel = Image.fromarray((data["gt"][image_index] * 255).astype(np.uint8)).convert("RGB")
        mask_panel = Image.fromarray((selected_prediction * 255).astype(np.uint8)).convert("RGB")
        candidate_panel = original.copy()
        draw = ImageDraw.Draw(candidate_panel)
        group_lookup = {}
        for group_id, group in enumerate(groups_by_image[image_index]):
            for candidate in group:
                group_lookup[candidate] = group_id
        for candidate in np.flatnonzero(data["valid"][image_index]):
            x, y = data["xy"][image_index, candidate]
            accepted = bool(selected_accepted[image_index, candidate])
            color = "lime" if accepted else "red"
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            draw.text(
                (x + 4, y - 6),
                f"g{group_lookup.get(int(candidate), -1)}:{objectness[image_index, candidate]:.2f}",
                fill=color,
            )
        canvas = Image.new("RGB", (1024, 286), "white")
        for panel_index, (panel, title) in enumerate(
            ((original, "image"), (gt_panel, "GT analysis"), (candidate_panel, "groups/scores"), (mask_panel, "final mask"))
        ):
            canvas.paste(panel, (panel_index * 256, 30))
            ImageDraw.Draw(canvas).text((panel_index * 256 + 4, 6), title, fill="black")
        target_dir = success_dir if is_success else failure_dir
        canvas.save(target_dir / f"{name}.png")
        if is_success:
            rendered_success += 1
        else:
            rendered_failure += 1
        if rendered_success >= 5 and rendered_failure >= 5:
            break


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    feature_path = Path(args.features).resolve()
    target_path = Path(args.targets).resolve()
    checkpoint_path = Path(args.objectness_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_cases").mkdir(exist_ok=True)
    (output_dir / "success_cases").mkdir(exist_ok=True)
    data = load_data(feature_path, target_path)
    if data["valid"].shape[1] != 10:
        raise RuntimeError("S1b cache protocol is frozen at K=10")
    objectness, learned_quality = predict_objectness(
        data["descriptors"], data["valid"], checkpoint_path
    )
    relations = [
        pairwise_candidate_relations(
            data["xy"][index], data["query"][index], data["descriptors"][index]
        )
        for index in range(len(data["names"]))
    ]

    audit_rows: list[dict] = []
    audit_groups_cache: dict[str, list[tuple[tuple[int, ...], ...]]] = {}
    collision_rows_by_name: dict[str, list[dict]] = {}
    configs = grouping_configs()
    for config in configs:
        name = config_name(config)
        groups_by_image = []
        for image_index in range(len(data["names"])):
            graph = build_candidate_graph(relations[image_index], data["valid"][image_index], config)
            groups_by_image.append(
                connected_candidate_groups(graph, data["valid"][image_index], data["xy"][image_index])
            )
        summary, collision_rows = audit_groups(data, groups_by_image, objectness)
        audit_rows.append({"grouping": name, **config.__dict__, **summary})
        audit_groups_cache[name] = groups_by_image
        collision_rows_by_name[name] = collision_rows
    hybrid_rows = [row for row in audit_rows if row["mode"] == "hybrid"]
    admissible_hybrid = [
        row
        for row in hybrid_rows
        if int(row["collision_groups"]) <= 1
        and float(row["group_collision_rate"]) <= 0.01
    ]
    hybrid_pool = admissible_hybrid if admissible_hybrid else hybrid_rows
    best_hybrid = min(
        hybrid_pool,
        key=lambda row: (
            int(row["collision_groups"]),
            -float(row["duplicate_with_primary_grouping_rate"]),
            float(row["background_contamination_in_target_groups"]),
            int(row["groups"]),
        ),
    )
    base_hybrid_config = next(
        config for config in configs if config_name(config) == best_hybrid["grouping"]
    )
    for tau_feat in (0.7, 0.8, 0.9):
        config = GroupingConfig(
            "hybrid_feature",
            r_near=base_hybrid_config.r_near,
            r_far=base_hybrid_config.r_far,
            tau_iou=base_hybrid_config.tau_iou,
            r_mask=base_hybrid_config.r_mask,
            tau_feat=tau_feat,
        )
        name = config_name(config)
        groups_by_image = []
        for image_index in range(len(data["names"])):
            graph = build_candidate_graph(relations[image_index], data["valid"][image_index], config)
            groups_by_image.append(
                connected_candidate_groups(graph, data["valid"][image_index], data["xy"][image_index])
            )
        summary, collision_rows = audit_groups(data, groups_by_image, objectness)
        audit_rows.append({"grouping": name, **config.__dict__, **summary})
        audit_groups_cache[name] = groups_by_image
        collision_rows_by_name[name] = collision_rows

    # G2 remains the preregistered primary unless a feature relation strictly
    # reduces collisions or improves duplicate grouping without contamination.
    best_feature = min(
        [row for row in audit_rows if row["mode"] == "hybrid_feature"],
        key=lambda row: (
            int(row["collision_groups"]),
            -float(row["duplicate_with_primary_grouping_rate"]),
            float(row["background_contamination_in_target_groups"]),
            int(row["groups"]),
        ),
    )
    feature_better = (
        int(best_feature["collision_groups"]) < int(best_hybrid["collision_groups"])
        or (
            int(best_feature["collision_groups"]) == int(best_hybrid["collision_groups"])
            and float(best_feature["duplicate_with_primary_grouping_rate"])
            > float(best_hybrid["duplicate_with_primary_grouping_rate"]) + 0.05
            and float(best_feature["background_contamination_in_target_groups"])
            <= float(best_hybrid["background_contamination_in_target_groups"])
        )
    )
    selected_grouping_row = best_feature if feature_better else best_hybrid
    selected_grouping = str(selected_grouping_row["grouping"])
    groups_by_image = audit_groups_cache[selected_grouping]

    specs: list[dict] = [
        {"name": "A1-hard-0.15", "kind": "hard", "threshold": 0.15},
        {"name": "M2-1-independent-all", "kind": "hard", "threshold": 0.0},
    ]
    specs.extend(
        {
            "name": f"A2-top{value}",
            "kind": "top_l",
            "threshold": 0.15,
            "minimum_count": value,
        }
        for value in (1, 2, 3)
    )
    for tau_high in (0.15, 0.20, 0.30, 0.40):
        for tau_group in (0.05, 0.08, 0.10, 0.12, 0.15):
            if tau_group <= tau_high:
                specs.append(
                    {
                        "name": f"A3-th{tau_high:g}-tg{tau_group:g}",
                        "kind": "a3",
                        "tau_high": tau_high,
                        "tau_group": tau_group,
                        "champion_formula": "object",
                    }
                )
    for tau_high in (0.20, 0.30, 0.40, 0.50):
        for tau_low in (0.03, 0.05, 0.08, 0.10, 0.12):
            for tau_rescue in (0.05, 0.08, 0.10, 0.12, 0.15):
                for weight in (0.5, 0.7, 1.0):
                    specs.append(
                        {
                            "name": f"A4-th{tau_high:g}-tl{tau_low:g}-tr{tau_rescue:g}-w{weight:g}",
                            "kind": "a4",
                            "tau_high": tau_high,
                            "tau_low": tau_low,
                            "tau_rescue": tau_rescue,
                            "uncertain_weight": weight,
                            "champion_formula": "object",
                        }
                    )

    policy_rows: list[dict] = []
    patterns: dict[str, tuple[np.ndarray, np.ndarray, list[tuple[str, ...]], dict]] = {}
    for spec in specs:
        accepted, weights, states = apply_policy(spec, data, objectness, groups_by_image)
        key = pattern_hash(accepted, weights)
        row = {
            "policy": spec["name"],
            "grouping": selected_grouping if spec["kind"] in {"a3", "a4"} else "none",
            "pattern_sha256": key,
            **spec,
            **candidate_metrics(data, accepted),
        }
        policy_rows.append(row)
        patterns.setdefault(key, (accepted, weights, states, spec))

    group_policy_rows = [row for row in policy_rows if row["kind"] in {"a3", "a4"}]
    base_a5_row = min(
        group_policy_rows,
        key=lambda row: (
            int(row["fully_lost_covered_components"]),
            int(float(row["false_candidate_rejection"]) < 0.70),
            -float(row["false_candidate_rejection"]),
            int(row["accepted_candidates"]),
        ),
    )
    base_a5_spec = next(spec for spec in specs if spec["name"] == base_a5_row["policy"])
    for formula in ("object_raw", "object_quality", "object_raw_quality"):
        spec = dict(base_a5_spec)
        spec["champion_formula"] = formula
        spec["name"] = f"A5-{formula}-from-{base_a5_spec['name']}"
        specs.append(spec)
        accepted, weights, states = apply_policy(spec, data, objectness, groups_by_image)
        key = pattern_hash(accepted, weights)
        policy_rows.append(
            {
                "policy": spec["name"],
                "grouping": selected_grouping,
                "pattern_sha256": key,
                **spec,
                **candidate_metrics(data, accepted),
            }
        )
        patterns.setdefault(key, (accepted, weights, states, spec))

    # Evaluate every candidate-safe unique pattern plus the strongest failures
    # and mandatory controls. This avoids recomputing identical mask aggregates.
    mandatory_names = {
        "A1-hard-0.15",
        "M2-1-independent-all",
        "A2-top1",
        "A2-top2",
        "A2-top3",
    }
    ranked_rows = sorted(
        policy_rows,
        key=lambda row: (
            int(row["fully_lost_covered_components"]),
            int(float(row["false_candidate_rejection"]) < 0.70),
            -float(row["false_candidate_rejection"]),
            int(row["accepted_candidates"]),
        ),
    )
    evaluation_hashes = {
        row["pattern_sha256"]
        for row in policy_rows
        if row["policy"] in mandatory_names
        or (
            int(row["fully_lost_covered_components"]) <= 1
            and float(row["false_candidate_rejection"]) >= 0.70
        )
    }
    evaluation_hashes.update(row["pattern_sha256"] for row in ranked_rows[:25])
    evaluated: dict[str, dict] = {}
    for key in sorted(evaluation_hashes):
        accepted, weights, _, _ = patterns[key]
        evaluated[key] = evaluate_pattern(
            data, objectness, accepted, weights, args.mask_threshold
        )
    for row in policy_rows:
        result = evaluated.get(row["pattern_sha256"])
        if result is not None:
            row.update(result["summary"])
            row["evaluated_mask"] = 1
        else:
            row["evaluated_mask"] = 0

    independent_row = next(row for row in policy_rows if row["policy"] == "M2-1-independent-all")
    m0_rows = list(csv.DictReader(
        (REPO_ROOT / "outputs/microquery/M0_independent_query/IRSTD-1k/a1_best_mask_val/fixed_0_5_summary.csv").open(encoding="utf-8")
    ))
    one_row = next(
        row for row in m0_rows if row["budget"] == "10" and row["condition"] == "M0-One"
    )
    passing_rows = [
        row
        for row in policy_rows
        if int(row.get("evaluated_mask", 0))
        and gate_passed(row, independent_row, one_row)
    ]
    evaluated_rows = [row for row in policy_rows if int(row.get("evaluated_mask", 0))]
    if passing_rows:
        selected_row = min(
            passing_rows,
            key=lambda row: (
                int(row["fully_lost_covered_components"]),
                -float(row["covered_target_recovery"]),
                float(row["fa"]),
                -float(row["pd"]),
                -float(row["mean_niou"]),
                int(row["accepted_candidates"]),
            ),
        )
    else:
        selected_row = min(
            evaluated_rows,
            key=lambda row: (
                int(row["fully_lost_covered_components"]),
                int(float(row["false_candidate_rejection"]) < 0.70),
                -float(row["covered_target_recovery"]),
                float(row["fa"]),
                -float(row["pd"]),
                -float(row["mean_niou"]),
            ),
        )
    selected_key = selected_row["pattern_sha256"]
    selected_result = evaluated[selected_key]
    # A pattern can be shared by several policies.  Reconstruct the exact
    # selected policy so the manifest and counterfactuals use its real rule,
    # rather than whichever equivalent pattern was inserted first.
    selected_spec = next(spec for spec in specs if spec["name"] == selected_row["policy"])
    selected_accepted, selected_weights, selected_states = apply_policy(
        selected_spec, data, objectness, groups_by_image
    )

    # Required counterfactuals and grouping controls around the selected policy.
    counterfactual_specs: list[tuple[str, dict, np.ndarray, list, np.ndarray | None]] = []
    counterfactual_specs.append(("Correct", selected_spec, objectness, groups_by_image, None))
    counterfactual_specs.append(("Shuffled-objectness", selected_spec, np.roll(objectness, 1, axis=0), groups_by_image, None))
    counterfactual_specs.append(("Inverted-objectness", selected_spec, 1.0 - objectness, groups_by_image, None))
    counterfactual_specs.append(("Random-groups", selected_spec, objectness, randomize_groups(groups_by_image, data["valid"], args.seed), None))
    counterfactual_specs.append(("Zero-quality", selected_spec, objectness, groups_by_image, np.zeros_like(data["sam_quality"])))
    quality_rng = np.random.default_rng(args.seed)
    counterfactual_specs.append(
        (
            "Random-quality",
            selected_spec,
            objectness,
            groups_by_image,
            quality_rng.random(data["sam_quality"].shape, dtype=np.float32),
        )
    )
    counterfactual_specs.append(("Shuffled-quality", selected_spec, objectness, groups_by_image, np.roll(data["sam_quality"], 1, axis=0)))
    no_group_spec = {"name": "No-grouping", "kind": "hard", "threshold": float(selected_spec.get("tau_group", selected_spec.get("tau_low", 0.15)))}
    counterfactual_specs.append(("No-grouping", no_group_spec, objectness, groups_by_image, None))
    best_g0 = min(
        [row for row in audit_rows if row["mode"] == "coordinate"],
        key=lambda row: (int(row["collision_groups"]), -float(row["duplicate_with_primary_grouping_rate"]), int(row["groups"])),
    )
    best_g1 = min(
        [row for row in audit_rows if row["mode"] == "mask"],
        key=lambda row: (int(row["collision_groups"]), -float(row["duplicate_with_primary_grouping_rate"]), int(row["groups"])),
    )
    counterfactual_specs.append(("Coordinate-groups", selected_spec, objectness, audit_groups_cache[best_g0["grouping"]], None))
    counterfactual_specs.append(("Mask-groups", selected_spec, objectness, audit_groups_cache[best_g1["grouping"]], None))
    counterfactual_rows = []
    counterfactual_results: dict[str, dict] = {}
    for condition, spec, scores, group_values, quality_values in counterfactual_specs:
        accepted, weights, _ = apply_policy(spec, data, scores, group_values, quality_values)
        result = evaluate_pattern(data, scores, accepted, weights, args.mask_threshold)
        counterfactual_results[condition] = result
        counterfactual_rows.append({"condition": condition, **result["summary"]})

    # Mask-threshold curves for selected and mandatory baselines.
    curve_rows = []
    curve_conditions = {
        "selected": (selected_accepted, selected_weights),
        "A1-hard-0.15": patterns[next(row["pattern_sha256"] for row in policy_rows if row["policy"] == "A1-hard-0.15")][:2],
        "M2-1-independent-all": patterns[next(row["pattern_sha256"] for row in policy_rows if row["policy"] == "M2-1-independent-all")][:2],
    }
    for condition, (accepted, weights) in curve_conditions.items():
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
            result = evaluate_pattern(data, objectness, accepted, weights, threshold)
            curve_rows.append({"condition": condition, "mask_threshold": threshold, **result["summary"]})

    hard_key = next(
        row["pattern_sha256"] for row in policy_rows if row["policy"] == "A1-hard-0.15"
    )
    hard_result = evaluated.get(hard_key)
    if hard_result is None:
        raise RuntimeError("A1-hard-0.15 was not evaluated")
    bootstrap_rows = bootstrap_delta(
        hard_result["per_image_rows"],
        selected_result["per_image_rows"],
        args.bootstrap_samples,
        args.seed,
    )

    # Selected grouping/member tables.
    group_rows = []
    member_rows = []
    candidate_rows = []
    for image_index, groups in enumerate(groups_by_image):
        summaries = summarize_candidate_groups(
            groups,
            data["xy"][image_index],
            objectness[image_index],
            data["raw"][image_index],
            data["sam_quality"][image_index],
            relations[image_index].soft_iou,
        )
        for row in summaries:
            indices = row.pop("candidate_indices")
            target_components = sorted(
                {
                    int(data["component"][image_index, candidate])
                    for candidate in indices
                    if data["semantic"][image_index, candidate]
                    and int(data["component"][image_index, candidate]) >= 0
                }
            )
            group_rows.append(
                {
                    "image": data["names"][image_index],
                    **row,
                    "candidate_indices": " ".join(str(index) for index in indices),
                    "analysis_target_components": " ".join(str(value) for value in target_components),
                    "analysis_collision": int(len(target_components) >= 2),
                    "accepted_candidates": sum(selected_accepted[image_index, candidate] for candidate in indices),
                }
            )
            for candidate in indices:
                member_rows.append(
                    {
                        "image": data["names"][image_index],
                        "group_id": row["group_id"],
                        "candidate_index": candidate,
                    }
                )
        for candidate in range(data["valid"].shape[1]):
            candidate_rows.append(
                {
                    "image": data["names"][image_index],
                    "candidate_index": candidate,
                    "x": float(data["xy"][image_index, candidate, 0]),
                    "y": float(data["xy"][image_index, candidate, 1]),
                    "valid": int(data["valid"][image_index, candidate]),
                    "raw_score": float(data["raw"][image_index, candidate]),
                    "objectness": float(objectness[image_index, candidate]),
                    "sam_quality": float(data["sam_quality"][image_index, candidate]),
                    "learned_quality": float(learned_quality[image_index, candidate]),
                    "accepted": int(selected_accepted[image_index, candidate]),
                    "weight": float(selected_weights[image_index, candidate]),
                    "state": selected_states[image_index][candidate],
                    "analysis_semantic": int(data["semantic"][image_index, candidate]),
                    "analysis_primary": int(data["primary"][image_index, candidate]),
                    "analysis_duplicate": int(data["duplicate"][image_index, candidate]),
                    "analysis_component": int(data["component"][image_index, candidate]),
                }
            )

    write_csv(output_dir / "group_metrics_summary.csv", audit_rows)
    write_csv(output_dir / "groups_per_image.csv", group_rows)
    write_csv(output_dir / "group_members.csv", member_rows)
    write_csv(output_dir / "group_collision_cases.csv", collision_rows_by_name[selected_grouping])
    write_csv(output_dir / "policy_search.csv", policy_rows)
    write_csv(output_dir / "threshold_curve.csv", curve_rows)
    write_csv(output_dir / "counterfactual_summary.csv", counterfactual_rows)
    write_csv(output_dir / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(output_dir / "per_image.csv", selected_result["per_image_rows"])
    write_csv(output_dir / "per_component.csv", selected_result["per_component_rows"])
    write_csv(output_dir / "per_candidate.csv", candidate_rows)
    write_csv(output_dir / "per_group.csv", group_rows)

    if args.data_root:
        hard_spec = {"name": "A1-hard-0.15", "kind": "hard", "threshold": 0.15}
        hard_accepted, _, _ = apply_policy(hard_spec, data, objectness, groups_by_image)
        render_cases(
            data,
            Path(args.data_root).resolve(),
            groups_by_image,
            hard_accepted,
            selected_accepted,
            selected_result["final_probability"],
            objectness,
            output_dir,
        )

    manifest = {
        "schema_version": 1,
        "experiment": "MicroQuery M2-S1b cache-level component-safe rejection",
        "dataset": args.dataset,
        "images": len(data["names"]),
        "candidate_budget": 10,
        "candidate_cache_sha256": sha256_file(Path(args.candidate_cache).resolve()),
        "feature_cache_sha256": sha256_file(feature_path),
        "query_mask_cache_sha256": sha256_file(feature_path),
        "analysis_target_sha256": sha256_file(target_path),
        "objectness_checkpoint_sha256": sha256_file(checkpoint_path),
        "a1_checkpoint_sha256": sha256_file(Path(args.a1_checkpoint).resolve()),
        "probe_checkpoint_sha256": sha256_file(Path(args.probe_checkpoint).resolve()),
        "val_split_sha256": sha256_file(Path(args.val_split).resolve()),
        "git_commit": current_git_commit(),
        "selected_grouping": selected_grouping,
        "selected_policy": selected_row["policy"],
        "mask_threshold": args.mask_threshold,
        "bootstrap_samples": args.bootstrap_samples,
        "gt_boundary": (
            "Grouping and policy selection functions consume only deployable candidate coordinates, "
            "query probabilities, descriptors, scores, qualities, and validity. GT analysis labels "
            "are read only after grouping for validation metrics and never enter deployable forward."
        ),
    }
    resolved = {
        **vars(args),
        "selected_grouping_config": {
            key: value
            for key, value in selected_grouping_row.items()
            if key in GroupingConfig.__dataclass_fields__
        },
        "selected_policy_spec": selected_spec,
    }
    summary = {
        "manifest": manifest,
        "grouping": selected_grouping_row,
        "selected_policy": selected_row,
        "cache_level_gate_passed_before_hard_negative": bool(passing_rows),
        "passing_configurations": len(passing_rows),
        "one_query_reference": one_row,
        "independent_reference": independent_row,
        "counterfactuals": counterfactual_rows,
        "decision": (
            "Run hard-negative safety validation before declaring A-stage pass"
            if passing_rows
            else "Cache-level validation gate failed; proceed sequentially to B1/B2"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
