#!/usr/bin/env python3
"""Freeze A1 proposal coordinates and write physically separated GT analysis labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.microquery_metrics import assign_candidates
from efficient_sam.prompt_metrics import area_bin, extract_components
from scripts.eval_prompt_quality import sha256_file
from scripts.train_experiment1_single_view import ImageOnlyProposalGenerator, set_deterministic
from sirst_dataset import make_loader


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not budgets or min(budgets) <= 0:
        raise argparse.ArgumentTypeError("budgets must contain positive integers")
    return budgets


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe_checkpoint", required=True)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="IRSTD-1k")
    parser.add_argument("--budgets", type=parse_budgets, default=(5, 10, 20))
    parser.add_argument("--candidate_k_raw", type=int, default=32)
    parser.add_argument("--nms_radius", type=float, default=3.0)
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if max(args.budgets) > args.candidate_k_raw:
        raise ValueError("budgets cannot exceed candidate_k_raw")
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    checkpoint_path = Path(args.checkpoint).resolve()
    probe_path = Path(args.probe_checkpoint).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    generator = ImageOnlyProposalGenerator(
        SimpleNamespace(
            generator="probe",
            candidate_k_raw=args.candidate_k_raw,
            nms_radius=args.nms_radius,
            score_threshold=args.score_threshold,
            probe_checkpoint=str(probe_path),
        ),
        device,
    )
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

    names: list[str] = []
    xy_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []
    valid_batches: list[np.ndarray] = []
    candidate_rows: list[dict] = []
    analysis_rows: list[dict] = []
    component_rows: list[dict] = []
    image_budget_rows: list[dict] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc="microquery-P0")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            encoded = model.get_image_embeddings(images)
            proposal = generator(images, encoded[0], encoded[2])
            batch_xy = proposal.candidate_xy.detach().cpu().numpy().astype(np.float32)
            batch_scores = proposal.candidate_scores.detach().cpu().numpy().astype(np.float32)
            batch_valid = proposal.candidate_valid.detach().cpu().numpy().astype(bool)
            masks = (batch["mask"].detach().cpu().numpy() > 0)
            batch_names = [str(name) for name in batch["name"]]
            names.extend(batch_names)
            xy_batches.append(batch_xy)
            score_batches.append(batch_scores)
            valid_batches.append(batch_valid)
            for image_index, name in enumerate(batch_names):
                components = extract_components(masks[image_index])
                assignment = assign_candidates(
                    batch_xy[image_index],
                    batch_valid[image_index],
                    components,
                    budget=args.candidate_k_raw,
                )
                for candidate_index in range(args.candidate_k_raw):
                    candidate_rows.append(
                        {
                            "image": name,
                            "candidate_rank": candidate_index + 1,
                            "x": float(batch_xy[image_index, candidate_index, 0]),
                            "y": float(batch_xy[image_index, candidate_index, 1]),
                            "candidate_score": float(batch_scores[image_index, candidate_index]),
                            "candidate_valid": int(batch_valid[image_index, candidate_index]),
                            "candidate_source": "A1_neck_probe",
                        }
                    )
                    analysis_rows.append(
                        {
                            "image": name,
                            "candidate_rank": candidate_index + 1,
                            "semantic_candidate_label": (
                                "target-associated"
                                if assignment.semantic_target[candidate_index]
                                else "background"
                            ),
                            "assignment_label": assignment.assignment[candidate_index],
                            "matched_component_id": int(
                                assignment.component_index[candidate_index]
                            ),
                        }
                    )
                for component in components:
                    best_rank = assignment.best_rank_by_component.get(component.index, -1)
                    row = {
                        "image": name,
                        "component_index": component.index,
                        "area": component.area,
                        "area_bin": area_bin(component.area),
                        "best_candidate_rank": best_rank,
                    }
                    for budget in args.budgets:
                        row[f"covered_at_{budget}"] = int(0 < best_rank <= budget)
                    component_rows.append(row)
                for budget in args.budgets:
                    valid_count = int(batch_valid[image_index, :budget].sum())
                    semantic = assignment.semantic_target[:budget] & batch_valid[image_index, :budget]
                    duplicates = sum(
                        label_name == "duplicate"
                        for label_name, is_valid in zip(
                            assignment.assignment[:budget], batch_valid[image_index, :budget]
                        )
                        if is_valid
                    )
                    covered_count = sum(
                        0 < rank <= budget for rank in assignment.best_rank_by_component.values()
                    )
                    image_budget_rows.append(
                        {
                            "image": name,
                            "budget": budget,
                            "components": len(components),
                            "covered_components": covered_count,
                            "all_targets_covered": int(
                                bool(components) and covered_count == len(components)
                            ),
                            "valid_candidates": valid_count,
                            "target_associated_candidates": int(semantic.sum()),
                            "background_candidates": int(valid_count - semantic.sum()),
                            "duplicate_candidates": int(duplicates),
                        }
                    )

    if not names:
        raise RuntimeError("No images were cached")
    xy = np.concatenate(xy_batches, axis=0)
    scores = np.concatenate(score_batches, axis=0)
    valid = np.concatenate(valid_batches, axis=0)
    cache_path = output_dir / "candidates.npz"
    np.savez_compressed(
        cache_path,
        image_names=np.asarray(names),
        candidate_xy=xy,
        candidate_scores=scores,
        candidate_valid=valid,
    )
    write_csv(output_dir / "per_image_candidates.csv", candidate_rows)
    write_csv(output_dir / "analysis_candidate_labels.csv", analysis_rows)
    write_csv(output_dir / "per_component_coverage.csv", component_rows)

    curve_rows = []
    total_components = len(component_rows)
    for budget in args.budgets:
        rows = [row for row in image_budget_rows if int(row["budget"]) == budget]
        covered = sum(int(row["covered_components"]) for row in rows)
        target_images = [row for row in rows if int(row["components"]) > 0]
        curve_rows.append(
            {
                "budget": budget,
                "components": total_components,
                "covered_components": covered,
                "candidate_coverage": covered / max(1, total_components),
                "all_target_image_coverage": sum(
                    int(row["all_targets_covered"]) for row in target_images
                )
                / max(1, len(target_images)),
                "tiny_1_9_coverage": sum(
                    int(row[f"covered_at_{budget}"])
                    for row in component_rows
                    if row["area_bin"] == "1-9"
                )
                / max(1, sum(row["area_bin"] == "1-9" for row in component_rows)),
                "small_10_16_coverage": sum(
                    int(row[f"covered_at_{budget}"])
                    for row in component_rows
                    if row["area_bin"] == "10-16"
                )
                / max(1, sum(row["area_bin"] == "10-16" for row in component_rows)),
                "medium_17_25_coverage": sum(
                    int(row[f"covered_at_{budget}"])
                    for row in component_rows
                    if row["area_bin"] == "17-25"
                )
                / max(1, sum(row["area_bin"] == "17-25" for row in component_rows)),
                "large_gt25_coverage": sum(
                    int(row[f"covered_at_{budget}"])
                    for row in component_rows
                    if row["area_bin"] == ">25"
                )
                / max(1, sum(row["area_bin"] == ">25" for row in component_rows)),
                "duplicate_candidates_per_component": sum(
                    int(row["duplicate_candidates"]) for row in rows
                )
                / max(1, total_components),
                "false_candidates_per_image": sum(
                    int(row["background_candidates"]) for row in rows
                )
                / max(1, len(rows)),
            }
        )
    write_csv(output_dir / "coverage_budget_curve.csv", curve_rows)
    manifest = {
        "schema_version": 1,
        "experiment": "MicroQuery P0 frozen candidate cache",
        "dataset": args.dataset,
        "images": len(names),
        "candidate_source": "A1 single-view neck SpatialProbeHead",
        "candidate_k_raw": args.candidate_k_raw,
        "budgets": list(args.budgets),
        "nms_radius": args.nms_radius,
        "score_threshold": args.score_threshold,
        "split_sha256": sha256_file(split_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "probe_checkpoint_sha256": sha256_file(probe_path),
        "weights_sha256": sha256_file(weights_path),
        "cache_sha256": sha256_file(cache_path),
        "git_commit": current_git_commit(),
        "preprocessing": "SCTransNet normalization + fixed resize 256x256",
        "gt_boundary": (
            "candidates.npz and per_image_candidates.csv contain no GT; "
            "GT-derived labels are isolated in analysis_candidate_labels.csv and coverage files"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "sha256.txt").write_text(
        f"{manifest['cache_sha256']}  candidates.npz\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "coverage": curve_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

