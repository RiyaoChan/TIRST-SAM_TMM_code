#!/usr/bin/env python3
"""Replay the frozen image-only probe online and audit cache/deployment equivalence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery_gate_deployment import GateDeploymentConfig
from efficient_sam.microquery_metrics import assign_candidates
from efficient_sam.prompt_metrics import extract_components
from scripts.audit_microquery_gate_deployment import DEFAULT_C1, DEFAULT_F1, dump_json
from scripts.eval_microquery_end2end import build_from_checkpoint, write_csv
from scripts.microquery_end2end_dataset import MicroQueryEndToEndDataset
from scripts.microquery_end2end_metrics import FullMaskMetricAccumulator
from scripts.microquery_end2end_runtime import encode_frozen_image, forward_deployable
from scripts.train_experiment1_single_view import ImageOnlyProposalGenerator
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


DEFAULT_PROBE = "outputs/experiment1_p0/IRSTD-1k/probe_20ep_seed20260825/best_neck.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", default="outputs/microquery/gate_deployment_audit/IRSTD-1k")
    parser.add_argument("--c1_checkpoint", default=DEFAULT_C1)
    parser.add_argument("--f1_checkpoint", default=DEFAULT_F1)
    parser.add_argument("--probe_checkpoint", default=DEFAULT_PROBE)
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def coordinate_replay_metrics(
    cached_xy: np.ndarray,
    cached_scores: np.ndarray,
    cached_valid: np.ndarray,
    online_xy: np.ndarray,
    online_scores: np.ndarray,
    online_valid: np.ndarray,
) -> dict:
    if cached_xy.shape != online_xy.shape or cached_valid.shape != online_valid.shape:
        raise ValueError("cache and online candidates must have identical shapes")
    selected = cached_valid & online_valid
    distances = np.linalg.norm(cached_xy - online_xy, axis=-1)
    valid_union = cached_valid | online_valid
    return {
        "candidate_slots": int(cached_valid.size),
        "joint_valid_slots": int(selected.sum()),
        "validity_exact_fraction": float((cached_valid == online_valid).mean()),
        "coordinate_within_0_5_fraction": float((distances[selected] <= 0.5).mean()) if selected.any() else 1.0,
        "coordinate_within_1_0_fraction": float((distances[selected] <= 1.0).mean()) if selected.any() else 1.0,
        "coordinate_mean_distance": float(distances[selected].mean()) if selected.any() else 0.0,
        "coordinate_max_distance": float(distances[selected].max()) if selected.any() else 0.0,
        "rank_agreement_within_1px": float((distances[valid_union] <= 1.0).mean()) if valid_union.any() else 1.0,
        "score_mean_absolute_error": float(np.abs(cached_scores[selected] - online_scores[selected]).mean()) if selected.any() else 0.0,
        "score_max_absolute_error": float(np.abs(cached_scores[selected] - online_scores[selected]).max()) if selected.any() else 0.0,
    }


def selected_gate_config(decision: dict) -> GateDeploymentConfig:
    config = decision["selected_gate_config"]
    if config["mode"] == "legacy_checkpoint_schedule":
        raise RuntimeError("online deployment cannot select an implicit legacy schedule")
    return GateDeploymentConfig(
        config["mode"], rho=float(config.get("rho", 0.0)), temperature=float(config.get("temperature", 1.0))
    )


def make_generator(probe_checkpoint: Path, device: torch.device) -> ImageOnlyProposalGenerator:
    return ImageOnlyProposalGenerator(
        SimpleNamespace(
            generator="probe",
            candidate_k_raw=32,
            nms_radius=3.0,
            score_threshold=0.1,
            probe_checkpoint=str(probe_checkpoint),
        ),
        device,
    )


def proposal_to_deployable(images: torch.Tensor, proposal, budget: int = 10) -> dict[str, torch.Tensor]:
    return {
        "image": images,
        "candidate_xy": proposal.candidate_xy[:, :budget],
        "candidate_scores": proposal.candidate_scores[:, :budget],
        "candidate_valid": proposal.candidate_valid[:, :budget],
    }


def benchmark_online_chain(model, head, variant, generator, sample, gate_config, device, args) -> dict:
    deployable_cache = {key: value.unsqueeze(0).to(device) for key, value in sample["deployable"].items()}
    images = deployable_cache["image"]
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    def elapsed(function, repeats):
        for _ in range(args.warmup):
            function()
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        sync()
        return (time.perf_counter() - start) * 1000.0 / repeats

    def encode():
        return encode_frozen_image(model, images)

    encoded = encode()

    def propose():
        return generator(images, encoded[0], [encoded[2]])

    proposal = propose()
    deployable = proposal_to_deployable(images, proposal)

    def decode():
        with autocast_context(device, args.amp_dtype):
            return forward_deployable(
                model,
                head,
                deployable,
                variant=variant,
                gate_deployment_config=gate_config,
                query_chunk=args.query_chunk,
                image_embeddings=encoded,
            )

    def total():
        current_encoded = encode()
        current_proposal = generator(images, current_encoded[0], [current_encoded[2]])
        current_deployable = proposal_to_deployable(images, current_proposal)
        with autocast_context(device, args.amp_dtype):
            return forward_deployable(
                model,
                head,
                current_deployable,
                variant=variant,
                gate_deployment_config=gate_config,
                query_chunk=args.query_chunk,
                image_embeddings=current_encoded,
            )

    with torch.inference_mode():
        encoder_ms = elapsed(encode, args.repeats)
        probe_ms = elapsed(propose, args.repeats)
        microquery_decoder_ms = elapsed(decode, args.repeats)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total_ms = elapsed(total, args.repeats)
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "encoder_ms": encoder_ms,
        "online_probe_ms": probe_ms,
        "microquery_head_prompt_decoder_ms": microquery_decoder_ms,
        "sum_of_components_ms": encoder_ms + probe_ms + microquery_decoder_ms,
        "end_to_end_online_ms": total_ms,
        "peak_gpu_memory_bytes": int(peak),
        "encoder_calls_per_end_to_end": 1,
        "decoder_queries": 10,
    }


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    audit_dir = resolve_repo_path(args.audit_dir)
    output_dir = audit_dir / "online_probe"
    decision = json.loads((audit_dir / "comparison" / "decision.json").read_text(encoding="utf-8"))
    gate_config = selected_gate_config(decision)
    data_root = resolve_repo_path(args.data_root)
    val_split = Path(args.val_split).resolve() if Path(args.val_split).is_absolute() else data_root / args.val_split
    val_cache = resolve_repo_path(args.val_candidate_cache)
    probe_path = resolve_repo_path(args.probe_checkpoint)
    a1 = resolve_repo_path(args.a1_checkpoint)
    weights = resolve_repo_path(args.weights)
    dataset = MicroQueryEndToEndDataset(
        data_root=data_root, split=val_split, candidate_cache=val_cache, augment=False, budget=10, seed=args.seed
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    online_rows = []
    model_results = []
    replay_arrays = None
    for model_id, checkpoint_path, expected_variant in (
        ("C1", resolve_repo_path(args.c1_checkpoint), "c1_independent_aux"),
        ("F1", resolve_repo_path(args.f1_checkpoint), "f1_soft_gate"),
    ):
        checkpoint, variant, model, head = build_from_checkpoint(checkpoint_path, a1, weights, device)
        if variant != expected_variant:
            raise RuntimeError(f"{model_id} checkpoint variant mismatch")
        generator = make_generator(probe_path, device)
        condition = str(decision["selected_condition"])
        matrix_path = audit_dir / "comparison" / "full_gate_matrix.csv"
        with matrix_path.open(encoding="utf-8-sig", newline="") as handle:
            import csv

            matrix = list(csv.DictReader(handle))
        matrix_row = next(row for row in matrix if row["id"] == f"{model_id}-{condition}")
        threshold = float(matrix_row["matched_pd_threshold"])
        accumulator = FullMaskMetricAccumulator(threshold, 3.0)
        names, xy_rows, score_rows, valid_rows = [], [], [], []
        coverage_components = coverage_hit = 0
        with torch.inference_mode():
            for batch in loader:
                images = batch["deployable"]["image"].to(device)
                encoded = encode_frozen_image(model, images)
                proposal = generator(images, encoded[0], [encoded[2]])
                deployable = proposal_to_deployable(images, proposal)
                with autocast_context(device, args.amp_dtype):
                    output = forward_deployable(
                        model,
                        head,
                        deployable,
                        variant=variant,
                        gate_deployment_config=gate_config,
                        query_chunk=args.query_chunk,
                        image_embeddings=encoded,
                    )
                batch_names = list(batch["meta"]["name"])
                accumulator.update(
                    names=batch_names,
                    probabilities=output.final_probability,
                    targets=batch["supervision"]["full_mask"],
                    candidate_valid=deployable["candidate_valid"],
                    candidate_scores=deployable["candidate_scores"],
                    semantic_labels=batch["supervision"]["semantic_labels"],
                    object_scores=output.raw_gates,
                )
                batch_xy = deployable["candidate_xy"].detach().float().cpu().numpy()
                batch_score = deployable["candidate_scores"].detach().float().cpu().numpy()
                batch_valid = deployable["candidate_valid"].detach().cpu().numpy().astype(bool)
                names.extend(batch_names)
                xy_rows.append(batch_xy)
                score_rows.append(batch_score)
                valid_rows.append(batch_valid)
                masks = batch["supervision"]["full_mask"].numpy() > 0
                for index, name in enumerate(batch_names):
                    components = extract_components(masks[index])
                    assignment = assign_candidates(batch_xy[index], batch_valid[index], components, budget=10)
                    coverage_components += len(components)
                    coverage_hit += len(assignment.covered_components)
                    online_rows.append(
                        {
                            "model": model_id,
                            "image": name,
                            "components": len(components),
                            "covered_components": len(assignment.covered_components),
                        }
                    )
        current_arrays = {
            "names": np.asarray(names), "xy": np.concatenate(xy_rows),
            "scores": np.concatenate(score_rows), "valid": np.concatenate(valid_rows),
        }
        if replay_arrays is None:
            replay_arrays = current_arrays
        else:
            if not (
                np.array_equal(replay_arrays["names"], current_arrays["names"])
                and np.array_equal(replay_arrays["xy"], current_arrays["xy"])
                and np.array_equal(replay_arrays["scores"], current_arrays["scores"])
                and np.array_equal(replay_arrays["valid"], current_arrays["valid"])
            ):
                raise RuntimeError("C1/F1 online candidates differ despite the shared frozen source")
        metrics = accumulator.finalize()
        metrics.update(
            {
                "model": model_id,
                "condition": condition,
                "threshold": threshold,
                "online_coverage_at_10": coverage_hit / max(1, coverage_components),
                "coverage_components": coverage_components,
                "covered_components": coverage_hit,
            }
        )
        offline_reference = {
            "global_iou": float(matrix_row["matched_pd_global_iou"]),
            "mean_niou": float(matrix_row["matched_pd_mean_niou"]),
            "f1": float(matrix_row["matched_pd_f1"]),
            "pd": float(matrix_row["matched_pd_pd"]),
            "fa": float(matrix_row["matched_pd_fa_per_million"]) / 1e6,
        }
        offline_difference = {
            key: float(metrics[key]) - value for key, value in offline_reference.items()
        }
        if any(abs(value) > 1e-12 for value in offline_difference.values()):
            raise RuntimeError(
                f"{model_id} online final metrics differ from the frozen offline audit: "
                f"{offline_difference}"
            )
        metrics["offline_reference"] = offline_reference
        metrics["online_minus_offline"] = offline_difference
        model_results.append(metrics)
        if model_id == "C1":
            latency = benchmark_online_chain(model, head, variant, generator, dataset[0], gate_config, device, args)
            dump_json(output_dir / "latency.json", latency)
        del model, head, generator
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cached = dataset.candidates
    replay = coordinate_replay_metrics(
        cached["candidate_xy"], cached["candidate_scores"], cached["candidate_valid"],
        replay_arrays["xy"], replay_arrays["scores"], replay_arrays["valid"],
    )
    replay.update(
        {
            "cache_sha256": sha256_file(val_cache),
            "probe_checkpoint_sha256": sha256_file(probe_path),
            "val_split_sha256": sha256_file(val_split),
            "images": len(dataset),
            "candidate_budget": 10,
            "same_candidates_for_c1_f1": True,
            "test_split_read": False,
        }
    )
    dump_json(output_dir / "candidate_replay_summary.json", replay)
    dump_json(output_dir / "online_final_metrics.json", model_results)
    dump_json(
        output_dir / "offline_online_equivalence.json",
        {
            row["model"]: {
                "offline_reference": row["offline_reference"],
                "online_minus_offline": row["online_minus_offline"],
            }
            for row in model_results
        },
    )
    write_csv(output_dir / "per_image_coverage.csv", online_rows)
    print(json.dumps(json_safe({"replay": replay, "metrics": model_results}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
