#!/usr/bin/env python3
"""Evaluate image-only prompt generators with the Experiment 1 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.prompt_metrics import DEFAULT_BUDGETS, PromptMetricAccumulator
from efficient_sam.prompt_proposal import (
    DoGLoGProposalAdapter,
    PGAPProposalAdapter,
    DenseHeadProposalAdapter,
)
from efficient_sam.prompt_training import SpatialProbeHead
from sirst_dataset import make_loader


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_budgets(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("budgets must be a comma-separated list of positive integers")
    return values


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_probe(checkpoint_path: Path, weights_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    level = str(checkpoint["level"])
    config = dict(checkpoint["head_config"])
    head = SpatialProbeHead(
        in_channels=int(config["in_channels"]),
        hidden_channels=int(config.get("hidden_channels", 64)),
    )
    head.load_state_dict(checkpoint["head_state"], strict=True)
    head.to(device).eval()
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(weights_path),
        use_adapter=False,
        return_encoder_multi_scale=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, head, level, checkpoint


def select_probe_features(level: str, neck: torch.Tensor, multi_scale: list[torch.Tensor]) -> torch.Tensor:
    if level == "early":
        return multi_scale[0]
    if level == "mid":
        return multi_scale[2]
    if level == "neck":
        return neck
    raise ValueError(f"Unsupported probe level {level!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split_role", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen_config_manifest", default=None)
    parser.add_argument("--generator", choices=("pgap", "doglog", "probe"), required=True)
    parser.add_argument("--probe_checkpoint", default=None)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--candidate_k_raw", type=int, default=32)
    parser.add_argument("--nms_radius", type=float, default=3.0)
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--budgets", default=",".join(str(item) for item in DEFAULT_BUDGETS))
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split_role == "test" and not args.frozen_config_manifest:
        raise ValueError("Test evaluation requires --frozen_config_manifest from validation selection")
    if args.generator == "probe" and not args.probe_checkpoint:
        raise ValueError("--generator probe requires --probe_checkpoint")
    set_deterministic(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = parse_budgets(args.budgets)

    loader = make_loader(
        str(data_root),
        str(split_path),
        size=int(args.size),
        batch_size=int(args.batch_size),
        augment=False,
        shuffle=False,
        workers=int(args.workers),
        keep_ratio_pad=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=True,
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_eval_crop="resize",
        mllm_features_path=None,
    )

    probe_checkpoint = None
    probe_info = None
    if args.generator == "pgap":
        adapter = PGAPProposalAdapter(
            candidate_k_raw=args.candidate_k_raw,
            nms_radius=args.nms_radius,
            score_threshold=args.score_threshold,
        ).to(device)
        model = head = level = None
    elif args.generator == "doglog":
        adapter = DoGLoGProposalAdapter(
            candidate_k_raw=args.candidate_k_raw,
            nms_radius=args.nms_radius,
            score_threshold=args.score_threshold,
        ).to(device)
        model = head = level = None
    else:
        probe_checkpoint = Path(args.probe_checkpoint).resolve()
        weights_path = Path(args.weights).resolve()
        model, head, level, checkpoint = build_probe(probe_checkpoint, weights_path, device)
        adapter = DenseHeadProposalAdapter(
            head,
            candidate_k_raw=args.candidate_k_raw,
            nms_radius=args.nms_radius,
            score_threshold=args.score_threshold,
        )
        probe_info = {
            "checkpoint_sha256": sha256_file(probe_checkpoint),
            "epoch": int(checkpoint.get("epoch", -1)),
            "level": level,
            "selection_metrics": checkpoint.get("metrics", {}),
        }

    accumulator = PromptMetricAccumulator(budgets=budgets)
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"prompt-eval:{args.generator}")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            names = list(batch["name"])
            if args.generator == "probe":
                encoded = model.get_image_embeddings(images)
                neck, multi_scale = encoded[0], encoded[2]
                features = select_probe_features(level, neck, multi_scale)
                proposal = adapter(features, output_size=(args.size, args.size))
            else:
                proposal = adapter(images)
            accumulator.update(proposal, masks, names)

    result = accumulator.finalize()
    manifest = {
        "schema_version": 1,
        "experiment": "TIRST-SAM Experiment 1 prompt screening",
        "generator": args.generator,
        "split_role": args.split_role,
        "split_sha256": sha256_file(split_path),
        "seed": int(args.seed),
        "deterministic_eval": True,
        "preprocessing": "SCTransNet normalization + fixed resize 256x256",
        "candidate_k_raw": int(args.candidate_k_raw),
        "nms_radius": float(args.nms_radius),
        "score_threshold": float(args.score_threshold),
        "budgets": list(budgets),
        "gt_boundary": "GT is passed only to PromptMetricAccumulator after proposals are finalized",
        "probe": probe_info,
        "max_batches": int(args.max_batches),
    }
    summary_payload = {"manifest": manifest, "metrics": result["summary"]}
    (output_dir / "prompt_metrics_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "prompt_metrics_per_image.csv", result["per_image_rows"])
    write_csv(output_dir / "prompt_metrics_per_component.csv", result["per_component_rows"])
    write_csv(output_dir / "candidate_budget_curve.csv", result["budget_rows"])
    write_csv(output_dir / "area_bin_metrics.csv", result["area_rows"])
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=True))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
