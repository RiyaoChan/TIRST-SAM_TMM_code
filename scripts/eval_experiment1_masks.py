#!/usr/bin/env python3
"""Evaluate A0/A1/A2/A3 masks with frozen validation-selected configurations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.multiview_prompt import DEFAULT_VIEWS, multiview_propose
from efficient_sam.prompt_metrics import PromptMetricAccumulator
from scripts.eval_prompt_quality import enforce_test_config_freeze, sha256_file, write_csv
from scripts.train_experiment1_single_view import (
    DetectionAccumulator,
    ImageOnlyProposalGenerator,
    make_sam_inputs,
    set_deterministic,
)
from sirst_dataset import make_loader


class SharedModelImageGenerator:
    """Run the selected image-only generator with the evaluated model encoder."""

    def __init__(self, model, generator: ImageOnlyProposalGenerator):
        self.model = model
        self.generator = generator

    def __call__(self, images: torch.Tensor):
        if self.generator.kind in {"null", "pgap", "doglog"}:
            return self.generator(images, images.new_empty(0), None)
        encoded = self.model.get_image_embeddings(images)
        return self.generator(images, encoded[0], encoded[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split_role", choices=("val", "test"), default="val")
    parser.add_argument("--frozen_config_manifest", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--probe_checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("A0", "A1", "A2", "A3"), required=True)
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--cluster_radius", type=float, default=3.0)
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--max_dispersion", type=float, default=2.0)
    parser.add_argument("--reliability_threshold", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--score_mode", choices=("mean_max", "mean", "max", "support"), default="mean_max")
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_test_config_freeze(args.split_role, args.frozen_config_manifest)
    set_deterministic(20260825)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resolved = dict(checkpoint["resolved_args"])
    generator_kind = "null" if args.mode == "A0" else str(resolved["generator"])
    probe_checkpoint = args.probe_checkpoint or resolved.get("probe_checkpoint")
    if generator_kind == "probe" and not probe_checkpoint:
        raise ValueError("Probe evaluation requires --probe_checkpoint")

    generator_args = SimpleNamespace(
        generator=generator_kind,
        candidate_k_raw=int(resolved.get("candidate_k_raw", 32)),
        nms_radius=float(resolved.get("nms_radius", 3.0)),
        score_threshold=float(resolved.get("score_threshold", 0.1)),
        probe_checkpoint=probe_checkpoint,
    )

    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(Path(args.weights).resolve()),
        use_adapter=False,
        return_encoder_multi_scale=generator_kind == "probe",
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    generator = ImageOnlyProposalGenerator(generator_args, device)
    shared_generator = SharedModelImageGenerator(model, generator)
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    loader = make_loader(
        str(data_root),
        str(split_path),
        batch_size=args.batch_size,
        size=int(resolved.get("size", 256)),
        augment=False,
        keep_ratio_pad=False,
        workers=args.workers,
        shuffle=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=True,
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_eval_crop="resize",
        mllm_features_path=None,
    )

    prompt_accumulator = PromptMetricAccumulator()
    detector = DetectionAccumulator(float(resolved.get("pd_fa_dist", 3.0)))
    global_intersection = 0.0
    global_union = 0.0
    image_ious = []
    image_f1s = []
    image_rows = []
    latency_seconds = 0.0
    evaluated_images = 0
    zero_prompt_images = 0
    zero_prompt_target_images = 0
    prompt_budget = int(resolved.get("prompt_budget", 5))
    segmentation_threshold = float(resolved.get("segmentation_threshold", 0.5))
    checkpoint_epoch = int(checkpoint["epoch"])

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"mask-eval:{args.mode}")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True).unsqueeze(1).float()
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            encoded = model.get_image_embeddings(images)
            neck, interms = encoded[0], encoded[1]
            multi_scale = encoded[2] if generator_kind == "probe" else None
            if args.mode in {"A0", "A1"}:
                proposal = generator(images, neck, multi_scale)
            else:
                proposal, _, _, _ = multiview_propose(
                    images,
                    shared_generator,
                    views=views,
                    cluster_radius=args.cluster_radius,
                    max_candidates=int(resolved.get("candidate_k_raw", 32)),
                    gate="none" if args.mode == "A2" else "rule",
                    min_support=args.min_support,
                    max_dispersion=args.max_dispersion,
                    reliability_threshold=args.reliability_threshold,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    score_mode=args.score_mode,
                )
            prompt_input = "points" if args.mode in {"A0", "A2", "A3"} else str(resolved["prompt_input"])
            points, labels, dense = make_sam_inputs(
                model, proposal, prompt_input, prompt_budget
            )
            height, width = images.shape[-2:]
            predicted_masks, _ = model.predict_masks(
                neck,
                interms,
                points,
                labels,
                batched_masks=dense,
                text_sparse_embeddings=None,
                multimask_output=False,
                input_h=height,
                input_w=width,
                output_h=height,
                output_w=width,
                hq_token_only=bool(
                    int(resolved.get("hq_warmup_epochs", 30)) > 0
                    and checkpoint_epoch <= int(resolved.get("hq_warmup_epochs", 30))
                ),
            )
            logits = predicted_masks[:, 0, 0].unsqueeze(1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_seconds += time.perf_counter() - start

            prediction = (torch.sigmoid(logits) >= segmentation_threshold).float()
            prompt_accumulator.update(proposal, batch["mask"], list(batch["name"]))
            intersection = (prediction * targets).sum(dim=(1, 2, 3))
            union = (prediction + targets - prediction * targets).sum(dim=(1, 2, 3))
            global_intersection += float(intersection.sum())
            global_union += float(union.sum())
            per_image_iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
            denominator = prediction.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            per_image_f1 = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
            image_ious.extend(per_image_iou.tolist())
            image_f1s.extend(per_image_f1.tolist())
            valid_counts = proposal.candidate_valid[:, :prompt_budget].sum(dim=1)
            target_presence = targets.flatten(1).sum(dim=1) > 0
            zero_prompt_images += int((valid_counts == 0).sum())
            zero_prompt_target_images += int(((valid_counts == 0) & target_presence).sum())
            evaluated_images += int(images.shape[0])
            for index, name in enumerate(list(batch["name"])):
                detector.update(prediction[index, 0], targets[index, 0])
                image_rows.append(
                    {
                        "image": name,
                        "iou": float(per_image_iou[index]),
                        "f1": float(per_image_f1[index]),
                        "candidate_count": int(valid_counts[index]),
                        "target_present": int(target_presence[index]),
                    }
                )

    pd_value, fa_value = detector.finalize()
    prompt_result = prompt_accumulator.finalize()
    summary = {
        "mode": args.mode,
        "images": evaluated_images,
        "global_iou": global_intersection / global_union if global_union > 0 else 1.0,
        "mean_niou": float(np.mean(image_ious)) if image_ious else 0.0,
        "f1": float(np.mean(image_f1s)) if image_f1s else 0.0,
        "pd": pd_value,
        "fa": fa_value,
        "zero_prompt_fraction": zero_prompt_images / max(1, evaluated_images),
        "zero_prompt_target_fraction": zero_prompt_target_images / max(1, evaluated_images),
        "latency_ms_per_image": latency_seconds * 1000.0 / max(1, evaluated_images),
        "checkpoint_epoch": checkpoint_epoch,
        "segmentation_threshold": segmentation_threshold,
    }
    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "split_role": args.split_role,
        "split_sha256": sha256_file(split_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "probe_checkpoint_sha256": sha256_file(Path(probe_checkpoint).resolve()) if probe_checkpoint else None,
        "generator": generator_kind,
        "views": list(views) if args.mode in {"A2", "A3"} else ["identity"],
        "rule": {
            "min_support": args.min_support,
            "max_dispersion": args.max_dispersion,
            "reliability_threshold": args.reliability_threshold,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
        } if args.mode == "A3" else None,
        "score_mode": args.score_mode,
        "gt_boundary": "GT is used only after proposals and masks are finalized",
    }
    (output_dir / "mask_metrics_summary.json").write_text(
        json.dumps({"manifest": manifest, "metrics": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt_metrics_summary.json").write_text(
        json.dumps(prompt_result["summary"], ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "mask_metrics_per_image.csv", image_rows)
    write_csv(output_dir / "prompt_metrics_per_image.csv", prompt_result["per_image_rows"])
    write_csv(output_dir / "prompt_metrics_per_component.csv", prompt_result["per_component_rows"])
    write_csv(output_dir / "candidate_budget_curve.csv", prompt_result["budget_rows"])
    write_csv(output_dir / "area_bin_metrics.csv", prompt_result["area_rows"])
    print(json.dumps({"manifest": manifest, "metrics": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
