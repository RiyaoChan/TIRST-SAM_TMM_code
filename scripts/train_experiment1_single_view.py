#!/usr/bin/env python3
"""Train the strict image-only A1 point/dense prompt screening variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.prompt_metrics import PromptMetricAccumulator
from efficient_sam.prompt_proposal import (
    DenseHeadProposalAdapter,
    DoGLoGProposalAdapter,
    PGAPProposalAdapter,
    PromptProposal,
)
from efficient_sam.prompt_training import SpatialProbeHead
from scripts.eval_prompt_quality import select_probe_features
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


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


class DetectionAccumulator:
    def __init__(self, distance_threshold: float = 3.0):
        self.distance_threshold = float(distance_threshold)
        self.targets = 0
        self.matches = 0
        self.false_pixels = 0
        self.pixels = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction_np = prediction.detach().cpu().numpy().astype(np.uint8)
        target_np = target.detach().cpu().numpy().astype(np.uint8)
        predicted = list(measure.regionprops(measure.label(prediction_np, connectivity=2)))
        targets = list(measure.regionprops(measure.label(target_np, connectivity=2)))
        self.targets += len(targets)
        self.pixels += int(target_np.size)
        for target_region in targets:
            target_center = np.asarray(target_region.centroid)
            best = None
            for index, predicted_region in enumerate(predicted):
                distance = float(np.linalg.norm(np.asarray(predicted_region.centroid) - target_center))
                if distance < self.distance_threshold and (best is None or distance < best[0]):
                    best = (distance, index)
            if best is not None:
                self.matches += 1
                predicted.pop(best[1])
        self.false_pixels += sum(int(region.area) for region in predicted)

    def finalize(self) -> tuple[float, float]:
        pd = self.matches / max(1, self.targets)
        fa = self.false_pixels / max(1, self.pixels)
        return float(pd), float(fa)


def empty_points(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((batch_size, 1, 1, 2), device=device, dtype=torch.float32),
        -torch.ones((batch_size, 1, 1), device=device, dtype=torch.int64),
    )


class ImageOnlyProposalGenerator:
    """Proposal generation deliberately has no GT/mask argument."""

    def __init__(self, args, device: torch.device):
        self.kind = args.generator
        self.args_candidate_k_raw = int(args.candidate_k_raw)
        self.level = None
        self.head = None
        if self.kind == "null":
            self.adapter = None
        elif self.kind == "pgap":
            self.adapter = PGAPProposalAdapter(
                candidate_k_raw=args.candidate_k_raw,
                nms_radius=args.nms_radius,
                score_threshold=args.score_threshold,
            ).to(device)
        elif self.kind == "doglog":
            self.adapter = DoGLoGProposalAdapter(
                candidate_k_raw=args.candidate_k_raw,
                nms_radius=args.nms_radius,
                score_threshold=args.score_threshold,
            ).to(device)
        elif self.kind == "probe":
            checkpoint = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
            config = dict(checkpoint["head_config"])
            self.level = str(checkpoint["level"])
            self.head = SpatialProbeHead(
                int(config["in_channels"]), int(config.get("hidden_channels", 64))
            ).to(device)
            self.head.load_state_dict(checkpoint["head_state"], strict=True)
            self.head.eval()
            for parameter in self.head.parameters():
                parameter.requires_grad_(False)
            self.adapter = DenseHeadProposalAdapter(
                self.head,
                candidate_k_raw=args.candidate_k_raw,
                nms_radius=args.nms_radius,
                score_threshold=args.score_threshold,
            )
        else:
            raise ValueError(self.kind)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        neck: torch.Tensor,
        multi_scale: list[torch.Tensor] | None,
    ):
        if self.kind == "null":
            batch = int(images.shape[0])
            candidate_k_raw = int(self.args_candidate_k_raw)
            return PromptProposal(
                dense_logits=None,
                dense_probs=None,
                candidate_xy=images.new_zeros((batch, candidate_k_raw, 2)),
                candidate_scores=images.new_zeros((batch, candidate_k_raw)),
                candidate_valid=torch.zeros(
                    (batch, candidate_k_raw), dtype=torch.bool, device=images.device
                ),
                candidate_source=[["null"] * candidate_k_raw for _ in range(batch)],
            ).validate()
        if self.kind in {"pgap", "doglog"}:
            return self.adapter(images)
        if multi_scale is None:
            raise RuntimeError("Probe proposal requires multi-scale encoder outputs")
        features = select_probe_features(self.level, neck, multi_scale)
        return self.adapter(features.detach(), output_size=images.shape[-2:])


def make_sam_inputs(model, proposal, prompt_input: str, budget: int):
    batch = proposal.candidate_xy.shape[0]
    device = proposal.candidate_xy.device
    if prompt_input in {"points", "dense_points"}:
        coords, labels = proposal.to_point_prompts()
        coords = coords[:, :, :budget]
        labels = labels[:, :, :budget]
    else:
        coords, labels = empty_points(batch, device)
    dense = None
    if prompt_input in {"dense", "dense_points"}:
        if proposal.dense_probs is None:
            raise RuntimeError("Dense prompt variant requires dense_probs")
        target_size = getattr(model.prompt_encoder, "mask_input_size", proposal.dense_probs.shape[-2:])
        dense = F.interpolate(
            proposal.dense_probs.detach(), size=target_size, mode="bilinear", align_corners=False
        )
    return coords, labels, dense


def set_trainable_stage(model, encoder_trainable: bool, prompt_encoder_trainable: bool) -> None:
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad_(encoder_trainable)
    for parameter in model.prompt_encoder.parameters():
        parameter.requires_grad_(prompt_encoder_trainable)
    for parameter in model.mask_decoder.parameters():
        parameter.requires_grad_(True)


def forward_model(model, images, proposal_generator, args, epoch: int):
    encoded = model.get_image_embeddings(images)
    neck, interms = encoded[0], encoded[1]
    multi_scale = encoded[2] if args.generator == "probe" else None
    proposal = proposal_generator(images, neck, multi_scale)
    points, labels, dense = make_sam_inputs(model, proposal, args.prompt_input, args.prompt_budget)
    height, width = images.shape[-2:]
    masks, _ = model.predict_masks(
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
        hq_token_only=bool(args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs),
    )
    return masks[:, 0, 0].unsqueeze(1), proposal


@torch.no_grad()
def validate(model, generator, loader, device, args, epoch: int) -> dict:
    model.eval()
    prompt_accumulator = PromptMetricAccumulator()
    global_intersection = 0.0
    global_union = 0.0
    image_ious = []
    image_f1s = []
    detector = DetectionAccumulator(args.pd_fa_dist)
    zero_prompt_target_images = 0
    zero_prompt_images = 0
    evaluated_images = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True).unsqueeze(1).float()
        logits, proposal = forward_model(model, images, generator, args, epoch)
        prompt_accumulator.update(proposal, batch["mask"], list(batch["name"]))
        prediction = (torch.sigmoid(logits) >= args.segmentation_threshold).float()
        intersection = (prediction * targets).sum(dim=(1, 2, 3))
        union = (prediction + targets - prediction * targets).sum(dim=(1, 2, 3))
        global_intersection += float(intersection.sum())
        global_union += float(union.sum())
        image_ious.extend(torch.where(union > 0, intersection / union, torch.ones_like(union)).tolist())
        precision_denominator = prediction.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        image_f1s.extend(((2.0 * intersection + 1e-6) / (precision_denominator + 1e-6)).tolist())
        valid_counts = proposal.candidate_valid[:, : args.prompt_budget].sum(dim=1)
        target_presence = targets.flatten(1).sum(dim=1) > 0
        zero_prompt_images += int((valid_counts == 0).sum())
        zero_prompt_target_images += int(((valid_counts == 0) & target_presence).sum())
        evaluated_images += int(images.shape[0])
        for index in range(prediction.shape[0]):
            detector.update(prediction[index, 0], targets[index, 0])
    pd_value, fa_value = detector.finalize()
    prompt_result = prompt_accumulator.finalize()
    budget5 = next(row for row in prompt_result["budget_rows"] if int(row["budget"]) == 5)
    return {
        "global_iou": global_intersection / global_union if global_union > 0 else 1.0,
        "mean_niou": float(np.mean(image_ious)) if image_ious else 0.0,
        "f1": float(np.mean(image_f1s)) if image_f1s else 0.0,
        "pd": pd_value,
        "fa": fa_value,
        "zero_prompt_fraction": zero_prompt_images / max(1, evaluated_images),
        "zero_prompt_target_fraction": zero_prompt_target_images / max(1, evaluated_images),
        "prompt_component_recall_at_5": float(budget5["component_recall"]),
        "prompt_false_per_mp_at_5": float(budget5["false_prompts_per_million_pixels"]),
        "prompt_metrics": prompt_result,
    }


def compact_metrics(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "prompt_metrics"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_split", default="splits/experiment1_seed20260825/train.txt")
    parser.add_argument("--val_split", default="splits/experiment1_seed20260825/val.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--generator", choices=("null", "pgap", "doglog", "probe"), required=True)
    parser.add_argument("--probe_checkpoint", default=None)
    parser.add_argument("--prompt_input", choices=("points", "dense", "dense_points"), default="points")
    parser.add_argument("--prompt_budget", type=int, default=5)
    parser.add_argument("--candidate_k_raw", type=int, default=32)
    parser.add_argument("--nms_radius", type=float, default=3.0)
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=60)
    parser.add_argument("--hq_warmup_epochs", type=int, default=30)
    parser.add_argument("--segmentation_threshold", type=float, default=0.5)
    parser.add_argument("--pd_fa_dist", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--amp_dtype", choices=("off", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--smoke_batches", type=int, default=0)
    parser.add_argument("--smoke_val_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generator == "probe" and not args.probe_checkpoint:
        raise ValueError("--generator probe requires --probe_checkpoint")
    if args.prompt_budget <= 0 or args.prompt_budget > args.candidate_k_raw:
        raise ValueError("prompt_budget must be in [1, candidate_k_raw]")
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    train_split = data_root / args.train_split if not Path(args.train_split).is_absolute() else Path(args.train_split)
    val_split = data_root / args.val_split if not Path(args.val_split).is_absolute() else Path(args.val_split)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights).resolve()

    resolved = vars(args).copy()
    resolved.update(
        {
            "device": str(device),
            "train_split_sha256": sha256_file(train_split),
            "val_split_sha256": sha256_file(val_split),
            "weights_sha256": sha256_file(weights),
            "self_prompt_gt_policy": "none",
            "text_features": None,
            "checkpoint_selection": "validation global IoU; fixed segmentation threshold",
        }
    )
    if args.probe_checkpoint:
        resolved["probe_checkpoint_sha256"] = sha256_file(Path(args.probe_checkpoint).resolve())
    (output_dir / "resolved_args.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    common_loader = dict(
        root=str(data_root),
        size=args.size,
        keep_ratio_pad=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=True,
        sc_use_noise=False,
        mllm_features_path=None,
    )
    train_loader = make_loader(
        split_txt=str(train_split),
        batch_size=args.batch_size,
        augment=True,
        shuffle=True,
        workers=args.workers,
        sc_use_gamma=True,
        sc_pos_prob=0.5,
        sc_eval_crop="random",
        **common_loader,
    )
    val_loader = make_loader(
        split_txt=str(val_split),
        batch_size=max(1, args.batch_size * 2),
        augment=False,
        shuffle=False,
        workers=args.workers,
        sc_use_gamma=False,
        sc_pos_prob=0.5,
        sc_eval_crop="resize",
        **common_loader,
    )

    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(weights),
        use_adapter=False,
        return_encoder_multi_scale=args.generator == "probe",
    ).to(device)
    generator = ImageOnlyProposalGenerator(args, device)
    set_trainable_stage(model, encoder_trainable=False, prompt_encoder_trainable=False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": args.lr_encoder},
            {"params": model.prompt_encoder.parameters(), "lr": args.lr_head},
            {"params": model.mask_decoder.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = device.type == "cuda" and args.amp_dtype != "off"
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and args.amp_dtype == "float16"
    )
    best_iou = -math.inf
    best_epoch = -1
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        encoder_trainable = epoch > args.freeze_encoder_epochs
        set_trainable_stage(model, encoder_trainable, encoder_trainable)
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_index, batch in enumerate(tqdm(train_loader, desc=f"A1:{epoch:03d}")):
            if args.smoke_batches > 0 and batch_index >= args.smoke_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True).unsqueeze(1).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits, _ = forward_model(model, images, generator, args, epoch)
                loss = segmentation_loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite A1 loss at epoch={epoch}, batch={batch_index}; "
                    f"amp_dtype={args.amp_dtype}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            steps += 1
        scheduler.step()

        record = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "train_loss": total_loss / max(1, steps),
            "encoder_trainable": encoder_trainable,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        if epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            if args.smoke_val_batches > 0:
                original_loader = val_loader
                val_loader = list(val_loader)[: args.smoke_val_batches]
                metrics = validate(model, generator, val_loader, device, args, epoch)
                val_loader = original_loader
            else:
                metrics = validate(model, generator, val_loader, device, args, epoch)
            record["val"] = compact_metrics(metrics)
            if float(metrics["global_iou"]) > best_iou:
                best_iou = float(metrics["global_iou"])
                best_epoch = epoch
                best_metrics = copy.deepcopy(compact_metrics(metrics))
                torch.save(
                    {
                        "schema_version": 1,
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "metrics": best_metrics,
                        "resolved_args": resolved,
                    },
                    output_dir / "best_mask.pt",
                )
                prompt_result = metrics["prompt_metrics"]
                (output_dir / "best_prompt_metrics.json").write_text(
                    json.dumps(prompt_result["summary"], ensure_ascii=False, indent=2, allow_nan=True) + "\n",
                    encoding="utf-8",
                )
        history.append(record)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
        print(json.dumps(record, ensure_ascii=False, allow_nan=True), flush=True)

    summary = {
        "completed_epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "selection_metric": "validation global IoU",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
