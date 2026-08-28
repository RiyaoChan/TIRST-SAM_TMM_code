#!/usr/bin/env python3
"""Unified 100-epoch trainer for C0/C1/F1/F2 full MicroQuery experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery_end2end import EndToEndMicroQueryHead, microquery_full_loss
from efficient_sam.microquery_gate_deployment import GateDeploymentConfig
from scripts.microquery_end2end_dataset import (
    MicroQueryEndToEndDataset,
    candidate_class_weights,
)
from scripts.microquery_end2end_metrics import FullMaskMetricAccumulator
from scripts.microquery_end2end_runtime import (
    VARIANTS,
    build_full_sam,
    checkpoint_state,
    forward_deployable,
    load_checkpoint_state,
    trainable_parameter_counts,
)


DEFAULT_DATA_ROOT = r"E:\code\SIRST-5K-main\SIRST-5K-main\dataset\IRSTD-1k"
DEFAULT_A1 = "outputs/experiment1_a1/IRSTD-1k/neck_points_100ep_dual_ckpt_seed20260825/best_mask.pt"
DEFAULT_WEIGHTS = "weights/efficient_sam_vitt.pt"
DEFAULT_TRAIN_CACHE = "outputs/microquery/P0_candidate_cache/IRSTD-1k/a1_best_mask_train/candidates.npz"
DEFAULT_VAL_CACHE = "outputs/microquery/P0_candidate_cache/IRSTD-1k/a1_best_mask_val/candidates.npz"
DEFAULT_SHARED_INIT = "outputs/microquery/end2end_full/IRSTD-1k/shared_init/microquery_head_init.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="f2_gate_token")
    parser.add_argument("--make_shared_init", action="store_true")
    parser.add_argument("--shared_head_init", default=DEFAULT_SHARED_INIT)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train_split", default="splits/experiment1_seed20260825/train.txt")
    parser.add_argument("--val_split", default="splits/experiment1_seed20260825/val.txt")
    parser.add_argument("--train_candidate_cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val_candidate_cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--a1_checkpoint", default=DEFAULT_A1)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--query_chunk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bfloat16", "off"), default="bfloat16")
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--prompt_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_lr", type=float, default=2e-5)
    parser.add_argument("--head_weight_decay", type=float, default=1e-4)
    parser.add_argument("--sam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--segmentation_threshold", type=float, default=0.5)
    parser.add_argument("--smoke_batches", type=int, default=0)
    parser.add_argument("--smoke_val_batches", type=int, default=0)
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def create_shared_initialization(path: Path, seed: int) -> None:
    set_deterministic(seed)
    head = EndToEndMicroQueryHead(input_dim=451, hidden_dim=256, dropout=0.1)
    parameters = sum(value.numel() for value in head.parameters())
    if parameters > 500_000:
        raise RuntimeError(f"MicroQuery head exceeds 0.5M parameter cap: {parameters}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "seed": int(seed),
            "head_config": {"input_dim": 451, "hidden_dim": 256, "dropout": 0.1},
            "trainable_parameters": parameters,
            "head_state": head.state_dict(),
            "source": "fresh random initialization; no old M2/M2-S1b pretraining",
        },
        path,
    )
    print(json.dumps({"shared_head_init": str(path), "sha256": sha256_file(path), "parameters": parameters}))


def to_device(batch: dict, device: torch.device) -> tuple[dict, dict, dict]:
    deployable = {
        key: value.to(device, non_blocking=True)
        for key, value in batch["deployable"].items()
    }
    supervision = {
        key: value.to(device, non_blocking=True)
        for key, value in batch["supervision"].items()
    }
    return deployable, supervision, batch["meta"]


def lr_for_epoch(base: float, minimum: float, epoch: int, epochs: int, warmup: int) -> float:
    if epoch <= warmup:
        return float(base) * float(epoch) / float(max(1, warmup))
    progress = float(epoch - warmup) / float(max(1, epochs - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return float(minimum) + (float(base) - float(minimum)) * cosine


def set_epoch_lrs(optimizer, variant: str, epoch: int, epochs: int, warmup: int) -> dict[str, float]:
    bases = {"head": (3e-4, 3e-6), "prompt": (1e-5, 1e-6), "decoder": (2e-5, 1e-6)}
    values = {}
    for group in optimizer.param_groups:
        name = group["name"]
        base, minimum = bases[name]
        value = lr_for_epoch(base, minimum, epoch, epochs, warmup)
        group["lr"] = value
        values[name] = value
    return values


def make_optimizer(model, head, args):
    groups = []
    if head is not None:
        groups.append(
            {
                "name": "head",
                "params": list(head.parameters()),
                "lr": args.head_lr,
                "weight_decay": args.head_weight_decay,
            }
        )
    groups.extend(
        [
            {
                "name": "prompt",
                "params": list(model.prompt_encoder.parameters()),
                "lr": args.prompt_lr,
                "weight_decay": args.sam_weight_decay,
            },
            {
                "name": "decoder",
                "params": list(model.mask_decoder.parameters()),
                "lr": args.decoder_lr,
                "weight_decay": args.sam_weight_decay,
            },
        ]
    )
    return torch.optim.AdamW(groups)


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type == "cuda" and amp_dtype == "bfloat16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def checkpoint_payload(model, head, args, epoch: int, metrics: dict, resolved: dict) -> dict:
    return {
        "schema_version": 1,
        "variant": args.variant,
        "epoch": int(epoch),
        "seed": int(args.seed),
        "metrics": json_safe(metrics),
        "state": checkpoint_state(model, head),
        "resolved_config": resolved,
    }


def gradient_audit(model, head) -> dict:
    def summarize(parameters):
        gradients = [p.grad.detach().float().abs().sum() for p in parameters if p.grad is not None]
        return {
            "parameters_with_grad": len(gradients),
            "absolute_gradient_sum": float(torch.stack(gradients).sum()) if gradients else 0.0,
        }

    output = {
        "image_encoder": summarize(model.image_encoder.parameters()),
        "prompt_encoder": summarize(model.prompt_encoder.parameters()),
        "mask_decoder": summarize(model.mask_decoder.parameters()),
    }
    if head is not None:
        output["microquery_head"] = summarize(head.parameters())
        output["object_head"] = summarize(head.object_head.parameters())
        output["candidate_token_norm"] = summarize(head.token_norm.parameters())
        output["token_scale"] = summarize([head.token_scale])
    return output


@torch.no_grad()
def validate(model, head, loader, device, args, epoch: int) -> tuple[dict, list[dict]]:
    model.eval()
    model.image_encoder.eval()
    if head is not None:
        head.eval()
    accumulator = FullMaskMetricAccumulator(args.segmentation_threshold, 3.0)
    multi_matches = 0
    gate_values: list[torch.Tensor] = []
    token_norms: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if args.smoke_val_batches and batch_index >= args.smoke_val_batches:
            break
        deployable, supervision, meta = to_device(batch, device)
        with autocast_context(device, args.amp_dtype):
            output = forward_deployable(
                model,
                head,
                deployable,
                variant=args.variant,
                gate_deployment_config=(
                    GateDeploymentConfig("all_one")
                    if args.variant == "c1_independent_aux"
                    else GateDeploymentConfig("legacy_checkpoint_schedule")
                ),
                checkpoint_epoch=epoch,
                query_chunk=args.query_chunk,
            )
        object_scores = output.raw_gates if output.object_logits is not None else None
        accumulator.update(
            names=list(meta["name"]),
            probabilities=output.final_probability,
            targets=supervision["full_mask"],
            candidate_valid=deployable["candidate_valid"],
            candidate_scores=deployable["candidate_scores"],
            semantic_labels=supervision["semantic_labels"],
            object_scores=object_scores,
        )
        multi_matches += int(supervision["multi_match_count"].sum())
        if args.variant == "c0_one_query":
            gate_values.append(output.effective_gates.reshape(-1).detach().float().cpu())
        else:
            gate_values.append(output.effective_gates[deployable["candidate_valid"]].detach().float().cpu())
        if output.candidate_tokens is not None:
            token_norms.append(
                output.candidate_tokens[deployable["candidate_valid"]].detach().float().norm(dim=-1).cpu()
            )
    metrics = accumulator.finalize()
    gates = torch.cat(gate_values) if gate_values and any(row.numel() for row in gate_values) else torch.zeros(0)
    tokens = torch.cat(token_norms) if token_norms and any(row.numel() for row in token_norms) else torch.zeros(0)
    metrics.update(
        {
            "assignment_multi_matches": multi_matches,
            "gate_mean": float(gates.mean()) if gates.numel() else 0.0,
            "gate_min": float(gates.min()) if gates.numel() else 0.0,
            "gate_max": float(gates.max()) if gates.numel() else 0.0,
            "token_norm_mean": float(tokens.mean()) if tokens.numel() else 0.0,
            "threshold": float(args.segmentation_threshold),
        }
    )
    return metrics, accumulator.per_image


def main() -> None:
    args = parse_args()
    shared_init = resolve_repo_path(args.shared_head_init)
    if args.make_shared_init:
        create_shared_initialization(shared_init, args.seed)
        return
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("epochs, batch_size and gradient_accumulation must be positive")
    if args.budget != 10:
        raise ValueError("formal end-to-end protocol fixes K=10")
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    data_root = resolve_repo_path(args.data_root)
    train_split = resolve_repo_path(args.train_split) if Path(args.train_split).is_absolute() else data_root / args.train_split
    val_split = resolve_repo_path(args.val_split) if Path(args.val_split).is_absolute() else data_root / args.val_split
    train_cache = resolve_repo_path(args.train_candidate_cache)
    val_cache = resolve_repo_path(args.val_candidate_cache)
    a1_checkpoint = resolve_repo_path(args.a1_checkpoint)
    weights = resolve_repo_path(args.weights)
    if args.output_dir:
        output_dir = resolve_repo_path(args.output_dir)
    else:
        run_names = {
            "c0_one_query": "C0_one_query",
            "c1_independent_aux": "C1_independent_aux",
            "f1_soft_gate": "F1_soft_gate",
            "f2_gate_token": "F2_gate_token",
        }
        output_dir = REPO_ROOT / "outputs/microquery/end2end_full/IRSTD-1k" / run_names[args.variant]
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MicroQueryEndToEndDataset(
        data_root=data_root,
        split=train_split,
        candidate_cache=train_cache,
        size=args.size,
        budget=args.budget,
        augment=True,
        seed=args.seed,
    )
    val_dataset = MicroQueryEndToEndDataset(
        data_root=data_root,
        split=val_split,
        candidate_cache=val_cache,
        size=args.size,
        budget=args.budget,
        augment=False,
        seed=args.seed,
    )
    if len(train_dataset) != 720 or len(val_dataset) != 80:
        raise RuntimeError(f"strict IRSTD split requires 720/80, got {len(train_dataset)}/{len(val_dataset)}")
    class_weights, class_counts = candidate_class_weights(train_dataset)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model, _ = build_full_sam(a1_checkpoint, weights, device)
    head = None
    if args.variant != "c0_one_query":
        if not shared_init.is_file():
            raise FileNotFoundError(f"shared head initialization missing: {shared_init}")
        shared = torch.load(shared_init, map_location="cpu", weights_only=False)
        head = EndToEndMicroQueryHead(**shared["head_config"]).to(device)
        head.load_state_dict(shared["head_state"], strict=True)
        if sum(p.numel() for p in head.parameters()) > 500_000:
            raise RuntimeError("MicroQuery head exceeds 0.5M parameter cap")
    counts = trainable_parameter_counts(model, head)
    if counts["image_encoder"] != 0:
        raise RuntimeError("image encoder is unexpectedly trainable")
    optimizer = make_optimizer(model, head, args)
    class_weights = class_weights.to(device)
    resolved = {
        **vars(args),
        "data_root": str(data_root),
        "train_split": str(train_split),
        "val_split": str(val_split),
        "train_candidate_cache": str(train_cache),
        "val_candidate_cache": str(val_cache),
        "a1_checkpoint": str(a1_checkpoint),
        "weights": str(weights),
        "shared_head_init": None if head is None else str(shared_init),
        "output_dir": str(output_dir),
        "device_resolved": str(device),
        "train_split_sha256": sha256_file(train_split),
        "val_split_sha256": sha256_file(val_split),
        "train_candidate_cache_sha256": sha256_file(train_cache),
        "val_candidate_cache_sha256": sha256_file(val_cache),
        "a1_checkpoint_sha256": sha256_file(a1_checkpoint),
        "weights_sha256": sha256_file(weights),
        "shared_head_init_sha256": None if head is None else sha256_file(shared_init),
        "candidate_class_counts": class_counts,
        "candidate_class_weights": class_weights.detach().cpu().tolist(),
        "parameter_counts": counts,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "augmentation": {"horizontal_flip_probability": 0.5, "vertical_flip_probability": 0.5},
        "optional_modules": "all off",
        "text_embeddings": None,
        "test_split_read": False,
        "gt_boundary": "GT is used only after forward for assignment/loss/metrics; forward_deployable accepts deployable inputs only.",
        "git_commit_at_start": git_commit(),
        "command_argv": [sys.executable, *sys.argv],
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(json_safe(resolved), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    history_path = output_dir / "train_history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best_iou = -math.inf
    best_auprc = -math.inf
    best_iou_epoch = -1
    best_auprc_epoch = -1
    first_gradient_audit = None
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        model.image_encoder.eval()
        if head is not None:
            head.train()
        lr_values = set_epoch_lrs(optimizer, args.variant, epoch, args.epochs, args.warmup_epochs)
        optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        batches = 0
        gate_means = []
        for batch_index, batch in enumerate(train_loader):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            deployable, supervision, _ = to_device(batch, device)
            with autocast_context(device, args.amp_dtype):
                output = forward_deployable(
                    model,
                    head,
                    deployable,
                    variant=args.variant,
                    training_epoch=epoch,
                    query_chunk=args.query_chunk,
                )
            loss_arguments = {
                "variant": args.variant,
                "final_probability": output.final_probability.float(),
                "full_target": supervision["full_mask"].float(),
                "covered_target": supervision["covered_mask"].float(),
            }
            if args.variant != "c0_one_query":
                loss_arguments.update(
                    {
                        "query_logits": output.query_logits.float(),
                        "query_targets": supervision["query_targets"].float(),
                        "semantic_labels": supervision["semantic_labels"],
                        "candidate_valid": deployable["candidate_valid"],
                        "object_logits": output.object_logits.float(),
                        "raw_gates": output.raw_gates.float(),
                        "component_ids": supervision["component_ids"],
                        "iou_predictions": output.iou_predictions.float(),
                        "class_weights": class_weights,
                        "candidate_tokens": output.candidate_tokens.float(),
                    }
                )
            losses = microquery_full_loss(**loss_arguments)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"non-finite loss at epoch={epoch} batch={batch_index}")
            (losses["total"] / args.gradient_accumulation).backward()
            if first_gradient_audit is None:
                first_gradient_audit = gradient_audit(model, head)
            should_step = (batch_index + 1) % args.gradient_accumulation == 0
            if should_step:
                trainable = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            valid_gates = (
                output.effective_gates.reshape(-1)
                if args.variant == "c0_one_query"
                else output.effective_gates[deployable["candidate_valid"]]
            )
            if valid_gates.numel():
                gate_means.append(float(valid_gates.detach().float().mean()))
            batches += 1
        if batches == 0:
            raise RuntimeError("training loader produced no batches")
        if batches % args.gradient_accumulation:
            trainable = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        validation, _ = validate(model, head, val_loader, device, args, epoch)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / batches for key, value in totals.items()},
            "train_gate_mean": float(np.mean(gate_means)) if gate_means else 0.0,
            **{f"val_{key}": value for key, value in validation.items()},
            **{f"lr_{key}": value for key, value in lr_values.items()},
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        print(json.dumps(json_safe(row), ensure_ascii=False), flush=True)
        payload = checkpoint_payload(model, head, args, epoch, validation, resolved)
        torch.save(payload, output_dir / "last.pt")
        if float(validation["global_iou"]) > best_iou:
            best_iou = float(validation["global_iou"])
            best_iou_epoch = epoch
            torch.save(payload, output_dir / "best_fixed05_global_iou.pt")
        auprc = float(validation["mask_auprc"])
        if math.isfinite(auprc) and auprc > best_auprc:
            best_auprc = auprc
            best_auprc_epoch = epoch
            torch.save(payload, output_dir / "best_mask_auprc.pt")
        if epoch == 20:
            reload_test = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
            if int(reload_test["epoch"]) != 20:
                raise RuntimeError("epoch-20 checkpoint reload audit failed")
    if first_gradient_audit is None:
        raise RuntimeError("gradient audit was not collected")
    (output_dir / "gradient_audit.json").write_text(
        json.dumps(json_safe(first_gradient_audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    best_checkpoint = torch.load(
        output_dir / "best_fixed05_global_iou.pt", map_location="cpu", weights_only=False
    )
    load_checkpoint_state(model, head, best_checkpoint["state"])
    fixed_metrics, per_image = validate(
        model, head, val_loader, device, args, int(best_checkpoint["epoch"])
    )
    (output_dir / "fixed05_metrics.json").write_text(
        json.dumps(json_safe(fixed_metrics), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "MicroQuery end-to-end full training",
        "dataset": "IRSTD-1k",
        "split_role": "validation",
        "variant": args.variant,
        "seed": args.seed,
        "epochs_completed": args.epochs,
        "formal_100_epoch_run": bool(args.epochs == 100 and not args.smoke_batches and not args.smoke_val_batches),
        "best_fixed05_global_iou_epoch": best_iou_epoch,
        "best_mask_auprc_epoch": best_auprc_epoch,
        "training_seconds": time.perf_counter() - start_time,
        "test_split_read": False,
        "image_encoder_frozen": True,
        "candidate_probe_frozen_cached_coordinates": True,
        "optional_modules_off": True,
        "checkpoint_selection_threshold": 0.5,
        "fixed05_metrics": json_safe(fixed_metrics),
        "git_commit_at_end": git_commit(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
