#!/usr/bin/env python3
"""Train early/mid/neck targetness probes with one frozen encoder pass."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
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
from efficient_sam.prompt_metrics import PromptMetricAccumulator
from efficient_sam.prompt_proposal import DenseHeadProposalAdapter
from efficient_sam.prompt_training import SpatialProbeHead, prompt_probe_loss
from sirst_dataset import make_loader


LEVEL_CHANNELS = {"early": 192, "mid": 192, "neck": 256}


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


def select_features(level: str, neck: torch.Tensor, multi_scale: list[torch.Tensor]) -> torch.Tensor:
    if level == "early":
        return multi_scale[0]
    if level == "mid":
        return multi_scale[2]
    if level == "neck":
        return neck
    raise ValueError(level)


def find_budget(result: dict, budget: int) -> dict:
    return next(row for row in result["budget_rows"] if int(row["budget"]) == int(budget))


def find_tiny(result: dict, budget: int) -> dict:
    return next(
        row
        for row in result["area_rows"]
        if row["area_bin"] == "1-9" and int(row["budget"]) == int(budget)
    )


def selection_key(result: dict) -> tuple[float, float, float, float]:
    at20 = find_budget(result, 20)
    tiny20 = find_tiny(result, 20)
    return (
        float(tiny20["component_recall"]),
        float(at20["component_recall"]),
        -float(at20["false_prompts_per_million_pixels"]),
        float(result["summary"]["dense_prompt_auprc"]),
    )


def compact_metrics(result: dict) -> dict:
    at5 = find_budget(result, 5)
    at20 = find_budget(result, 20)
    tiny20 = find_tiny(result, 20)
    return {
        "component_recall_at_5": float(at5["component_recall"]),
        "component_recall_at_20": float(at20["component_recall"]),
        "tiny_recall_at_20": float(tiny20["component_recall"]),
        "prompt_precision_at_5": float(at5["prompt_precision"]),
        "false_prompts_per_million_at_20": float(at20["false_prompts_per_million_pixels"]),
        "dense_prompt_auprc": float(result["summary"]["dense_prompt_auprc"]),
        "candidate_score_auprc": float(result["summary"]["candidate_score_auprc"]),
        "zero_prompt_fraction": float(at20["zero_prompt_fraction"]),
    }


def evaluate(
    model,
    heads: dict[str, SpatialProbeHead],
    loader,
    device: torch.device,
    size: int,
    candidate_k_raw: int,
    nms_radius: float,
    score_threshold: float,
) -> dict[str, dict]:
    model.eval()
    for head in heads.values():
        head.eval()
    accumulators = {level: PromptMetricAccumulator() for level in heads}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            encoded = model.get_image_embeddings(images)
            neck, multi_scale = encoded[0], encoded[2]
            for level, head in heads.items():
                adapter = DenseHeadProposalAdapter(
                    head,
                    candidate_k_raw=candidate_k_raw,
                    nms_radius=nms_radius,
                    score_threshold=score_threshold,
                )
                proposal = adapter(
                    select_features(level, neck, multi_scale),
                    output_size=(size, size),
                )
                accumulators[level].update(proposal, batch["mask"], list(batch["name"]))
    return {level: accumulator.finalize() for level, accumulator in accumulators.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_split", default="splits/experiment1_seed20260825/train.txt")
    parser.add_argument("--val_split", default="splits/experiment1_seed20260825/val.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--levels", default="early,mid,neck")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--center_weight", type=float, default=1.0)
    parser.add_argument("--foreground_weight", type=float, default=0.5)
    parser.add_argument("--component_weight", type=float, default=0.5)
    parser.add_argument("--candidate_k_raw", type=int, default=32)
    parser.add_argument("--nms_radius", type=float, default=3.0)
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--smoke_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = tuple(item.strip() for item in args.levels.split(",") if item.strip())
    if not levels or any(level not in LEVEL_CHANNELS for level in levels):
        raise ValueError(f"levels must be selected from {sorted(LEVEL_CHANNELS)}")
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
            "levels": list(levels),
            "device": str(device),
            "train_split_sha256": sha256_file(train_split),
            "val_split_sha256": sha256_file(val_split),
            "weights_sha256": sha256_file(weights),
            "gt_policy": "GT supervises losses only; proposal samplers never receive masks",
            "checkpoint_selection": "lexicographic tiny Recall@20, overall Recall@20, lower False Prompts/MP, Dense AUPRC",
        }
    )
    (output_dir / "resolved_args.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    train_loader = make_loader(
        str(data_root),
        str(train_split),
        size=args.size,
        batch_size=args.batch_size,
        augment=True,
        shuffle=True,
        workers=args.workers,
        keep_ratio_pad=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=True,
        sc_use_noise=False,
        sc_use_gamma=True,
        sc_pos_prob=0.5,
        sc_eval_crop="random",
        mllm_features_path=None,
    )
    val_loader = make_loader(
        str(data_root),
        str(val_split),
        size=args.size,
        batch_size=max(1, args.batch_size * 2),
        augment=False,
        shuffle=False,
        workers=args.workers,
        keep_ratio_pad=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=True,
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_eval_crop="resize",
        mllm_features_path=None,
    )

    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(weights),
        use_adapter=False,
        return_encoder_multi_scale=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    heads = {
        level: SpatialProbeHead(LEVEL_CHANNELS[level], hidden_channels=args.hidden_channels).to(device)
        for level in levels
    }
    optimizer = torch.optim.AdamW(
        [parameter for head in heads.values() for parameter in head.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_keys = {level: None for level in levels}
    best_metrics = {level: None for level in levels}
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for head in heads.values():
            head.train()
        sums = {level: 0.0 for level in levels}
        steps = 0
        iterator = tqdm(train_loader, desc=f"probe-train:{epoch:03d}")
        for batch_index, batch in enumerate(iterator):
            if args.smoke_batches > 0 and batch_index >= args.smoke_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=use_amp):
                encoded = model.get_image_embeddings(images)
                neck, multi_scale = encoded[0], encoded[2]
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                for level, head in heads.items():
                    logits = head(
                        select_features(level, neck, multi_scale).detach(),
                        output_size=(args.size, args.size),
                    )
                    losses = prompt_probe_loss(
                        logits,
                        masks,
                        center_weight=args.center_weight,
                        foreground_weight=args.foreground_weight,
                        component_weight=args.component_weight,
                    )
                    total_loss = total_loss + losses["total"]
                    sums[level] += float(losses["total"].detach().item())
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": {level: sums[level] / max(1, steps) for level in levels},
        }
        if epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            results = evaluate(
                model,
                heads,
                val_loader,
                device,
                args.size,
                args.candidate_k_raw,
                args.nms_radius,
                args.score_threshold,
            )
            epoch_record["val"] = {}
            for level, result in results.items():
                metrics = compact_metrics(result)
                key = selection_key(result)
                epoch_record["val"][level] = metrics
                if best_keys[level] is None or key > best_keys[level]:
                    best_keys[level] = key
                    best_metrics[level] = metrics
                    checkpoint = {
                        "schema_version": 1,
                        "epoch": epoch,
                        "level": level,
                        "head_state": copy.deepcopy(heads[level].state_dict()),
                        "head_config": {
                            "in_channels": LEVEL_CHANNELS[level],
                            "hidden_channels": args.hidden_channels,
                        },
                        "metrics": metrics,
                        "selection_key": list(key),
                        "seed": args.seed,
                        "train_split_sha256": resolved["train_split_sha256"],
                        "val_split_sha256": resolved["val_split_sha256"],
                        "weights_sha256": resolved["weights_sha256"],
                        "gt_policy": resolved["gt_policy"],
                    }
                    torch.save(checkpoint, output_dir / f"best_{level}.pt")
        history.append(epoch_record)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False, allow_nan=True) + "\n")
        print(json.dumps(epoch_record, ensure_ascii=False, allow_nan=True), flush=True)

    summary = {
        "completed_epochs": args.epochs,
        "levels": list(levels),
        "best_metrics": best_metrics,
        "best_selection_keys": {level: list(value) for level, value in best_keys.items()},
        "history_records": len(history),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
