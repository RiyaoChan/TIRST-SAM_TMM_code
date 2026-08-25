#!/usr/bin/env python
"""Deterministic C/N/S/W/O text counterfactual evaluation for role-token TIRST-SAM.

The deployable conditions C/N/S never construct point, box, or mask prompts from
ground truth.  W is a deterministic inversion of the cached GPT presence/count
fields and therefore also does not inspect ground truth.  O is the only oracle
condition: it converts the evaluation mask into presence/count role text and is
reported as NOT DEPLOYABLE.

All conditions share one image-encoder forward pass.  This script intentionally
supports the E3 configuration (fused role tokens, no text-conditioned backbone,
no text dense prompt) so the condition comparison changes only the two role
features presented to the learned sparse-prompt projector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from skimage.measure import label
from sklearn.metrics import average_precision_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sirst_dataset import make_loader
from scripts import eval_accuracy_metrics as eval_common
from scripts.build_structured_role_token_features import encode_unique_texts, role_text


CONDITIONS = ("C", "N", "S", "W", "O")
DEPLOYABLE_CONDITIONS = ("C", "N", "S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="50_50/test.txt")
    parser.add_argument("--mask_suffix", default=None)
    parser.add_argument("--mllm_features_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--reference_condition", default="N", choices=CONDITIONS)
    parser.add_argument("--sc_eval_crop", default="resize", choices=("center", "resize"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--clip_batch_size", type=int, default=64)
    parser.add_argument("--max_oracle_count", type=int, default=32)
    parser.add_argument("--pd_fa_dist", type=float, default=3.0)
    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_stop", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--save_probabilities", action="store_true")
    return parser.parse_args()


def parse_conditions(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip().upper() for part in value.split(",") if part.strip()]
    else:
        items = [str(part).strip().upper() for part in value if str(part).strip()]
    if not items:
        raise ValueError("At least one counterfactual condition is required.")
    unknown = [item for item in items if item not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unsupported conditions={unknown}; expected a subset of {CONDITIONS}.")
    if len(set(items)) != len(items):
        raise ValueError(f"Duplicate counterfactual conditions are not allowed: {items}")
    return tuple(items)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_derangement(names: Sequence[str], seed: int) -> Dict[str, str]:
    """Return a deterministic random donor map with no image keeping its own text."""

    names = list(names)
    if len(names) < 2:
        raise ValueError("Shuffled-text evaluation requires at least two samples.")
    rng = random.Random(int(seed))
    order = list(range(len(names)))
    for _ in range(1000):
        rng.shuffle(order)
        if all(index != donor for index, donor in enumerate(order)):
            return {name: names[donor] for name, donor in zip(names, order)}
    # Deterministic fallback; a one-position rotation is always a derangement.
    return {name: names[(index + 1) % len(names)] for index, name in enumerate(names)}


def _cache_item(cache: Mapping[str, object], name: str) -> Mapping[str, object]:
    item = cache.get(name)
    if not isinstance(item, Mapping):
        raise KeyError(f"Role-token cache has no structured item for sample {name!r}.")
    roles = tuple(str(role) for role in item.get("role_names", ()))
    if roles != ("presence", "count"):
        raise ValueError(f"Sample {name!r} roles={roles}; expected ('presence', 'count').")
    return item


def _cached_tokens(cache: Mapping[str, object], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    item = _cache_item(cache, name)
    tokens = item.get("token_features")
    mask = item.get("attention_mask")
    if not torch.is_tensor(tokens) or tuple(tokens.shape) != (2, 512):
        raise ValueError(f"Sample {name!r} token_features must have shape [2,512].")
    if not torch.is_tensor(mask) or tuple(mask.shape) != (2,):
        raise ValueError(f"Sample {name!r} attention_mask must have shape [2].")
    return tokens.float(), mask.long()


def _role_tokens(
    presence: bool,
    count: int,
    role_embeddings: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    attributes = {
        "target_present": bool(presence),
        "count": int(count),
        "position": "unknown",
        "size": "unknown",
    }
    texts = [role_text("presence", attributes), role_text("count", attributes)]
    missing = [text for text in texts if text not in role_embeddings]
    if missing:
        raise KeyError(f"Missing CLIP role embeddings: {missing}")
    tokens = torch.stack([role_embeddings[text].float() for text in texts], dim=0)
    return tokens, torch.ones(2, dtype=torch.long)


def make_condition_features(
    condition: str,
    names: Sequence[str],
    feature_cache: Mapping[str, object],
    shuffle_map: Mapping[str, str],
    role_embeddings: Mapping[str, torch.Tensor],
    gt_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct [B,2,512] role features with an explicit GT-access boundary."""

    condition = str(condition).upper()
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition={condition!r}")
    if condition == "O":
        if gt_masks is None:
            raise ValueError("Oracle condition O requires GT masks.")
        if int(gt_masks.shape[0]) != len(names):
            raise ValueError("Oracle GT batch size must match names.")
    elif gt_masks is not None:
        raise AssertionError(f"Condition {condition} must not receive GT masks.")

    tokens_out = []
    masks_out = []
    for index, name in enumerate(names):
        if condition == "C":
            tokens, mask = _cached_tokens(feature_cache, name)
        elif condition == "N":
            reference, _ = _cached_tokens(feature_cache, name)
            tokens = torch.zeros_like(reference)
            mask = torch.zeros(2, dtype=torch.long)
        elif condition == "S":
            donor = shuffle_map[name]
            if donor == name:
                raise AssertionError("Shuffled-text donor must differ from the image name.")
            tokens, mask = _cached_tokens(feature_cache, donor)
        elif condition == "W":
            item = _cache_item(feature_cache, name)
            role_values = item.get("role_values", {})
            cached_presence = bool(role_values.get("presence", True))
            wrong_presence = not cached_presence
            wrong_count = 1 if wrong_presence else 0
            tokens, mask = _role_tokens(wrong_presence, wrong_count, role_embeddings)
        else:  # O
            gt = gt_masks[index].detach().cpu().numpy() > 0.5
            count = int(label(gt, connectivity=2).max())
            tokens, mask = _role_tokens(count > 0, count, role_embeddings)
        tokens_out.append(tokens)
        masks_out.append(mask)
    return torch.stack(tokens_out, dim=0), torch.stack(masks_out, dim=0)


def _as_bhw(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    if logits.ndim < 3 or int(logits.shape[0]) != int(batch_size):
        raise ValueError(f"Unexpected predicted logits shape={tuple(logits.shape)}")
    height, width = int(logits.shape[-2]), int(logits.shape[-1])
    logits = logits.reshape(batch_size, -1, height, width)
    if int(logits.shape[1]) != 1:
        raise ValueError(f"Expected one output mask per image, got shape={tuple(logits.shape)}")
    return logits[:, 0]


def _safe_auprc(gt: np.ndarray, probability: np.ndarray) -> float:
    gt_flat = gt.reshape(-1).astype(np.uint8)
    if int(gt_flat.sum()) == 0:
        return float("nan")
    return float(average_precision_score(gt_flat, probability.reshape(-1)))


def metrics_at_threshold(
    probabilities: np.ndarray,
    gt_masks: np.ndarray,
    names: Sequence[str],
    threshold: float,
    distance_threshold: float,
) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    total_intersection = 0
    total_union = 0
    total_tp_objects = 0
    total_gt_objects = 0
    total_fp_pixels = 0
    total_pixels = 0

    for name, probability, gt in zip(names, probabilities, gt_masks):
        gt_bool = gt > 0.5
        pred = probability >= float(threshold)
        intersection = int(np.logical_and(pred, gt_bool).sum())
        union = int(np.logical_or(pred, gt_bool).sum())
        iou = intersection / union if union > 0 else 1.0
        denominator = int(pred.sum()) + int(gt_bool.sum())
        f1 = (2.0 * intersection / denominator) if denominator > 0 else 1.0
        tp_objects, gt_objects, fp_pixels, pixels = eval_common.calculate_pd_fa(
            pred,
            gt_bool,
            distance_thresh=float(distance_threshold),
        )
        rows.append(
            {
                "name": name,
                "iou": float(iou),
                "f1": float(f1),
                "mask_auprc": _safe_auprc(gt_bool, probability),
                "intersection": intersection,
                "union": union,
                "pred_pixels": int(pred.sum()),
                "gt_pixels": int(gt_bool.sum()),
                "tp_objects": int(tp_objects),
                "gt_objects": int(gt_objects),
                "fp_pixels": int(fp_pixels),
                "pixels": int(pixels),
            }
        )
        total_intersection += intersection
        total_union += union
        total_tp_objects += int(tp_objects)
        total_gt_objects += int(gt_objects)
        total_fp_pixels += int(fp_pixels)
        total_pixels += int(pixels)

    valid_auprc = [row["mask_auprc"] for row in rows if math.isfinite(row["mask_auprc"])]
    aggregate = {
        "threshold": float(threshold),
        "global_iou": float(total_intersection / total_union) if total_union > 0 else 1.0,
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
        "pd": float(total_tp_objects / total_gt_objects) if total_gt_objects > 0 else 0.0,
        "fa_per_million": float(total_fp_pixels / total_pixels * 1e6) if total_pixels > 0 else 0.0,
        "mean_mask_auprc": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
        "images": len(rows),
        "total_gt_objects": int(total_gt_objects),
        "total_fp_pixels": int(total_fp_pixels),
    }
    return aggregate, rows


def select_threshold(probabilities: np.ndarray, gt_masks: np.ndarray, thresholds: np.ndarray) -> float:
    best_threshold = float(thresholds[0])
    best_mean_iou = -1.0
    gt_bool = gt_masks > 0.5
    for threshold in thresholds:
        pred = probabilities >= float(threshold)
        intersections = np.logical_and(pred, gt_bool).sum(axis=(1, 2))
        unions = np.logical_or(pred, gt_bool).sum(axis=(1, 2))
        ious = np.divide(
            intersections,
            unions,
            out=np.ones_like(intersections, dtype=np.float64),
            where=unions > 0,
        )
        mean_iou = float(ious.mean())
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_threshold = float(threshold)
    return best_threshold


def paired_statistics(values: np.ndarray, seed: int, bootstrap_samples: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    count = int(values.size)
    mean = float(values.mean()) if count else float("nan")
    standard_error = float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else float("nan")
    if count == 0 or bootstrap_samples <= 0:
        low = high = float("nan")
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, count, size=(int(bootstrap_samples), count))
        bootstrap_means = values[indices].mean(axis=1)
        low, high = (float(x) for x in np.percentile(bootstrap_means, [2.5, 97.5]))
    return {
        "n": count,
        "mean": mean,
        "standard_error": standard_error,
        "bootstrap_ci95": [low, high],
        "positive_fraction": float((values > 0).mean()) if count else float("nan"),
    }


def representation_diagnostics(
    input_roles: Mapping[str, np.ndarray],
    sparse_prompts: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
) -> dict:
    """Quantify whether distinct text conditions survive the learned projector."""

    if "C" not in sparse_prompts:
        return {}

    def _paired_distance(left: np.ndarray, right: np.ndarray) -> dict:
        left_flat = left.astype(np.float64).reshape(left.shape[0], -1)
        right_flat = right.astype(np.float64).reshape(right.shape[0], -1)
        delta = left_flat - right_flat
        left_norm = np.linalg.norm(left_flat, axis=1)
        right_norm = np.linalg.norm(right_flat, axis=1)
        cosine = np.sum(left_flat * right_flat, axis=1) / np.maximum(left_norm * right_norm, 1e-12)
        return {
            "mean_l2": float(np.linalg.norm(delta, axis=1).mean()),
            "mean_cosine_distance": float((1.0 - cosine).mean()),
            "mean_abs": float(np.abs(delta).mean()),
            "max_abs": float(np.abs(delta).max()),
        }

    output = {
        "C_across_image_input_role_std_mean": float(input_roles["C"].astype(np.float64).std(axis=0).mean()),
        "C_across_image_sparse_prompt_std_mean": float(sparse_prompts["C"].astype(np.float64).std(axis=0).mean()),
        "conditions_vs_C": {},
    }
    for condition in sparse_prompts:
        if condition == "C":
            continue
        output["conditions_vs_C"][condition] = {
            "input_role_distance": _paired_distance(input_roles[condition], input_roles["C"]),
            "sparse_prompt_distance": _paired_distance(sparse_prompts[condition], sparse_prompts["C"]),
            "probability_map_distance": _paired_distance(probabilities[condition], probabilities["C"]),
        }
    return output


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = parse_args()
    conditions = parse_conditions(cli.conditions)
    if cli.reference_condition not in conditions:
        raise ValueError("reference_condition must be included in --conditions.")

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    checkpoint_path = Path(cli.ckpt).resolve()
    checkpoint_cpu = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint_cpu.get("args", {})
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Checkpoint args must be a mapping.")

    data_root = Path(cli.data_root or checkpoint_args.get("data_root", "")).resolve()
    feature_path = Path(
        cli.mllm_features_path or checkpoint_args.get("mllm_features_path", "")
    ).resolve()
    mask_suffix = cli.mask_suffix
    if mask_suffix is None:
        mask_suffix = str(checkpoint_args.get("mask_suffix", ""))
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(checkpoint_args.get("prompt_mode", "")) != "assp_only":
        raise ValueError("Counterfactual evaluation requires an assp_only checkpoint.")
    if not bool(checkpoint_args.get("use_text_sparse_prompt", False)):
        raise ValueError("Checkpoint does not use a text sparse-prompt projector.")
    if str(checkpoint_args.get("text_sparse_prompt_source", "")) != "fused_tokens":
        raise ValueError("This evaluator requires text_sparse_prompt_source=fused_tokens.")
    if int(checkpoint_args.get("text_sparse_num_tokens", 0)) != 2:
        raise ValueError("This evaluator requires exactly two role sparse tokens.")
    unsupported = {
        "use_gated_bifusion_backbone_blocks": bool(checkpoint_args.get("use_gated_bifusion_backbone_blocks", False)),
        "use_bifusion_backbone_blocks": bool(checkpoint_args.get("use_bifusion_backbone_blocks", False)),
        "use_text_dense_prompt": bool(checkpoint_args.get("use_text_dense_prompt", False)),
    }
    if any(unsupported.values()):
        raise ValueError(f"Shared image-encoder counterfactual evaluation does not support {unsupported}.")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Role-token cache not found: {feature_path}")

    feature_cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    if not isinstance(feature_cache, Mapping):
        raise ValueError("Role-token cache must be a mapping keyed by image stem.")

    size = int(checkpoint_args.get("size", 256))
    loader = make_loader(
        str(data_root),
        cli.split,
        size=size,
        batch_size=max(1, int(cli.batch_size)),
        augment=False,
        shuffle=False,
        workers=max(0, int(cli.workers)),
        keep_ratio_pad=False,
        mask_suffix=mask_suffix,
        sctransnet_preproc=bool(checkpoint_args.get("sctransnet_preproc", False)),
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_pos_prob=float(checkpoint_args.get("sc_pos_prob", 0.5)),
        sc_dataset_name=checkpoint_args.get("sc_dataset_name", None),
        sc_eval_crop=cli.sc_eval_crop,
        mllm_features_path=str(feature_path),
    )
    sample_names = [Path(sample[0]).stem for sample in loader.dataset.samples]
    missing_samples = [name for name in sample_names if name not in feature_cache]
    if missing_samples:
        raise KeyError(f"Feature cache misses {len(missing_samples)} samples, e.g. {missing_samples[:5]}")
    shuffle_map = build_derangement(sample_names, cli.seed)

    role_attribute_sets = [
        {"target_present": False, "count": 0, "position": "unknown", "size": "unknown"},
        {"target_present": True, "count": 1, "position": "unknown", "size": "unknown"},
    ]
    role_attribute_sets.extend(
        {
            "target_present": count > 0,
            "count": count,
            "position": "unknown",
            "size": "unknown",
        }
        for count in range(max(0, int(cli.max_oracle_count)) + 1)
    )
    role_texts = {
        role_text(role, attributes)
        for attributes in role_attribute_sets
        for role in ("presence", "count")
    }
    text_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(role_texts)} unique oracle/control role texts with {cli.clip_model} ...", flush=True)
    role_embeddings = encode_unique_texts(
        role_texts,
        clip_model=cli.clip_model,
        device=text_device,
        batch_size=max(1, int(cli.clip_batch_size)),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Confirm that on-the-fly control text uses the same CLIP representation as
    # the existing cache, up to the cache storage dtype.
    cache_match_error = None
    for name in sample_names:
        item = _cache_item(feature_cache, name)
        texts = list(item.get("role_texts", ()))
        tokens = item.get("token_features")
        mask = item.get("attention_mask")
        if len(texts) == 2 and torch.is_tensor(tokens) and torch.is_tensor(mask):
            active = [index for index in range(2) if int(mask[index]) > 0 and texts[index] in role_embeddings]
            if active:
                cache_match_error = max(
                    float((tokens[index].float() - role_embeddings[texts[index]].float()).abs().max())
                    for index in active
                )
                break
    if cache_match_error is None:
        raise RuntimeError("Could not find an active cached role token for CLIP consistency verification.")
    if cache_match_error > 5e-3:
        raise RuntimeError(f"On-the-fly CLIP role token mismatch is too large: {cache_match_error:.6f}")

    eval_common.cmd_args = SimpleNamespace(
        semantic_source="teacher",
        use_tassg=False,
        tassg_text_dim=None,
        tassg_img_dim=None,
        tassg_num_slots=None,
        tassg_hidden_dim=None,
        tassg_num_heads=None,
        tassg_dropout=None,
        tassg_two_pass_backbone=False,
        prompt_mode="assp_only",
        pd_fa_dist=float(cli.pd_fa_dist),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args = eval_common.build_model_from_ckpt(str(checkpoint_path), device)
    if getattr(model, "text_sparse_projector", None) is None:
        raise RuntimeError("Loaded model has no text_sparse_projector.")

    probability_lists: Dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    input_role_lists: Dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    sparse_prompt_lists: Dict[str, list[np.ndarray]] = {condition: [] for condition in conditions}
    gt_list: list[np.ndarray] = []
    ordered_names: list[str] = []
    active_role_counts = {0: 0, 1: 0, 2: 0}

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{data_root.name} counterfactuals", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            gt_batch = batch["mask"]
            names = [str(name) for name in batch["name"]]
            batch_size = int(images.shape[0])
            img_emb, interms = eval_common._first_two_embeddings(model.get_image_embeddings(images))

            point_coords = torch.zeros((batch_size, 1, 0, 2), dtype=torch.float32, device=device)
            point_labels = torch.zeros((batch_size, 1, 0), dtype=torch.float32, device=device)
            if point_coords.numel() != 0 or point_labels.numel() != 0:
                raise AssertionError("Counterfactual evaluation must use zero-length point prompts.")

            for condition in conditions:
                token_features, attention_mask = make_condition_features(
                    condition,
                    names,
                    feature_cache,
                    shuffle_map,
                    role_embeddings,
                    gt_masks=gt_batch if condition == "O" else None,
                )
                if condition == "C":
                    for active in attention_mask.sum(dim=1).tolist():
                        active_role_counts[int(active)] = active_role_counts.get(int(active), 0) + 1
                token_features = token_features.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                sparse_prompt = model.text_sparse_projector(
                    token_features,
                    attention_mask=attention_mask,
                )
                input_role_lists[condition].append(token_features.detach().cpu().numpy().astype(np.float32))
                sparse_prompt_lists[condition].append(sparse_prompt.detach().cpu().numpy().astype(np.float32))
                predicted_logits, _ = model.predict_masks(
                    img_emb,
                    interms,
                    point_coords,
                    point_labels,
                    multimask_output=False,
                    input_h=size,
                    input_w=size,
                    output_h=size,
                    output_w=size,
                    hq_token_only=False,
                    batched_masks=None,
                    text_sparse_embeddings=sparse_prompt,
                )
                logits = _as_bhw(predicted_logits, batch_size)
                probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                probability_lists[condition].append(probabilities)

            gt_list.append((gt_batch.detach().cpu().numpy() > 0.5).astype(np.uint8))
            ordered_names.extend(names)

    if ordered_names != sample_names:
        raise AssertionError("Evaluation loader order changed unexpectedly.")
    probabilities_by_condition = {
        condition: np.concatenate(parts, axis=0) for condition, parts in probability_lists.items()
    }
    input_roles_by_condition = {
        condition: np.concatenate(parts, axis=0) for condition, parts in input_role_lists.items()
    }
    sparse_prompts_by_condition = {
        condition: np.concatenate(parts, axis=0) for condition, parts in sparse_prompt_lists.items()
    }
    gt_masks = np.concatenate(gt_list, axis=0)
    if any(array.shape != gt_masks.shape for array in probabilities_by_condition.values()):
        shapes = {condition: array.shape for condition, array in probabilities_by_condition.items()}
        raise AssertionError(f"Probability/GT shape mismatch: probabilities={shapes}, gt={gt_masks.shape}")

    thresholds = np.arange(
        float(cli.threshold_start),
        float(cli.threshold_stop) + float(cli.threshold_step) * 0.5,
        float(cli.threshold_step),
        dtype=np.float64,
    )
    thresholds = np.round(thresholds, 6)
    own_thresholds = {
        condition: select_threshold(probabilities_by_condition[condition], gt_masks, thresholds)
        for condition in conditions
    }
    reference_threshold = own_thresholds[cli.reference_condition]

    aggregate_rows: list[dict] = []
    per_image_rows: list[dict] = []
    reference_metrics: Dict[str, dict] = {}
    reference_per_image: Dict[str, list[dict]] = {}
    own_metrics: Dict[str, dict] = {}
    for condition in conditions:
        aggregate, rows = metrics_at_threshold(
            probabilities_by_condition[condition],
            gt_masks,
            ordered_names,
            reference_threshold,
            cli.pd_fa_dist,
        )
        aggregate.update({"condition": condition, "threshold_policy": f"fixed_{cli.reference_condition}"})
        reference_metrics[condition] = aggregate
        reference_per_image[condition] = rows
        aggregate_rows.append(dict(aggregate))
        per_image_rows.extend({"condition": condition, **row} for row in rows)

        own_aggregate, _ = metrics_at_threshold(
            probabilities_by_condition[condition],
            gt_masks,
            ordered_names,
            own_thresholds[condition],
            cli.pd_fa_dist,
        )
        own_aggregate.update({"condition": condition, "threshold_policy": "condition_own"})
        own_metrics[condition] = own_aggregate
        aggregate_rows.append(dict(own_aggregate))

    paired_summary = {}
    paired_rows: list[dict] = []
    if "N" in conditions:
        no_text_by_name = {row["name"]: row for row in reference_per_image["N"]}
        for condition in conditions:
            if condition == "N":
                continue
            condition_rows = reference_per_image[condition]
            for row in condition_rows:
                baseline = no_text_by_name[row["name"]]
                paired_rows.append(
                    {
                        "name": row["name"],
                        "condition": condition,
                        "delta_iou_vs_N": float(row["iou"] - baseline["iou"]),
                        "delta_f1_vs_N": float(row["f1"] - baseline["f1"]),
                        "delta_auprc_vs_N": float(row["mask_auprc"] - baseline["mask_auprc"])
                        if math.isfinite(row["mask_auprc"]) and math.isfinite(baseline["mask_auprc"])
                        else float("nan"),
                    }
                )
            deltas = [item for item in paired_rows if item["condition"] == condition]
            paired_summary[condition] = {
                "iou_vs_N": paired_statistics(
                    np.asarray([item["delta_iou_vs_N"] for item in deltas]),
                    cli.seed + ord(condition),
                    cli.bootstrap_samples,
                ),
                "f1_vs_N": paired_statistics(
                    np.asarray([item["delta_f1_vs_N"] for item in deltas]),
                    cli.seed + 100 + ord(condition),
                    cli.bootstrap_samples,
                ),
                "auprc_vs_N": paired_statistics(
                    np.asarray(
                        [item["delta_auprc_vs_N"] for item in deltas if math.isfinite(item["delta_auprc_vs_N"])]
                    ),
                    cli.seed + 200 + ord(condition),
                    cli.bootstrap_samples,
                ),
                "aggregate_metric_delta_vs_N": {
                    key: float(reference_metrics[condition][key] - reference_metrics["N"][key])
                    for key in ("global_iou", "mean_iou", "mean_f1", "pd", "fa_per_million", "mean_mask_auprc")
                },
            }

    decision = {
        "screen_only_single_checkpoint": True,
        "teacher_positive_delta_required_before_distillation": True,
        "oracle_beats_no_text_mean_iou": None,
        "oracle_positive_delta_supported": None,
        "correct_text_beats_no_text_mean_iou": None,
        "correct_text_ci95_excludes_zero": None,
        "correct_text_positive_delta_supported": None,
        "proceed_to_behavior_distillation": False,
    }
    if "N" in conditions and "O" in conditions:
        decision["oracle_beats_no_text_mean_iou"] = bool(
            reference_metrics["O"]["mean_iou"] > reference_metrics["N"]["mean_iou"]
        )
        oracle_ci = paired_summary["O"]["iou_vs_N"]["bootstrap_ci95"]
        decision["oracle_positive_delta_supported"] = bool(oracle_ci[0] > 0)
    if "N" in conditions and "C" in conditions:
        decision["correct_text_beats_no_text_mean_iou"] = bool(
            reference_metrics["C"]["mean_iou"] > reference_metrics["N"]["mean_iou"]
        )
        ci = paired_summary["C"]["iou_vs_N"]["bootstrap_ci95"]
        decision["correct_text_ci95_excludes_zero"] = bool(ci[0] > 0 or ci[1] < 0)
        decision["correct_text_positive_delta_supported"] = bool(ci[0] > 0)
        decision["proceed_to_behavior_distillation"] = bool(ci[0] > 0)

    manifest = {
        "schema_version": "tirst-sam-text-counterfactual-eval-v1",
        "dataset": data_root.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_cpu.get("epoch", -1)),
        "feature_cache": str(feature_path),
        "feature_cache_sha256": sha256_file(feature_path),
        "split": cli.split,
        "images": len(ordered_names),
        "size": size,
        "sc_eval_crop": cli.sc_eval_crop,
        "conditions": list(conditions),
        "condition_contract": {
            "C": "cached GPT presence/count role tokens; no evaluation GT access",
            "N": "zero role features and zero attention mask; same learned sparse projector",
            "S": "deterministic derangement of cached GPT role tokens; no evaluation GT access",
            "W": "presence/count inversion derived only from cached GPT role values; no evaluation GT access",
            "O": "GT-mask-derived presence/count oracle; NOT DEPLOYABLE",
        },
        "prompt_contract": {
            "prompt_mode": "assp_only",
            "point_coords_shape": "[B,1,0,2]",
            "point_labels_shape": "[B,1,0]",
            "boxes": None,
            "mask_inputs": None,
            "gt_prompt_used": False,
        },
        "reference_condition": cli.reference_condition,
        "reference_threshold": reference_threshold,
        "thresholds": thresholds.tolist(),
        "own_thresholds": own_thresholds,
        "seed": cli.seed,
        "clip_model": cli.clip_model,
        "clip_cache_match_max_abs_error": cache_match_error,
        "cached_C_active_role_counts": active_role_counts,
        "device": device,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary = {
        "manifest": manifest,
        "reference_metrics": reference_metrics,
        "condition_own_threshold_metrics": own_metrics,
        "paired_summary": paired_summary,
        "representation_diagnostics": representation_diagnostics(
            input_roles_by_condition,
            sparse_prompts_by_condition,
            probabilities_by_condition,
        ),
        "decision_screen": decision,
    }

    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_dir / "per_image_metrics.csv", per_image_rows)
    write_csv(output_dir / "paired_deltas_vs_N.csv", paired_rows)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2)
    if cli.save_probabilities:
        np.savez_compressed(
            output_dir / "probabilities_and_gt.npz",
            names=np.asarray(ordered_names),
            gt=gt_masks,
            **{f"prob_{condition}": array for condition, array in probabilities_by_condition.items()},
        )

    print("\n=== Counterfactual reference-threshold metrics ===", flush=True)
    print(f"reference={cli.reference_condition}, threshold={reference_threshold:.2f}", flush=True)
    for condition in conditions:
        metrics = reference_metrics[condition]
        print(
            f"{condition}: globalIoU={metrics['global_iou'] * 100:.2f} "
            f"meanIoU={metrics['mean_iou'] * 100:.2f} "
            f"F1={metrics['mean_f1'] * 100:.2f} Pd={metrics['pd'] * 100:.2f} "
            f"Fa={metrics['fa_per_million']:.2f} AUPRC={metrics['mean_mask_auprc'] * 100:.2f}",
            flush=True,
        )
    print(json.dumps(_json_safe(decision), ensure_ascii=False, indent=2), flush=True)
    print(f"Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
