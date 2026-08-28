#!/usr/bin/env python3
"""Shared model construction and deployable forward for full MicroQuery runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.microquery import extract_candidate_roi_features
from efficient_sam.microquery_gate_deployment import (
    GateDeploymentConfig,
    compute_deployment_gate,
)
from efficient_sam.microquery_end2end import (
    EndToEndMicroQueryHead,
    aggregate_soft_gated_max,
    append_candidate_sparse_token,
    soft_gate_schedule,
)


VARIANTS = (
    "c0_one_query",
    "c1_independent_aux",
    "f1_soft_gate",
    "f2_gate_token",
)


@dataclass(frozen=True)
class FullForwardOutput:
    final_probability: torch.Tensor
    query_logits: torch.Tensor
    iou_predictions: torch.Tensor
    raw_gates: torch.Tensor
    effective_gates: torch.Tensor
    object_logits: Optional[torch.Tensor]
    candidate_tokens: Optional[torch.Tensor]
    descriptors: Optional[torch.Tensor]
    gate_rho: float
    gate_temperature: float


def assert_optional_modules_off(model) -> None:
    checks = {
        "use_adapter": bool(getattr(model.image_encoder, "use_adapter", False)),
        "use_ms_fusion": bool(getattr(model, "use_ms_fusion", False)),
        "use_detail_enhancer": bool(getattr(model, "use_detail_enhancer", False)),
        "use_amgd": bool(getattr(model.mask_decoder, "use_amgd", False)),
        "use_dog_amgd": bool(getattr(model.mask_decoder, "use_dog_amgd", False)),
        "use_hldf": bool(getattr(model, "use_hldf", False)),
        "use_center_mask_decoder": bool(
            getattr(model.mask_decoder, "use_center_mask_decoder", False)
        ),
        "task_tokens": getattr(model.prompt_encoder, "task_tokens", None) is not None,
    }
    enabled = [name for name, value in checks.items() if value]
    if enabled:
        raise RuntimeError(f"optional modules must be off: {enabled}")


def build_full_sam(
    a1_checkpoint: str | Path,
    baseline_weights: str | Path,
    device: torch.device,
):
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(Path(baseline_weights).resolve()),
        use_adapter=False,
        use_ms_fusion=False,
        use_detail_enhancer=False,
        use_amgd=False,
        use_dog_amgd=False,
        use_hldf=False,
        use_center_mask_decoder=False,
        return_encoder_multi_scale=True,
    )
    checkpoint = torch.load(Path(a1_checkpoint).resolve(), map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    assert_optional_modules_off(model)
    configure_trainable_modules(model)
    model.image_encoder.eval()
    return model, checkpoint


def configure_trainable_modules(model) -> None:
    """Freeze the complete model, then enable only PromptEncoderHQ/MaskDecoderHQ."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.prompt_encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.mask_decoder.parameters():
        parameter.requires_grad_(True)


def trainable_parameter_counts(model, head: Optional[EndToEndMicroQueryHead]) -> dict[str, int]:
    return {
        "head": 0 if head is None else sum(p.numel() for p in head.parameters() if p.requires_grad),
        "prompt_encoder": sum(p.numel() for p in model.prompt_encoder.parameters() if p.requires_grad),
        "mask_decoder": sum(p.numel() for p in model.mask_decoder.parameters() if p.requires_grad),
        "image_encoder": sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad),
        "total_model": sum(p.numel() for p in model.parameters()),
    }


def encode_frozen_image(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.image_encoder.eval()
    with torch.no_grad():
        encoded = model.get_image_embeddings(images)
    if len(encoded) < 3 or not encoded[2]:
        raise RuntimeError("ImageEncoderViTHQ did not return shallow multi-scale features")
    return encoded[0].detach(), encoded[1].detach(), encoded[2][0].detach()


def _decode_chunks(
    model,
    neck: torch.Tensor,
    interm: torch.Tensor,
    points: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_h: int,
    output_w: int,
    query_chunk: int,
    candidate_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, query_count, point_count, _ = points.shape
    scaled_points = model.get_rescaled_pts(points, output_h, output_w)
    chunk_size = min(max(1, int(query_chunk)), query_count)
    mask_rows: list[torch.Tensor] = []
    quality_rows: list[torch.Tensor] = []
    dense_pe = model.prompt_encoder.get_dense_pe()
    for start in range(0, query_count, chunk_size):
        end = min(query_count, start + chunk_size)
        width = end - start
        flat_points = scaled_points[:, start:end].reshape(batch_size * width, point_count, 2)
        flat_labels = labels[:, start:end].reshape(batch_size * width, point_count)
        sparse, dense = model.prompt_encoder(
            points=(flat_points, flat_labels), boxes=None, masks=None, text_embeds=None
        )
        if candidate_tokens is not None:
            flat_tokens = candidate_tokens[:, start:end].reshape(batch_size * width, -1)
            sparse = append_candidate_sparse_token(sparse, flat_tokens)
        low_res, quality = model.mask_decoder(
            neck.repeat_interleave(width, dim=0),
            dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            hq_token_only=False,
            interm_embeddings=interm.repeat_interleave(width, dim=0),
            multi_scale_embeddings=None,
            detail_branch_embeddings=None,
        )
        resized = F.interpolate(low_res, (output_h, output_w), mode="bicubic")
        mask_rows.append(resized.reshape(batch_size, width, output_h, output_w))
        quality_rows.append(quality.reshape(batch_size, width))
    return torch.cat(mask_rows, dim=1), torch.cat(quality_rows, dim=1)


def forward_deployable(
    model,
    head: Optional[EndToEndMicroQueryHead],
    deployable: dict[str, torch.Tensor],
    *,
    variant: str,
    training_epoch: Optional[int] = None,
    gate_deployment_config: Optional[GateDeploymentConfig] = None,
    checkpoint_epoch: Optional[int] = None,
    query_chunk: int = 5,
    gate_condition: str = "correct",
    token_condition: str = "correct",
    coordinate_condition: str = "correct",
    coordinate_override: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    image_embeddings: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> FullForwardOutput:
    """Forward without any GT/mask/assignment argument."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    images = deployable["image"]
    xy = deployable["candidate_xy"].to(images.dtype)
    scores = deployable["candidate_scores"].to(images.dtype)
    valid = deployable["candidate_valid"].bool()
    height, width = images.shape[-2:]
    point_xy = xy if coordinate_override is None else coordinate_override.to(device=xy.device, dtype=xy.dtype)
    if point_xy.shape != xy.shape:
        raise ValueError("coordinate_override must have shape [B,K,2]")
    point_valid = valid
    if coordinate_condition == "candidate_shuffled":
        point_xy = torch.roll(xy, shifts=1, dims=1)
    elif coordinate_condition == "invalid":
        point_valid = torch.zeros_like(valid)
    elif coordinate_condition == "random_background":
        point_xy = torch.rand(xy.shape, device=xy.device, dtype=xy.dtype, generator=generator)
        point_xy[..., 0] *= float(width - 1)
        point_xy[..., 1] *= float(height - 1)
    elif coordinate_condition != "correct":
        raise ValueError(f"unknown coordinate condition: {coordinate_condition}")
    if image_embeddings is None:
        neck, interm, shallow = encode_frozen_image(model, images)
    else:
        if len(image_embeddings) != 3:
            raise ValueError("image_embeddings must contain neck, intermediate and shallow tensors")
        neck, interm, shallow = image_embeddings
        if neck.shape[0] != images.shape[0] or interm.shape[0] != images.shape[0] or shallow.shape[0] != images.shape[0]:
            raise ValueError("image_embeddings batch dimension must match deployable image")
    labels = torch.where(
        point_valid,
        torch.ones_like(point_valid, dtype=torch.long),
        -torch.ones_like(point_valid, dtype=torch.long),
    )
    safe_xy = torch.where(point_valid.unsqueeze(-1), point_xy, torch.zeros_like(point_xy))
    if variant == "c0_one_query":
        query_logits, quality = _decode_chunks(
            model,
            neck,
            interm,
            safe_xy.unsqueeze(1),
            labels.unsqueeze(1),
            output_h=height,
            output_w=width,
            query_chunk=1,
        )
        one_valid = valid.any(dim=1, keepdim=True)
        gates = one_valid.to(query_logits.dtype)
        final = aggregate_soft_gated_max(query_logits, gates, one_valid)
        return FullForwardOutput(
            final, query_logits, quality, gates, gates, None, None, None, 0.0, 1.0
        )
    if head is None:
        raise ValueError("C1/F1/F2 require a MicroQuery head")
    descriptors = extract_candidate_roi_features(
        shallow,
        neck,
        xy,
        scores,
        valid,
        input_h=height,
        input_w=width,
    ).detach()
    head_output = head(descriptors, valid)
    if model.training:
        if training_epoch is None:
            raise ValueError("training forward requires explicit training_epoch")
        if gate_deployment_config is not None or checkpoint_epoch is not None:
            raise ValueError("training forward cannot consume a deployment gate configuration")
        raw_gate, effective_gate, rho, temperature = soft_gate_schedule(
            head_output.object_logits,
            valid,
            training_epoch,
            force_all_one=variant == "c1_independent_aux",
        )
    else:
        if training_epoch is not None:
            raise ValueError("evaluation forward cannot consume training_epoch")
        if gate_deployment_config is None:
            raise ValueError("evaluation forward requires explicit gate_deployment_config")
        raw_gate, effective_gate, rho, temperature = compute_deployment_gate(
            head_output.object_logits,
            valid,
            gate_deployment_config,
            checkpoint_epoch=checkpoint_epoch,
        )
    if gate_condition == "all_one":
        effective_gate = valid.to(raw_gate.dtype)
    elif gate_condition == "zero":
        effective_gate = torch.zeros_like(raw_gate)
    elif gate_condition == "batch_shuffled":
        effective_gate = torch.roll(effective_gate, shifts=1, dims=0)
    elif gate_condition == "candidate_shuffled":
        effective_gate = torch.roll(effective_gate, shifts=1, dims=1)
    elif gate_condition == "inverted":
        effective_gate = torch.where(valid, 1.0 - effective_gate, 0.0)
    elif gate_condition != "correct":
        raise ValueError(f"unknown gate condition: {gate_condition}")
    tokens = head_output.candidate_token
    decode_tokens: Optional[torch.Tensor] = None
    if variant == "f2_gate_token":
        if token_condition == "correct":
            decode_tokens = tokens
        elif token_condition in {"zero", "coordinate_only"}:
            decode_tokens = torch.zeros_like(tokens) if token_condition == "zero" else None
        elif token_condition == "batch_shuffled":
            decode_tokens = torch.roll(tokens, shifts=1, dims=0)
        elif token_condition == "candidate_shuffled":
            decode_tokens = torch.roll(tokens, shifts=1, dims=1)
        elif token_condition == "random":
            mean = tokens.detach().mean()
            std = tokens.detach().std().clamp_min(1e-6)
            decode_tokens = torch.randn(tokens.shape, device=tokens.device, dtype=tokens.dtype, generator=generator) * std + mean
            decode_tokens = torch.where(valid.unsqueeze(-1), decode_tokens, torch.zeros_like(decode_tokens))
        else:
            raise ValueError(f"unknown token condition: {token_condition}")
    query_logits, quality = _decode_chunks(
        model,
        neck,
        interm,
        safe_xy.unsqueeze(2),
        labels.unsqueeze(2),
        output_h=height,
        output_w=width,
        query_chunk=query_chunk,
        candidate_tokens=decode_tokens,
    )
    final = aggregate_soft_gated_max(query_logits, effective_gate, valid)
    return FullForwardOutput(
        final,
        query_logits,
        quality,
        raw_gate,
        effective_gate,
        head_output.object_logits,
        tokens,
        descriptors,
        rho,
        temperature,
    )


def checkpoint_state(model, head: Optional[EndToEndMicroQueryHead]) -> dict:
    return {
        "prompt_encoder": {key: value.detach().cpu() for key, value in model.prompt_encoder.state_dict().items()},
        "mask_decoder": {key: value.detach().cpu() for key, value in model.mask_decoder.state_dict().items()},
        "head": None if head is None else {key: value.detach().cpu() for key, value in head.state_dict().items()},
    }


def load_checkpoint_state(model, head: Optional[EndToEndMicroQueryHead], state: dict) -> None:
    model.prompt_encoder.load_state_dict(state["prompt_encoder"], strict=True)
    model.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
    if head is None:
        if state.get("head") is not None:
            raise RuntimeError("C0 checkpoint unexpectedly contains a head")
    else:
        if state.get("head") is None:
            raise RuntimeError("MicroQuery checkpoint does not contain a head")
        head.load_state_dict(state["head"], strict=True)
