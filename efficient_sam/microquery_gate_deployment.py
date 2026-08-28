"""Explicit deployment-time gate policies for MicroQuery.

Training warm-up remains in :mod:`efficient_sam.microquery_end2end`.  This
module deliberately contains no trainable state and never infers a deployment
policy from the checkpoint epoch unless the caller explicitly requests the
legacy reproduction mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from efficient_sam.microquery_end2end import gate_temperature, gate_warmup_rho


GateDeploymentMode = Literal[
    "all_one",
    "raw",
    "residual",
    "legacy_checkpoint_schedule",
]


@dataclass(frozen=True)
class GateDeploymentConfig:
    """Frozen policy used only by evaluation/deployment forward passes."""

    mode: GateDeploymentMode
    rho: float = 0.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        validate_gate_config(self)


def validate_gate_config(config: GateDeploymentConfig) -> None:
    if config.mode not in {
        "all_one",
        "raw",
        "residual",
        "legacy_checkpoint_schedule",
    }:
        raise ValueError(f"unknown deployment gate mode: {config.mode}")
    if not 0.0 <= float(config.rho) <= 1.0:
        raise ValueError("deployment gate rho must be in [0, 1]")
    if float(config.temperature) <= 0.0:
        raise ValueError("deployment gate temperature must be positive")
    if config.mode == "raw" and float(config.rho) != 0.0:
        raise ValueError("raw deployment gate requires rho=0")
    if config.mode == "legacy_checkpoint_schedule" and (
        float(config.rho) != 0.0 or float(config.temperature) != 1.0
    ):
        raise ValueError(
            "legacy_checkpoint_schedule derives rho/temperature from the epoch; "
            "leave config values at their defaults"
        )


def legacy_gate_from_epoch(checkpoint_epoch: int) -> tuple[float, float]:
    """Return the historical residual floor and temperature for one epoch."""

    if checkpoint_epoch is None:
        raise ValueError("legacy deployment gate requires checkpoint_epoch")
    epoch = int(checkpoint_epoch)
    if epoch <= 0:
        raise ValueError("checkpoint_epoch must be positive")
    return gate_warmup_rho(epoch), gate_temperature(epoch)


def resolve_gate_parameters(
    config: GateDeploymentConfig, *, checkpoint_epoch: Optional[int] = None
) -> tuple[float, float]:
    validate_gate_config(config)
    if config.mode == "legacy_checkpoint_schedule":
        return legacy_gate_from_epoch(checkpoint_epoch)  # type: ignore[arg-type]
    if checkpoint_epoch is not None:
        # An explicit configuration must be independent of checkpoint naming.
        checkpoint_epoch = int(checkpoint_epoch)
    if config.mode == "all_one":
        return 1.0, float(config.temperature)
    if config.mode == "raw":
        return 0.0, float(config.temperature)
    return float(config.rho), float(config.temperature)


def compute_deployment_gate(
    object_logits: torch.Tensor,
    candidate_valid: torch.Tensor,
    config: GateDeploymentConfig,
    *,
    checkpoint_epoch: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Compute raw/effective gates from an explicit immutable policy."""

    if object_logits.shape[:-1] != candidate_valid.shape or object_logits.shape[-1] != 2:
        raise ValueError("object_logits must have shape [B,K,2]")
    rho, temperature = resolve_gate_parameters(config, checkpoint_epoch=checkpoint_epoch)
    raw = torch.softmax(object_logits / temperature, dim=-1)[..., 1]
    raw = torch.where(candidate_valid, raw, torch.zeros_like(raw))
    if config.mode == "all_one":
        effective = candidate_valid.to(raw.dtype)
    else:
        effective = rho + (1.0 - rho) * raw
        effective = torch.where(candidate_valid, effective, torch.zeros_like(effective))
    return raw, effective, rho, temperature


def gate_config_id(config: GateDeploymentConfig, *, checkpoint_epoch: Optional[int] = None) -> str:
    """Stable filesystem/record identifier for a deployment gate policy."""

    rho, temperature = resolve_gate_parameters(config, checkpoint_epoch=checkpoint_epoch)
    if config.mode == "all_one":
        return "all_one"
    if config.mode == "legacy_checkpoint_schedule":
        return f"legacy_rho{rho:.6g}_t{temperature:.7g}"
    return f"{config.mode}_rho{rho:.6g}_t{temperature:.6g}"
