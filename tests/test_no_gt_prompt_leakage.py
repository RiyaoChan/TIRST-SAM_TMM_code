"""Regression tests for the Experiment-1 image-only proposal boundary."""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from efficient_sam.prompt_proposal import (
    DenseHeadProposalAdapter,
    DoGLoGProposalAdapter,
    PGAPProposalAdapter,
)
from efficient_sam.multiview_prompt import multiview_propose, rule_reliability
from scripts.eval_prompt_quality import enforce_test_config_freeze
from scripts.train_experiment1_single_view import ImageOnlyProposalGenerator


class _IdentityHead(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :1]


def test_proposal_forward_signatures_do_not_accept_masks_or_points():
    forbidden = {"mask", "masks", "gt", "gt_mask", "points", "point_labels"}
    for adapter in (PGAPProposalAdapter, DoGLoGProposalAdapter, DenseHeadProposalAdapter):
        parameters = set(inspect.signature(adapter.forward).parameters)
        assert not parameters.intersection(forbidden), (adapter.__name__, parameters)
    for callable_object in (
        ImageOnlyProposalGenerator.__call__,
        multiview_propose,
        rule_reliability,
    ):
        parameters = set(inspect.signature(callable_object).parameters)
        assert not parameters.intersection(forbidden), (callable_object, parameters)


def test_strict_image_only_generator_rejects_gt_keyword_before_sampling():
    sampler = object.__new__(ImageOnlyProposalGenerator)
    try:
        sampler(None, None, None, gt_mask=torch.ones(1, 1, 8, 8))
    except TypeError as error:
        assert "gt_mask" in str(error)
    else:
        raise AssertionError("Image-only generator unexpectedly accepted gt_mask")


def test_dense_proposal_is_invariant_to_external_gt_changes():
    adapter = DenseHeadProposalAdapter(
        _IdentityHead(), candidate_k_raw=4, nms_radius=1.0, score_threshold=0.1
    )
    features = torch.full((1, 2, 8, 8), -8.0)
    features[0, 0, 2, 3] = 8.0
    features[0, 0, 6, 5] = 7.0

    gt_a = torch.zeros((1, 1, 8, 8))
    gt_b = torch.ones((1, 1, 8, 8))
    proposal_a = adapter(features.clone(), output_size=(8, 8))
    proposal_b = adapter(features.clone(), output_size=(8, 8))

    # Keep the deliberately different masks alive in the test while proving
    # that neither can cross the adapter boundary.
    assert not torch.equal(gt_a, gt_b)
    assert torch.equal(proposal_a.candidate_xy, proposal_b.candidate_xy)
    assert torch.equal(proposal_a.candidate_scores, proposal_b.candidate_scores)
    assert torch.equal(proposal_a.candidate_valid, proposal_b.candidate_valid)


def test_all_adapters_allow_zero_prompt_outputs():
    dense = DenseHeadProposalAdapter(
        _IdentityHead(), candidate_k_raw=4, nms_radius=1.0, score_threshold=0.9
    )
    proposal = dense(torch.full((2, 1, 8, 8), -20.0), output_size=(8, 8))
    coords, labels = proposal.to_point_prompts()

    assert not proposal.candidate_valid.any()
    assert torch.count_nonzero(coords) == 0
    assert torch.all(labels == -1)


def test_image_only_proposal_is_bitwise_deterministic():
    torch.manual_seed(7)
    images = torch.rand((2, 1, 32, 32))
    adapter = DoGLoGProposalAdapter(
        candidate_k_raw=8, nms_radius=3.0, score_threshold=0.1
    )
    first = adapter(images)
    second = adapter(images)
    assert torch.equal(first.dense_probs, second.dense_probs)
    assert torch.equal(first.candidate_xy, second.candidate_xy)
    assert torch.equal(first.candidate_valid, second.candidate_valid)


def test_test_split_requires_frozen_validation_config():
    enforce_test_config_freeze("val", None)
    enforce_test_config_freeze("test", "frozen-val-config.json")
    try:
        enforce_test_config_freeze("test", None)
    except ValueError as error:
        assert "frozen_config_manifest" in str(error)
    else:
        raise AssertionError("Test split was allowed to enter an unfrozen selection run")
