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


class _IdentityHead(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :1]


def test_proposal_forward_signatures_do_not_accept_masks_or_points():
    forbidden = {"mask", "masks", "gt", "gt_mask", "points", "point_labels"}
    for adapter in (PGAPProposalAdapter, DoGLoGProposalAdapter, DenseHeadProposalAdapter):
        parameters = set(inspect.signature(adapter.forward).parameters)
        assert not parameters.intersection(forbidden), (adapter.__name__, parameters)


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

