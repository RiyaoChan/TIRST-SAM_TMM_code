from dataclasses import replace

import torch

from efficient_sam.multiview_prompt import (
    CandidateCluster,
    clusters_to_proposal,
    rule_reliability,
)


def _cluster(**overrides):
    base = CandidateCluster(
        center_xy=(10.0, 12.0),
        mean_score=0.8,
        max_score=0.9,
        score_variance=0.02,
        support_count=3,
        support_fraction=0.6,
        center_dispersion=0.5,
        rank_mean=2.0,
        rank_variance=0.1,
    )
    return replace(base, **overrides)


def test_rule_reliability_monotonicity():
    base = _cluster()
    assert rule_reliability(_cluster(support_count=4, support_fraction=0.8)) >= rule_reliability(base)
    assert rule_reliability(_cluster(center_dispersion=1.0)) <= rule_reliability(base)
    assert rule_reliability(_cluster(score_variance=0.2)) <= rule_reliability(base)


def test_all_rejected_returns_valid_zero_prompt_fp32_and_fp16():
    for dtype in (torch.float32, torch.float16):
        dense = torch.zeros((1, 1, 16, 16), dtype=dtype)
        proposal = clusters_to_proposal(
            [[_cluster(support_count=1, support_fraction=0.2)]],
            dense_mean=dense,
            gate="rule",
            min_support=3,
        )
        assert not proposal.candidate_valid.any()
        assert proposal.candidate_xy.dtype == dtype


def test_a2_score_controls_are_explicit_and_deterministic():
    clusters = [[_cluster(mean_score=0.6, max_score=0.9, support_fraction=0.8)]]
    dense = torch.zeros((1, 1, 8, 8))
    expected = {"mean": 0.6, "max": 0.9, "support": 0.8, "mean_max": 0.75}
    for mode, score in expected.items():
        proposal = clusters_to_proposal(
            clusters, dense_mean=dense, gate="none", score_mode=mode
        )
        assert torch.isclose(proposal.candidate_scores[0, 0], torch.tensor(score))
