import inspect

import torch

from efficient_sam.prompt_proposal import (
    DoGLoGProposalAdapter,
    PGAPProposalAdapter,
    extract_local_maxima,
)


def test_local_maxima_allows_zero_prompt():
    dense = torch.zeros(2, 1, 16, 16)
    proposal = extract_local_maxima(dense, candidate_k_raw=5, score_threshold=0.5)
    assert proposal.candidate_xy.shape == (2, 5, 2)
    assert not proposal.candidate_valid.any()
    coords, labels = proposal.to_point_prompts()
    assert coords.shape == (2, 1, 5, 2)
    assert torch.equal(labels, torch.full((2, 1, 5), -1, dtype=torch.int64))


def test_local_maxima_is_score_sorted_and_nms_filtered():
    dense = torch.zeros(1, 1, 12, 12)
    dense[0, 0, 2, 3] = 0.9
    dense[0, 0, 2, 4] = 0.8
    dense[0, 0, 9, 8] = 0.7
    proposal = extract_local_maxima(
        dense,
        candidate_k_raw=4,
        nms_radius=3,
        score_threshold=0.5,
    )
    assert proposal.candidate_valid.sum().item() == 2
    assert torch.equal(proposal.candidate_xy[0, 0], torch.tensor([3.0, 2.0]))
    assert torch.equal(proposal.candidate_xy[0, 1], torch.tensor([8.0, 9.0]))
    assert proposal.candidate_scores[0, 0] > proposal.candidate_scores[0, 1]


def test_image_only_adapters_do_not_accept_gt():
    for adapter in (PGAPProposalAdapter, DoGLoGProposalAdapter):
        signature = inspect.signature(adapter.forward)
        assert "gt" not in " ".join(signature.parameters).lower()
        assert "mask" not in " ".join(signature.parameters).lower()


def test_same_image_proposal_is_independent_of_external_masks():
    adapter = DoGLoGProposalAdapter(candidate_k_raw=6, score_threshold=0.2)
    image = torch.zeros(1, 1, 32, 32)
    image[0, 0, 10, 12] = 1.0
    first = adapter(image)
    _unused_mask_a = torch.zeros(1, 32, 32)
    _unused_mask_b = torch.ones(1, 32, 32)
    second = adapter(image.clone())
    assert torch.equal(first.candidate_valid, second.candidate_valid)
    assert torch.equal(first.candidate_xy, second.candidate_xy)
    assert torch.equal(first.candidate_scores, second.candidate_scores)
