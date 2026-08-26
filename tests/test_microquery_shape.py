import torch

from efficient_sam.microquery import proposal_to_point_queries
from efficient_sam.prompt_proposal import PromptProposal


def _proposal() -> PromptProposal:
    return PromptProposal(
        dense_logits=None,
        dense_probs=None,
        candidate_xy=torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]),
        candidate_scores=torch.tensor([[0.9, 0.8, 0.0]]),
        candidate_valid=torch.tensor([[True, True, False]]),
    )


def test_one_query_and_independent_query_shapes() -> None:
    one_coords, one_labels = proposal_to_point_queries(_proposal(), 3, independent=False)
    micro_coords, micro_labels = proposal_to_point_queries(_proposal(), 3, independent=True)
    assert one_coords.shape == (1, 1, 3, 2)
    assert one_labels.shape == (1, 1, 3)
    assert micro_coords.shape == (1, 3, 1, 2)
    assert micro_labels.shape == (1, 3, 1)
    assert micro_labels.tolist() == [[[1], [1], [-1]]]

