import torch

from efficient_sam.microquery import MicroQueryHead, aggregate_query_probabilities


def test_objectness_threshold_can_produce_zero_queries() -> None:
    head = MicroQueryHead(input_dim=5, hidden_dim=8, dropout=0.0).eval()
    descriptors = torch.zeros(1, 3, 5)
    valid = torch.tensor([[True, True, True]])
    with torch.inference_mode():
        scores = torch.softmax(head(descriptors, valid).object_logits, dim=-1)[..., 1]
    accepted = valid & (scores > 1.0)
    probabilities = torch.ones(1, 3, 4, 4)
    final = aggregate_query_probabilities(probabilities, accepted)
    assert not accepted.any()
    assert torch.count_nonzero(final) == 0


def test_empty_query_axis_is_safe() -> None:
    probabilities = torch.empty(2, 0, 4, 5)
    valid = torch.empty(2, 0, dtype=torch.bool)
    result = aggregate_query_probabilities(probabilities, valid)
    assert result.shape == (2, 4, 5)
    assert torch.count_nonzero(result) == 0

