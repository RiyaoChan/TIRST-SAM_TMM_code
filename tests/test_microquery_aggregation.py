import torch

from efficient_sam.microquery import aggregate_query_probabilities


def test_invalid_queries_and_weights_do_not_leak() -> None:
    probabilities = torch.tensor(
        [[[[0.8, 0.0]], [[0.0, 0.9]], [[1.0, 1.0]]]], dtype=torch.float32
    )
    valid = torch.tensor([[True, True, False]])
    weights = torch.tensor([[1.0, 0.5, 1.0]])
    result = aggregate_query_probabilities(probabilities, valid, weights=weights)
    assert torch.allclose(result, torch.tensor([[[0.8, 0.45]]]))


def test_top_n_uses_stable_weight_ranking() -> None:
    probabilities = torch.tensor(
        [[[[0.2]], [[0.7]], [[0.9]]]], dtype=torch.float32
    )
    valid = torch.tensor([[True, True, True]])
    weights = torch.tensor([[0.9, 0.8, 0.1]])
    result = aggregate_query_probabilities(
        probabilities, valid, weights=weights, top_n=1
    )
    assert torch.allclose(result, torch.tensor([[[0.18]]]))

