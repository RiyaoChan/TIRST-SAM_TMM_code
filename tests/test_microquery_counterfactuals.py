import torch

from efficient_sam.microquery import aggregate_query_probabilities


def test_shuffled_reliability_changes_aggregation_not_coordinates() -> None:
    coordinates = torch.tensor([[[2.0, 3.0], [7.0, 8.0]]])
    frozen_coordinates = coordinates.clone()
    probabilities = torch.tensor([[[[0.9, 0.1]], [[0.2, 0.8]]]])
    valid = torch.tensor([[True, True]])
    reliability = torch.tensor([[0.9, 0.1]])
    shuffled = reliability.flip(1)
    correct = aggregate_query_probabilities(
        probabilities, valid, weights=reliability
    )
    counterfactual = aggregate_query_probabilities(
        probabilities, valid, weights=shuffled
    )
    assert torch.equal(coordinates, frozen_coordinates)
    assert not torch.equal(correct, counterfactual)

