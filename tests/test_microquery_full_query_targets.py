import torch

from efficient_sam.microquery_end2end import build_query_targets


def test_query_targets_are_component_or_zero_masks():
    components = torch.tensor([[0, 1], [2, 2]])
    targets = build_query_targets(components, torch.tensor([1, 2, -1]))
    assert targets.shape == (3, 2, 2)
    assert targets[0].sum() == 1
    assert targets[1].sum() == 2
    assert targets[2].sum() == 0

