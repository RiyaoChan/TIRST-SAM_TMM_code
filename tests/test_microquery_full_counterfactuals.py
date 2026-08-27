import torch

from efficient_sam.microquery_end2end import aggregate_soft_gated_max


def test_gate_counterfactual_changes_aggregation():
    logits = torch.tensor([[[[4.0]], [[-4.0]]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    correct = aggregate_soft_gated_max(logits, torch.tensor([[1.0, 0.0]]), valid)
    inverted = aggregate_soft_gated_max(logits, torch.tensor([[0.0, 1.0]]), valid)
    assert correct > inverted

