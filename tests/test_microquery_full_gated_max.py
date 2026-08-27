import torch

from efficient_sam.microquery_end2end import aggregate_soft_gated_max


def test_gated_max_matches_manual_pixelwise_max():
    logits = torch.tensor([[[[0.0]], [[2.0]]]])
    gates = torch.tensor([[1.0, 0.25]])
    output = aggregate_soft_gated_max(logits, gates, torch.tensor([[True, True]]))
    expected = max(0.5, float(torch.sigmoid(torch.tensor(2.0))) * 0.25)
    assert torch.allclose(output, torch.tensor([[[expected]]]))

