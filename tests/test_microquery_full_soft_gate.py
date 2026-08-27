import torch

from efficient_sam.microquery_end2end import soft_gate_schedule


def test_soft_gate_is_target_softmax_and_invalid_zero():
    logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
    raw, effective, _, _ = soft_gate_schedule(logits, torch.tensor([[True, False]]), 30)
    assert raw[0, 0] > 0.8
    assert raw[0, 1] == 0 and effective[0, 1] == 0

