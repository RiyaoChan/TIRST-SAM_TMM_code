import torch

from efficient_sam.microquery_end2end import component_survival_loss


def test_survival_loss_rewards_at_least_one_high_gate():
    ids = torch.tensor([[1, 1, -1]])
    semantic = torch.tensor([[True, True, False]])
    valid = torch.ones_like(semantic)
    low = component_survival_loss(torch.tensor([[0.1, 0.1, 0.0]]), ids, semantic, valid)
    high = component_survival_loss(torch.tensor([[0.9, 0.1, 0.0]]), ids, semantic, valid)
    assert high < low

