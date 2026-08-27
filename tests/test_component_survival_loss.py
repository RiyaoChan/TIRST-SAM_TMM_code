import torch

from efficient_sam.microquery_component_safe import component_survival_loss


def test_all_low_component_has_larger_gradient_than_one_high_candidate() -> None:
    component = torch.tensor([[0, 0]])
    target = torch.tensor([[True, True]])
    valid = torch.tensor([[True, True]])
    low = torch.tensor([[-4.0, -4.0]], requires_grad=True)
    high = torch.tensor([[4.0, -4.0]], requires_grad=True)
    low_loss = component_survival_loss(low, component, target, valid)
    high_loss = component_survival_loss(high, component, target, valid)
    low_loss.backward()
    high_loss.backward()
    assert low_loss > high_loss
    assert low.grad.abs().sum() > high.grad.abs().sum()
