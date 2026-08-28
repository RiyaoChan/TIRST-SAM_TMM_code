import torch

from efficient_sam.microquery_gate_deployment import GateDeploymentConfig, compute_deployment_gate


def test_residual_gate_applies_explicit_floor_only_to_valid_candidates():
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, False]])
    raw, effective, rho, temperature = compute_deployment_gate(
        logits, valid, GateDeploymentConfig("residual", rho=0.2, temperature=1.5)
    )
    assert torch.allclose(raw, torch.tensor([[0.5, 0.0]]))
    assert torch.allclose(effective, torch.tensor([[0.6, 0.0]]))
    assert (rho, temperature) == (0.2, 1.5)
