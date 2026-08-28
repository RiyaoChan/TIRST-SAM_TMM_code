import torch

from efficient_sam.microquery_gate_deployment import GateDeploymentConfig, compute_deployment_gate


def test_raw_gate_is_temperature_scaled_object_probability():
    logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0], [1.0, 9.0]]])
    valid = torch.tensor([[True, True, False]])
    raw, effective, rho, temperature = compute_deployment_gate(
        logits, valid, GateDeploymentConfig("raw", temperature=1.0)
    )
    expected = torch.softmax(logits, dim=-1)[..., 1] * valid
    assert torch.allclose(raw, expected)
    assert torch.allclose(effective, expected)
    assert (rho, temperature) == (0.0, 1.0)
