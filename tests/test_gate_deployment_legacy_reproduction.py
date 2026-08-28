import pytest
import torch

from efficient_sam.microquery_end2end import soft_gate_schedule
from efficient_sam.microquery_gate_deployment import GateDeploymentConfig, compute_deployment_gate


@pytest.mark.parametrize("epoch,expected_rho,expected_temperature", [(18, 0.1, 2 - 17 / 29), (16, 0.2, 2 - 15 / 29)])
def test_legacy_predicted_gate_reproduces_historical_schedule(epoch, expected_rho, expected_temperature):
    logits = torch.tensor([[[0.3, 1.2], [1.0, -0.5]]])
    valid = torch.tensor([[True, True]])
    legacy = compute_deployment_gate(
        logits, valid, GateDeploymentConfig("legacy_checkpoint_schedule"), checkpoint_epoch=epoch
    )
    historical = soft_gate_schedule(logits, valid, epoch)
    assert torch.allclose(legacy[0], historical[0])
    assert torch.allclose(legacy[1], historical[1])
    assert legacy[2] == pytest.approx(expected_rho)
    assert legacy[3] == pytest.approx(expected_temperature)
