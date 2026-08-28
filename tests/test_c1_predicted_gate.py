import torch

from efficient_sam.microquery_gate_deployment import GateDeploymentConfig
from tests.gate_runtime_test_utils import DummyHead, DummyModel, deployable, patch_runtime


def test_c1_predicted_gate_changes_mask_while_all_one_does_not(monkeypatch):
    runtime = patch_runtime(monkeypatch)
    model = DummyModel(training=False)
    first_logits = torch.tensor([[[0.0, 4.0], [4.0, 0.0]]])
    second_logits = -first_logits
    all_one_first = runtime.forward_deployable(
        model, DummyHead(first_logits), deployable(), variant="c1_independent_aux",
        gate_deployment_config=GateDeploymentConfig("all_one"),
    )
    all_one_second = runtime.forward_deployable(
        model, DummyHead(second_logits), deployable(), variant="c1_independent_aux",
        gate_deployment_config=GateDeploymentConfig("all_one"),
    )
    predicted_first = runtime.forward_deployable(
        model, DummyHead(first_logits), deployable(), variant="c1_independent_aux",
        gate_deployment_config=GateDeploymentConfig("raw"),
    )
    predicted_second = runtime.forward_deployable(
        model, DummyHead(second_logits), deployable(), variant="c1_independent_aux",
        gate_deployment_config=GateDeploymentConfig("raw"),
    )
    assert torch.allclose(all_one_first.final_probability, all_one_second.final_probability)
    assert not torch.allclose(predicted_first.final_probability, predicted_second.final_probability)
