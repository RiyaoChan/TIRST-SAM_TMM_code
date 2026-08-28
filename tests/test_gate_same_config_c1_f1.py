import torch

from efficient_sam.microquery_gate_deployment import GateDeploymentConfig
from tests.gate_runtime_test_utils import DummyHead, DummyModel, deployable, patch_runtime


def test_same_explicit_config_has_same_gate_formula_for_c1_and_f1(monkeypatch):
    runtime = patch_runtime(monkeypatch)
    config = GateDeploymentConfig("residual", rho=0.2, temperature=1.5)
    c1 = runtime.forward_deployable(
        DummyModel(False), DummyHead(), deployable(), variant="c1_independent_aux", gate_deployment_config=config
    )
    f1 = runtime.forward_deployable(
        DummyModel(False), DummyHead(), deployable(), variant="f1_soft_gate", gate_deployment_config=config
    )
    assert torch.allclose(c1.raw_gates, f1.raw_gates)
    assert torch.allclose(c1.effective_gates, f1.effective_gates)
