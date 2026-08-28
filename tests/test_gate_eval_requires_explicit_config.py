import pytest

from tests.gate_runtime_test_utils import DummyHead, DummyModel, deployable, patch_runtime


def test_eval_forward_without_deployment_config_raises(monkeypatch):
    runtime = patch_runtime(monkeypatch)
    with pytest.raises(ValueError, match="explicit gate_deployment_config"):
        runtime.forward_deployable(
            DummyModel(training=False), DummyHead(), deployable(), variant="c1_independent_aux"
        )
