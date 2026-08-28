import pytest

from efficient_sam.microquery_gate_deployment import GateDeploymentConfig, gate_config_id


def test_valid_configs_have_stable_ids():
    assert gate_config_id(GateDeploymentConfig("all_one")) == "all_one"
    assert gate_config_id(GateDeploymentConfig("residual", rho=0.2, temperature=1.5)) == "residual_rho0.2_t1.5"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "residual", "rho": -0.1},
        {"mode": "residual", "rho": 1.1},
        {"mode": "raw", "rho": 0.1},
        {"mode": "raw", "temperature": 0.0},
        {"mode": "legacy_checkpoint_schedule", "rho": 0.1},
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        GateDeploymentConfig(**kwargs)
