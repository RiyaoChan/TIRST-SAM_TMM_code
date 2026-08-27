from efficient_sam.microquery_end2end import gate_temperature, gate_warmup_rho


def test_gate_warmup_and_temperature_schedules():
    assert gate_warmup_rho(1) == 0.95
    assert gate_warmup_rho(20) == 0.0
    assert gate_warmup_rho(100) == 0.0
    assert gate_temperature(1) == 2.0
    assert gate_temperature(30) == 1.0
    assert gate_temperature(100) == 1.0

