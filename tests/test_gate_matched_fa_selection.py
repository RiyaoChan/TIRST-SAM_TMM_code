from scripts.audit_microquery_gate_deployment import matched_fa_selection


def test_matched_fa_maximizes_pd_then_tiny_pd_then_niou():
    rows = [
        {"threshold": 0.1, "pd": 0.9, "tiny_pd": 0.7, "fa": 2.0, "mean_niou": 0.8},
        {"threshold": 0.2, "pd": 0.9, "tiny_pd": 0.8, "fa": 2.5, "mean_niou": 0.5},
        {"threshold": 0.3, "pd": 1.0, "tiny_pd": 1.0, "fa": 4.0, "mean_niou": 1.0},
    ]
    assert matched_fa_selection(rows, fa_reference=3.0)["threshold"] == 0.2
