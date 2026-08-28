from scripts.audit_microquery_gate_deployment import matched_pd_selection


def test_matched_pd_minimizes_fa_then_maximizes_niou_and_iou():
    rows = [
        {"threshold": 0.1, "pd": 0.90, "fa": 3.0, "mean_niou": 0.4, "global_iou": 0.5},
        {"threshold": 0.2, "pd": 0.90, "fa": 2.0, "mean_niou": 0.5, "global_iou": 0.4},
        {"threshold": 0.3, "pd": 0.89, "fa": 1.0, "mean_niou": 0.9, "global_iou": 0.9},
    ]
    assert matched_pd_selection(rows, pd_floor=0.90)["threshold"] == 0.2
