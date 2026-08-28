from scripts.audit_microquery_gate_deployment import pareto_frontier


def test_pareto_frontier_removes_dominated_thresholds():
    rows = [
        {"threshold": 0.1, "pd": 0.9, "mean_niou": 0.5, "fa": 2.0},
        {"threshold": 0.2, "pd": 0.8, "mean_niou": 0.4, "fa": 3.0},
        {"threshold": 0.3, "pd": 0.8, "mean_niou": 0.6, "fa": 1.0},
    ]
    assert [row["threshold"] for row in pareto_frontier(rows)] == [0.1, 0.3]
