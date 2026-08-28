from scripts.audit_microquery_gate_deployment import AREA_ORDER, enriched_area_bins


def test_area_bins_report_safety_and_gate_statistics():
    components = [
        {"image": "a", "component_index": 0, "area_bin": "1-9", "covered": 1, "final_detected": 1, "best_query_iou": 0.6},
        {"image": "a", "component_index": 1, "area_bin": ">25", "covered": 0, "final_detected": 0, "best_query_iou": 0.1},
    ]
    queries = [
        {"image": "a", "component_index": 0, "assignment": "primary", "object_score": 0.4}
    ]
    rows = enriched_area_bins(components, queries)
    assert [row["area_bin"] for row in rows] == list(AREA_ORDER)
    tiny = rows[0]
    assert tiny["components"] == 1 and tiny["pd"] == 1.0
    assert tiny["target_gate_below_0_5"] == 1
