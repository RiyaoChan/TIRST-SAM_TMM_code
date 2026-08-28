import numpy as np

from scripts.audit_microquery_gate_deployment import counterfactual_rows


def test_candidate_shuffled_gate_changes_counterfactual_mask_metrics():
    query = np.zeros((1, 2, 4, 4), dtype=np.float32)
    query[0, 0, 0, 0] = 1.0
    query[0, 1, 3, 3] = 1.0
    cache = {
        "names": np.asarray(["a"]), "query_probability": query,
        "probability": query[:, 0], "gt": (query[:, 0] > 0).astype(np.uint8),
        "xy": np.zeros((1, 2, 2), np.float32), "raw": np.ones((1, 2), np.float32),
        "valid": np.ones((1, 2), bool), "semantic": np.asarray([[True, False]]),
        "gates": np.asarray([[1.0, 0.0]], np.float32), "raw_gates": np.asarray([[1.0, 0.0]], np.float32),
    }
    rows = counterfactual_rows({"cache": cache}, threshold=0.5)
    by_name = {row["condition"]: row for row in rows}
    assert by_name["correct"]["global_iou"] > by_name["candidate_shuffled"]["global_iou"]
