from scripts.audit_microquery_gate_deployment import paired_bootstrap


def _rows(delta=0):
    return [
        {
            "image": str(index), "intersection_pixels": 2 + delta, "union_pixels": 4,
            "iou": (2 + delta) / 4, "f1": 0.5 + delta / 10,
            "detected_components": 1, "target_components": 1,
            "false_pixels": 1 - delta, "pixels": 16,
            "tiny_detected_components": 1, "tiny_target_components": 1,
        }
        for index in range(4)
    ]


def test_paired_bootstrap_is_seed_deterministic():
    first = paired_bootstrap(_rows(1), _rows(0), 50, 123)
    second = paired_bootstrap(_rows(1), _rows(0), 50, 123)
    assert first == second
