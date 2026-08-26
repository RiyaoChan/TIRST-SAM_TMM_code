import numpy as np

from efficient_sam.microquery_metrics import MicroQueryMetricAccumulator


def test_covered_target_recovery_and_rejection_metrics() -> None:
    gt = np.zeros((16, 16), dtype=np.uint8)
    gt[4:6, 4:6] = 1
    gt[10:12, 10:12] = 1
    query_probabilities = np.zeros((3, 16, 16), dtype=np.float32)
    query_probabilities[0, 4:6, 4:6] = 1.0
    query_probabilities[1, 4:6, 4:6] = 1.0
    query_probabilities[2, 1:3, 13:15] = 1.0
    final_probability = query_probabilities[0]
    accumulator = MicroQueryMetricAccumulator(budget=3)
    accumulator.update(
        name="sample",
        gt_mask=gt,
        candidate_xy=np.asarray([[4.5, 4.5], [5.0, 5.0], [14.0, 2.0]]),
        candidate_scores=np.asarray([0.9, 0.8, 0.7]),
        candidate_valid=np.asarray([True, True, True]),
        query_probabilities=query_probabilities,
        final_probability=final_probability,
        accepted=np.asarray([True, False, False]),
        object_scores=np.asarray([0.9, 0.2, 0.1]),
    )
    summary = accumulator.finalize()["summary"]
    assert summary["candidate_coverage"] == 0.5
    assert summary["covered_target_recovery"] == 1.0
    assert summary["target_candidate_retention"] == 1.0
    assert summary["false_candidate_rejection"] == 1.0
    assert summary["duplicate_suppression"] == 1.0
    assert summary["qmsr_at_0_5"] == 1.0

