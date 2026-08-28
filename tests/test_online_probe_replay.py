import numpy as np

from scripts.eval_microquery_online_probe import coordinate_replay_metrics


def test_online_replay_coordinate_tolerances_and_rank_agreement():
    cached = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], np.float32)
    online = cached + np.asarray([[[0.3, 0.0], [0.0, 0.8]]], np.float32)
    valid = np.ones((1, 2), bool)
    metrics = coordinate_replay_metrics(cached, np.ones((1, 2)), valid, online, np.ones((1, 2)), valid)
    assert metrics["coordinate_within_0_5_fraction"] == 0.5
    assert metrics["coordinate_within_1_0_fraction"] == 1.0
    assert metrics["rank_agreement_within_1px"] == 1.0
