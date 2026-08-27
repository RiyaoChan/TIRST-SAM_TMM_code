import numpy as np

from efficient_sam.microquery_end2end import assign_candidates_to_components


def test_duplicate_candidates_remain_target_like():
    mask = np.zeros((12, 12), np.uint8)
    mask[4:7, 4:7] = 1
    result = assign_candidates_to_components(
        np.asarray([[4.0, 4.0], [6.0, 6.0]], np.float32),
        np.asarray([True, True]),
        mask,
    )
    assert result.semantic_labels.tolist() == [True, True]
    assert result.component_ids[0] == result.component_ids[1]

