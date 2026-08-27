import numpy as np

from efficient_sam.microquery_end2end import assign_candidates_to_components


def test_dilation_and_centroid_assignment_with_background():
    mask = np.zeros((16, 16), np.uint8)
    mask[5:7, 5:7] = 1
    result = assign_candidates_to_components(
        np.asarray([[4.0, 5.0], [15.0, 15.0]], np.float32),
        np.asarray([True, True]),
        mask,
    )
    assert result.semantic_labels.tolist() == [True, False]
    assert result.component_ids[0] > 0 and result.component_ids[1] == -1

