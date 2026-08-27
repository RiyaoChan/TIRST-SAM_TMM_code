import numpy as np

from efficient_sam.microquery_component_safe import semantic_label_from_roles


def test_primary_and_duplicate_are_positive_background_is_negative() -> None:
    labels = semantic_label_from_roles(
        np.array([[True, False, False]]),
        np.array([[False, True, False]]),
        np.array([[True, True, True]]),
    )
    assert labels.tolist() == [[1, 1, 0]]
