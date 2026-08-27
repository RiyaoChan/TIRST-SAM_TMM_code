import numpy as np

from efficient_sam.microquery_component_safe import connected_candidate_groups, select_group_champions


def test_empty_candidate_image_is_supported() -> None:
    valid = np.zeros(3, dtype=bool)
    assert connected_candidate_groups(np.zeros((3, 3), dtype=bool), valid) == ()
    selected = select_group_champions((), np.zeros(3), valid, tau_high=0.2, tau_group=0.1)
    assert not selected.accepted.any()
