import numpy as np

from efficient_sam.microquery_component_safe import select_group_champions


def test_champion_is_valid_and_zero_queries_are_allowed() -> None:
    groups = ((0, 1), (2,))
    scores = np.array([0.1, 0.9, 0.1], dtype=np.float32)
    valid = np.array([True, False, True])
    selected = select_group_champions(groups, scores, valid, tau_high=0.5, tau_group=0.15)
    assert not selected.accepted.any()
    assert not selected.accepted[1]
