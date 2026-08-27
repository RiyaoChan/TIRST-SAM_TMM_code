import numpy as np

from scripts.train_microquery_component_safe import select_candidates


def test_random_group_membership_can_change_champion() -> None:
    semantic = np.array([[0.9, 0.8, 0.7]])
    utility = np.array([[0.9, 0.8, 0.1]])
    valid = np.ones((1, 3), dtype=bool)
    correct = select_candidates(semantic, utility, valid, [((0, 1), (2,))], 0.5, group_safe=True, use_utility=True)
    random = select_candidates(semantic, utility, valid, [((0, 2), (1,))], 0.5, group_safe=True, use_utility=True)
    assert not np.array_equal(correct, random)
