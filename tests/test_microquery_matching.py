import numpy as np

from efficient_sam.microquery_metrics import (
    ASSIGNMENT_BACKGROUND,
    ASSIGNMENT_DUPLICATE,
    ASSIGNMENT_PRIMARY,
    assign_candidates,
)
from efficient_sam.prompt_metrics import extract_components


def test_primary_duplicate_and_background_are_separate() -> None:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:6, 4:6] = 1
    mask[10:12, 10:12] = 1
    components = extract_components(mask)
    result = assign_candidates(
        np.asarray([[4.5, 4.5], [5.0, 5.0], [10.5, 10.5], [1.0, 14.0]]),
        np.asarray([True, True, True, True]),
        components,
        budget=4,
    )
    assert result.assignment == (
        ASSIGNMENT_PRIMARY,
        ASSIGNMENT_DUPLICATE,
        ASSIGNMENT_PRIMARY,
        ASSIGNMENT_BACKGROUND,
    )
    assert result.covered_components == frozenset({0, 1})
    assert result.best_rank_by_component == {0: 1, 1: 3}

