import numpy as np

from efficient_sam.microquery_component_safe import tri_state_group_rejection


def test_background_group_is_not_unconditionally_rescued() -> None:
    selected = tri_state_group_rejection(
        ((0, 1),), np.array([0.01, 0.02]), np.array([True, True]),
        tau_high=0.4, tau_low=0.1, tau_rescue=0.15, uncertain_weight=0.7,
    )
    assert not selected.accepted.any()
    assert selected.state == ("REJECT", "REJECT")
