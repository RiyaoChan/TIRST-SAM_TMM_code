import inspect

from scripts.microquery_end2end_runtime import forward_deployable


def test_deployable_forward_has_no_gt_argument():
    parameters = set(inspect.signature(forward_deployable).parameters)
    forbidden = {"gt", "mask", "target", "component", "semantic", "assignment"}
    assert not parameters & forbidden

