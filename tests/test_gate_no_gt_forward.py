import inspect

from scripts.microquery_end2end_runtime import forward_deployable


def test_gate_deployment_forward_has_no_gt_or_supervision_argument():
    parameters = set(inspect.signature(forward_deployable).parameters)
    assert not parameters & {"gt", "mask", "target", "component", "semantic", "assignment", "supervision"}
