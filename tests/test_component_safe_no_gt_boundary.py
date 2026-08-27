import inspect

from efficient_sam.microquery_component_safe import ComponentSafeMicroQueryHead


def test_deployable_forward_has_no_gt_arguments() -> None:
    parameters = set(inspect.signature(ComponentSafeMicroQueryHead.forward).parameters)
    assert parameters == {"self", "descriptors", "candidate_valid"}
