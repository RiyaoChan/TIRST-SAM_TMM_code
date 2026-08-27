def test_collision_definition_detects_multiple_components() -> None:
    component_by_candidate = {0: 2, 1: 2, 2: 7}
    safe_group = (0, 1)
    collision_group = (0, 2)
    assert len({component_by_candidate[index] for index in safe_group}) == 1
    assert len({component_by_candidate[index] for index in collision_group}) > 1
