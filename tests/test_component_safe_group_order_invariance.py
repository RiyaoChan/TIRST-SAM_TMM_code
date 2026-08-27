import numpy as np

from efficient_sam.microquery_component_safe import GroupingConfig, build_candidate_graph, connected_candidate_groups, pairwise_candidate_relations


def _coordinate_sets(xy: np.ndarray) -> set[frozenset[tuple[float, float]]]:
    masks = np.zeros((len(xy), 32, 32), dtype=np.float32)
    relations = pairwise_candidate_relations(xy, masks)
    graph = build_candidate_graph(relations, np.ones(len(xy), bool), GroupingConfig("coordinate", r_xy=2))
    return {frozenset(tuple(xy[index]) for index in group) for group in connected_candidate_groups(graph, np.ones(len(xy), bool))}


def test_candidate_permutation_preserves_groups() -> None:
    xy = np.array([[1, 1], [2, 1], [20, 20]], dtype=np.float32)
    assert _coordinate_sets(xy) == _coordinate_sets(xy[[2, 0, 1]])
