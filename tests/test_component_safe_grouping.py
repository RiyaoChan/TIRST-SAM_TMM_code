import numpy as np

from efficient_sam.microquery_component_safe import (
    GroupingConfig, build_candidate_graph, connected_candidate_groups,
    pairwise_candidate_relations,
)


def test_hybrid_grouping_joins_near_candidates() -> None:
    xy = np.array([[2, 2], [3, 2], [20, 20]], dtype=np.float32)
    masks = np.zeros((3, 24, 24), dtype=np.float32)
    masks[0:2, 1:5, 1:5] = 1
    masks[2, 19:22, 19:22] = 1
    relations = pairwise_candidate_relations(xy, masks)
    graph = build_candidate_graph(relations, np.ones(3, bool), GroupingConfig("hybrid", r_near=2, r_far=8, tau_iou=0.2, r_mask=5))
    assert connected_candidate_groups(graph, np.ones(3, bool)) == ((0, 1), (2,))
