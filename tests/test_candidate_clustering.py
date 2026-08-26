import torch

from efficient_sam.multiview_prompt import cluster_candidates


def _signature(clusters):
    return [
        (
            tuple(round(value, 5) for value in cluster.center_xy),
            round(cluster.mean_score, 5),
            cluster.support_count,
        )
        for cluster in clusters
    ]


def test_clustering_is_invariant_to_candidate_order():
    xy = torch.tensor([[[10.0, 10.0], [30.0, 30.0]], [[11.0, 10.0], [29.0, 30.0]]])
    scores = torch.tensor([[0.9, 0.6], [0.8, 0.7]])
    valid = torch.ones((2, 2), dtype=torch.bool)
    first = cluster_candidates(xy, scores, valid, radius=3.0)
    second = cluster_candidates(xy.flip(1), scores.flip(1), valid.flip(1), radius=3.0)
    assert _signature(first) == _signature(second)


def test_same_view_contributes_only_once_to_cluster_support():
    xy = torch.tensor([[[10.0, 10.0], [11.0, 10.0]], [[10.5, 10.0], [40.0, 40.0]]])
    scores = torch.tensor([[0.9, 0.8], [0.7, 0.1]])
    valid = torch.ones((2, 2), dtype=torch.bool)
    clusters = cluster_candidates(xy, scores, valid, radius=3.0)
    target_cluster = min(clusters, key=lambda item: abs(item.center_xy[0] - 10.0))
    assert target_cluster.support_count == 2
    assert len(target_cluster.observations) == 2


def test_zero_candidate_is_supported():
    xy = torch.zeros((5, 4, 2))
    scores = torch.zeros((5, 4))
    valid = torch.zeros((5, 4), dtype=torch.bool)
    assert cluster_candidates(xy, scores, valid) == []

