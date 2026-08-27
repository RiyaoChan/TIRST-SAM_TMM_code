import torch

from efficient_sam.microquery_end2end import build_covered_gt_mask


def test_covered_mask_is_union_of_assigned_components():
    components = torch.tensor([[0, 1, 1], [2, 2, 0]])
    covered = build_covered_gt_mask(components, torch.tensor([2, 2, -1]))
    assert covered.sum() == 2
    assert not covered[0, 1]

