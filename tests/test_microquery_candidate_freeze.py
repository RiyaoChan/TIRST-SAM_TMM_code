import torch

from efficient_sam.microquery import extract_candidate_roi_features


def test_roi_extraction_is_repeatable_and_does_not_mutate_candidates() -> None:
    shallow = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8)
    neck = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    xy = torch.tensor([[[8.0, 9.0], [20.0, 21.0]], [[4.0, 5.0], [27.0, 28.0]]])
    score = torch.tensor([[0.9, 0.4], [0.8, 0.2]])
    valid = torch.tensor([[True, False], [True, True]])
    frozen_xy = xy.clone()

    first = extract_candidate_roi_features(
        shallow, neck, xy, score, valid, input_h=32, input_w=32
    )
    second = extract_candidate_roi_features(
        shallow, neck, xy, score, valid, input_h=32, input_w=32
    )

    assert torch.equal(xy, frozen_xy)
    assert torch.equal(first, second)
    assert first.shape == (2, 2, 10)
    assert torch.count_nonzero(first[0, 1]) == 0

