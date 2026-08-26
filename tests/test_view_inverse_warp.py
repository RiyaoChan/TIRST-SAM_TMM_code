import torch

from efficient_sam.multiview_prompt import apply_view, inverse_warp_map, inverse_warp_xy


def test_flip_coordinate_round_trip_is_exact():
    xy = torch.tensor([[0.0, 0.0], [3.0, 5.0], [7.0, 9.0]])
    for view in ("hflip", "vflip"):
        transformed = inverse_warp_xy(xy, view, width=8, height=10)
        restored = inverse_warp_xy(transformed, view, width=8, height=10)
        assert torch.equal(restored, xy)


def test_map_inverse_and_identity_are_exact():
    values = torch.arange(48).reshape(1, 1, 6, 8)
    assert torch.equal(inverse_warp_map(values, "identity"), values)
    for view in ("hflip", "vflip"):
        transformed = apply_view(values, view)
        assert torch.equal(inverse_warp_map(transformed, view), values)

