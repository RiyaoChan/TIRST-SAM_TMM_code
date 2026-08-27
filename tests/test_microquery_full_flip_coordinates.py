import torch

from scripts.microquery_end2end_dataset import flip_image_mask_coordinates


def test_flip_transforms_image_mask_and_xy_together():
    image = torch.arange(16).reshape(1, 4, 4).float()
    mask = torch.zeros(4, 4)
    mask[1, 2] = 1
    xy = torch.tensor([[2.0, 1.0]])
    flipped_image, flipped_mask, flipped_xy = flip_image_mask_coordinates(
        image, mask, xy, horizontal=True, vertical=True
    )
    assert flipped_mask[2, 1] == 1
    assert torch.equal(flipped_xy, torch.tensor([[1.0, 2.0]]))
    assert flipped_image[0, 0, 0] == image[0, 3, 3]

