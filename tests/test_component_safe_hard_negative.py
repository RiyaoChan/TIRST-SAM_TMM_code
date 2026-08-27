import numpy as np

from scripts.build_microquery_hard_negative_split import crop_is_background, dilate_gt, enumerate_background_crops


def test_hard_negative_crop_excludes_dilated_gt() -> None:
    image = np.zeros((300, 600), dtype=np.uint8)
    mask = np.zeros_like(image, dtype=bool)
    mask[120, 100] = True
    crops = enumerate_background_crops(image, mask, crop_size=256, dilation_pixels=12, stride=32)
    assert crops
    dilated_source = dilate_gt(mask, 12)
    assert all(crop_is_background(dilated_source, row["left"], row["top"], 256) for row in crops)
    dilated = np.zeros((256, 256), dtype=bool)
    dilated[2, 2] = True
    assert not crop_is_background(dilated, 0, 0, 256)
