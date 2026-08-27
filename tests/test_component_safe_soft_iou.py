import numpy as np

from efficient_sam.microquery_component_safe import soft_mask_iou


def test_soft_iou_is_symmetric_and_bounded() -> None:
    masks = np.zeros((3, 5, 5), dtype=np.float32)
    masks[0, 1:3, 1:3] = 1
    masks[1, 2:4, 2:4] = 1
    result = soft_mask_iou(masks)
    assert np.array_equal(result, result.T)
    assert np.all((0 <= result) & (result <= 1))
    assert np.array_equal(np.diag(result), np.ones(3))
