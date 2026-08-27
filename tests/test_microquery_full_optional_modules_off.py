from types import SimpleNamespace

import pytest

from scripts.microquery_end2end_runtime import assert_optional_modules_off


def make_model(adapter=False):
    return SimpleNamespace(
        image_encoder=SimpleNamespace(use_adapter=adapter),
        prompt_encoder=SimpleNamespace(task_tokens=None),
        mask_decoder=SimpleNamespace(
            use_amgd=False, use_dog_amgd=False, use_center_mask_decoder=False
        ),
        use_ms_fusion=False,
        use_detail_enhancer=False,
        use_hldf=False,
    )


def test_optional_modules_are_rejected():
    assert_optional_modules_off(make_model())
    with pytest.raises(RuntimeError):
        assert_optional_modules_off(make_model(adapter=True))

