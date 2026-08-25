import torch

from scripts.build_null_role_token_features import build_null_cache, null_cache_item


def _item():
    return {
        "token_features": torch.randn(2, 512, dtype=torch.float16),
        "global_feat": torch.randn(512, dtype=torch.float16),
        "attention_mask": torch.ones(2, dtype=torch.long),
        "field_mask": torch.tensor([1, 0], dtype=torch.long),
        "role_names": ["presence", "count"],
        "role_values": {"presence": True, "count": 1},
        "schema_version": "source-v1",
    }


def test_null_cache_item_preserves_shape_dtype_and_metadata():
    source = _item()
    output = null_cache_item(source)

    assert output["token_features"].shape == (2, 512)
    assert output["token_features"].dtype == torch.float16
    assert output["global_feat"].shape == (512,)
    assert output["global_feat"].dtype == torch.float16
    assert torch.count_nonzero(output["token_features"]) == 0
    assert torch.count_nonzero(output["global_feat"]) == 0
    assert torch.count_nonzero(output["attention_mask"]) == 0
    assert torch.count_nonzero(output["field_mask"]) == 0
    assert output["role_names"] == source["role_names"]
    assert output["role_values"] == source["role_values"]
    assert output["source_schema_version"] == "source-v1"
    assert output["schema_version"] == "gpt-structured-role-tokens-null-control-v1"


def test_build_null_cache_transforms_every_entry_without_mutating_source():
    source = {"a": _item(), "b": _item()}
    output = build_null_cache(source)

    assert set(output) == {"a", "b"}
    assert torch.count_nonzero(output["a"]["token_features"]) == 0
    assert torch.count_nonzero(output["b"]["attention_mask"]) == 0
    assert torch.count_nonzero(source["a"]["token_features"]) > 0
