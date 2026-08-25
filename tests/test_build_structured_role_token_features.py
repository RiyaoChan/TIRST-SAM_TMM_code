from __future__ import annotations

import torch

from scripts.build_structured_role_token_features import (
    assemble_cache_item,
    audited_field_mask,
    normalize_attributes,
    parse_roles,
    role_text,
)


def test_parse_roles_rejects_duplicates() -> None:
    try:
        parse_roles("presence,presence")
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("duplicate roles should fail")


def test_training_count_conflict_keeps_presence_only() -> None:
    attributes = normalize_attributes(
        {"target_present": True, "count": 2, "position": "multiple-regions", "size": "tiny"}
    )
    mask, policy = audited_field_mask(
        attributes,
        {
            "field_status": {
                "presence": "pass",
                "count": "conflict",
                "location": "pass",
                "size": "pass",
            }
        },
    )
    assert policy == "presence_only_auto"
    assert mask == {"presence": 1, "count": 0, "location": 1, "size": 1}


def test_presence_conflict_masks_every_role() -> None:
    attributes = normalize_attributes(
        {"target_present": True, "count": 1, "position": "center", "size": "tiny"}
    )
    mask, policy = audited_field_mask(
        attributes,
        {"field_status": {"presence": "conflict", "count": "blocked_by_presence"}},
    )
    assert policy == "reject_auto"
    assert not any(mask.values())


def test_inference_record_uses_raw_gpt_without_audit() -> None:
    attributes = normalize_attributes(
        {"target_present": True, "count": 2, "position": "upper-right", "size": "small"}
    )
    mask, policy = audited_field_mask(attributes, None)
    assert policy == "raw_gpt_inference"
    assert mask == {"presence": 1, "count": 1, "location": 1, "size": 1}


def test_role_item_is_field_isolated_and_global_is_masked_mean() -> None:
    roles = ("presence", "count")
    attributes = normalize_attributes(
        {"target_present": True, "count": 2, "position": "center", "size": "tiny"}
    )
    embeddings = {
        role_text("presence", attributes): torch.tensor([3.0, 0.0, 0.0]),
        role_text("count", attributes): torch.tensor([0.0, 4.0, 0.0]),
    }
    item = assemble_cache_item(
        roles=roles,
        attributes=attributes,
        field_mask={"presence": 1, "count": 0},
        text_embeddings=embeddings,
        store_dtype=torch.float32,
        policy="presence_only_auto",
    )
    assert item["token_features"].shape == (2, 3)
    assert item["attention_mask"].tolist() == [1, 0]
    assert torch.equal(item["token_features"][0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(item["token_features"][1], torch.zeros(3))
    assert torch.equal(item["global_feat"], torch.tensor([1.0, 0.0, 0.0]))


def test_verified_negative_has_presence_and_zero_count_roles() -> None:
    attributes = normalize_attributes(
        {"target_present": False, "count": 9, "position": "none", "size": "none"}
    )
    mask, policy = audited_field_mask(
        attributes,
        {
            "field_status": {
                "presence": "pass",
                "count": "not_applicable",
                "location": "not_applicable",
                "size": "not_applicable",
            }
        },
    )
    assert attributes["count"] == 0
    assert policy == "presence_count_verified_auto"
    assert mask["presence"] == 1 and mask["count"] == 1
    assert mask["location"] == 0 and mask["size"] == 0
