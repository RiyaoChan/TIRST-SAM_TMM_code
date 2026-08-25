import torch

from scripts.build_structured_role_token_features import role_text
from scripts.eval_text_counterfactuals import (
    build_derangement,
    make_condition_features,
    parse_conditions,
)


def _role_embeddings():
    values = []
    for presence, count in ((False, 0), (True, 1), (True, 2)):
        attrs = {
            "target_present": presence,
            "count": count,
            "position": "unknown",
            "size": "unknown",
        }
        values.extend((role_text("presence", attrs), role_text("count", attrs)))
    return {text: torch.full((512,), float(index + 1)) for index, text in enumerate(dict.fromkeys(values))}


def _feature_cache():
    return {
        "a": {
            "role_names": ["presence", "count"],
            "role_values": {"presence": True, "count": 1},
            "token_features": torch.stack((torch.ones(512), torch.full((512,), 2.0))),
            "attention_mask": torch.ones(2, dtype=torch.long),
        },
        "b": {
            "role_names": ["presence", "count"],
            "role_values": {"presence": False, "count": 0},
            "token_features": torch.stack((torch.full((512,), 3.0), torch.full((512,), 4.0))),
            "attention_mask": torch.ones(2, dtype=torch.long),
        },
    }


def test_parse_conditions_and_derangement_are_deterministic():
    assert parse_conditions("C,N,S,W,O") == ("C", "N", "S", "W", "O")
    names = ["a", "b", "c", "d"]
    first = build_derangement(names, seed=7)
    second = build_derangement(names, seed=7)
    assert first == second
    assert all(first[name] != name for name in names)


def test_non_oracle_conditions_reject_gt_access():
    cache = _feature_cache()
    embeddings = _role_embeddings()
    shuffle_map = {"a": "b", "b": "a"}
    gt = torch.zeros((1, 8, 8))

    for condition in ("C", "N", "S", "W"):
        try:
            make_condition_features(
                condition,
                ["a"],
                cache,
                shuffle_map,
                embeddings,
                gt_masks=gt,
            )
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{condition} unexpectedly accepted GT masks")


def test_condition_features_have_expected_semantics():
    cache = _feature_cache()
    embeddings = _role_embeddings()
    shuffle_map = {"a": "b", "b": "a"}

    c_tokens, c_mask = make_condition_features("C", ["a"], cache, shuffle_map, embeddings)
    n_tokens, n_mask = make_condition_features("N", ["a"], cache, shuffle_map, embeddings)
    s_tokens, s_mask = make_condition_features("S", ["a"], cache, shuffle_map, embeddings)
    w_tokens, w_mask = make_condition_features("W", ["a"], cache, shuffle_map, embeddings)

    assert torch.equal(c_tokens[0], cache["a"]["token_features"])
    assert torch.equal(c_mask[0], torch.ones(2, dtype=torch.long))
    assert torch.count_nonzero(n_tokens) == 0
    assert torch.count_nonzero(n_mask) == 0
    assert torch.equal(s_tokens[0], cache["b"]["token_features"])
    assert torch.equal(s_mask[0], torch.ones(2, dtype=torch.long))
    absent = {
        "target_present": False,
        "count": 0,
        "position": "unknown",
        "size": "unknown",
    }
    assert torch.equal(w_tokens[0, 0], embeddings[role_text("presence", absent)])
    assert torch.equal(w_tokens[0, 1], embeddings[role_text("count", absent)])
    assert torch.equal(w_mask[0], torch.ones(2, dtype=torch.long))


def test_oracle_uses_connected_component_count():
    cache = _feature_cache()
    embeddings = _role_embeddings()
    shuffle_map = {"a": "b", "b": "a"}
    gt = torch.zeros((1, 8, 8))
    gt[0, 1, 1] = 1
    gt[0, 6, 6] = 1

    tokens, mask = make_condition_features(
        "O",
        ["a"],
        cache,
        shuffle_map,
        embeddings,
        gt_masks=gt,
    )
    oracle = {
        "target_present": True,
        "count": 2,
        "position": "unknown",
        "size": "unknown",
    }
    assert torch.equal(tokens[0, 0], embeddings[role_text("presence", oracle)])
    assert torch.equal(tokens[0, 1], embeddings[role_text("count", oracle)])
    assert torch.equal(mask[0], torch.ones(2, dtype=torch.long))
