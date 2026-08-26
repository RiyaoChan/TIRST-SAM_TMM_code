import inspect

import torch

from efficient_sam.microquery import MicroQueryHead


def test_deployable_head_has_no_gt_argument() -> None:
    parameters = set(inspect.signature(MicroQueryHead.forward).parameters)
    assert parameters == {"self", "descriptors", "candidate_valid"}
    assert not any("gt" in name.lower() or "mask" in name.lower() for name in parameters)


def test_deployable_head_accepts_only_features_and_validity() -> None:
    head = MicroQueryHead(input_dim=6, hidden_dim=8, dropout=0.0).eval()
    output = head(torch.ones(2, 3, 6), torch.ones(2, 3, dtype=torch.bool))
    assert output.object_logits.shape == (2, 3, 2)
    assert output.quality_logits.shape == (2, 3)

