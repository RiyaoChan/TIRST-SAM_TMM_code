import torch

from efficient_sam.microquery_end2end import microquery_full_loss
from tests.test_microquery_full_losses_finite import make_loss_inputs


def test_background_false_mask_has_positive_loss_and_gradient():
    inputs = make_loss_inputs()
    inputs["semantic_labels"] = torch.zeros(2, 3, dtype=torch.bool)
    inputs["component_ids"] = torch.full((2, 3), -1)
    losses = microquery_full_loss(variant="f1_soft_gate", **inputs)
    assert losses["background_query"] > 0
    losses["total"].backward()
    assert inputs["query_logits"].grad is not None

