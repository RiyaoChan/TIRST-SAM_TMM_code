import torch

from efficient_sam.microquery_end2end import EndToEndMicroQueryHead, microquery_full_loss
from tests.test_microquery_full_losses_finite import make_loss_inputs


def test_object_and_token_branches_receive_expected_gradients():
    head = EndToEndMicroQueryHead(dropout=0.0)
    valid = torch.ones(2, 3, dtype=torch.bool)
    output = head(torch.randn(2, 3, 451), valid)
    inputs = make_loss_inputs()
    inputs.update(
        object_logits=output.object_logits,
        raw_gates=torch.softmax(output.object_logits, -1)[..., 1],
        candidate_tokens=output.candidate_token,
        candidate_valid=valid,
    )
    loss = microquery_full_loss(variant="f2_gate_token", **inputs)["total"]
    loss.backward()
    assert head.object_head.weight.grad is not None
    assert head.object_head.weight.grad.abs().sum() > 0
    assert head.token_scale.grad is not None and head.token_scale.grad.abs() > 0

