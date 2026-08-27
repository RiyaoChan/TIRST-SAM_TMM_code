import torch

from efficient_sam.microquery_end2end import microquery_full_loss


def make_loss_inputs():
    return dict(
        final_probability=torch.sigmoid(torch.randn(2, 4, 4, requires_grad=True)),
        full_target=torch.randint(0, 2, (2, 4, 4)).float(),
        covered_target=torch.randint(0, 2, (2, 4, 4)).float(),
        query_logits=torch.randn(2, 3, 4, 4, requires_grad=True),
        query_targets=torch.randint(0, 2, (2, 3, 4, 4)).float(),
        semantic_labels=torch.tensor([[True, False, True], [False, True, False]]),
        candidate_valid=torch.ones(2, 3, dtype=torch.bool),
        object_logits=torch.randn(2, 3, 2, requires_grad=True),
        raw_gates=torch.sigmoid(torch.randn(2, 3, requires_grad=True)),
        component_ids=torch.tensor([[1, -1, 2], [-1, 1, -1]]),
        iou_predictions=torch.randn(2, 3, requires_grad=True),
        class_weights=torch.tensor([1.0, 2.0]),
        candidate_tokens=torch.randn(2, 3, 256, requires_grad=True),
    )


def test_all_variant_losses_are_finite():
    inputs = make_loss_inputs()
    for variant in ("c1_independent_aux", "f1_soft_gate", "f2_gate_token"):
        losses = microquery_full_loss(variant=variant, **inputs)
        assert all(torch.isfinite(value) for value in losses.values())

