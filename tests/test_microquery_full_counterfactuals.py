import torch

from efficient_sam.microquery_end2end import aggregate_soft_gated_max
from scripts.eval_microquery_counterfactuals import counterfactual_effect


def test_gate_counterfactual_changes_aggregation():
    logits = torch.tensor([[[[4.0]], [[-4.0]]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    correct = aggregate_soft_gated_max(logits, torch.tensor([[1.0, 0.0]]), valid)
    inverted = aggregate_soft_gated_max(logits, torch.tensor([[0.0, 1.0]]), valid)
    assert correct > inverted


def test_counterfactual_requires_meaningful_segmentation_delta():
    correct = {
        "selected_global_iou": 0.56740,
        "selected_mean_niou": 0.48486,
        "selected_mask_auprc": 0.64479,
    }
    numerically_lower_only = {
        "condition": "zero",
        "selected_global_iou": 0.56702,
        "selected_mean_niou": 0.48484,
        "selected_mask_auprc": 0.64473,
    }
    clear_auprc_drop = {
        "condition": "all_one",
        "selected_global_iou": 0.56680,
        "selected_mean_niou": 0.48370,
        "selected_mask_auprc": 0.61000,
    }

    tiny = counterfactual_effect(correct, numerically_lower_only)
    clear = counterfactual_effect(correct, clear_auprc_drop)
    assert tiny["ordering_pass"]
    assert not tiny["meaningful_effect_pass"]
    assert clear["meaningful_effect_pass"]
