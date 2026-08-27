import torch

from efficient_sam.microquery_end2end import EndToEndMicroQueryHead


def test_head_forward_is_deterministic_in_eval_mode():
    torch.manual_seed(20260825)
    head = EndToEndMicroQueryHead().eval()
    descriptors = torch.randn(2, 10, 451)
    valid = torch.ones(2, 10, dtype=torch.bool)
    first = head(descriptors, valid)
    second = head(descriptors, valid)
    assert torch.equal(first.object_logits, second.object_logits)
    assert torch.equal(first.candidate_token, second.candidate_token)
