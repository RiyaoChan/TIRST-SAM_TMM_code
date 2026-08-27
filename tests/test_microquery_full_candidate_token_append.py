import torch

from efficient_sam.microquery_end2end import append_candidate_sparse_token


def test_candidate_token_is_appended_after_native_point_tokens():
    sparse = torch.randn(3, 2, 256)
    token = torch.randn(3, 256)
    output = append_candidate_sparse_token(sparse, token)
    assert output.shape == (3, 3, 256)
    assert torch.equal(output[:, :2], sparse)
    assert torch.equal(output[:, 2], token)

