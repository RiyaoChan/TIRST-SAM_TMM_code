import numpy as np
import torch

from efficient_sam.microquery_component_safe import ComponentSafeMicroQueryHead


def test_batch_size_does_not_change_eval_output_bytes() -> None:
    torch.manual_seed(7)
    model = ComponentSafeMicroQueryHead(input_dim=5, hidden_dim=8, dropout=0.1).eval()
    descriptors = torch.randn(4, 3, 5)
    valid = torch.ones(4, 3, dtype=torch.bool)
    with torch.inference_mode():
        full = model(descriptors, valid).semantic_logits.numpy()
        split = torch.cat([model(descriptors[:2], valid[:2]).semantic_logits, model(descriptors[2:], valid[2:]).semantic_logits]).numpy()
    assert full.tobytes() == split.tobytes()
