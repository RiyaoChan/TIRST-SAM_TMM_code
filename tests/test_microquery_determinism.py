import io

import torch

from efficient_sam.microquery import MicroQueryHead


def _serialized_prediction(seed: int) -> bytes:
    torch.manual_seed(seed)
    head = MicroQueryHead(input_dim=7, hidden_dim=12, dropout=0.0).eval()
    descriptors = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7) / 42.0
    valid = torch.tensor([[True, True, False], [True, False, True]])
    with torch.inference_mode():
        output = head(descriptors, valid)
    stream = io.BytesIO()
    torch.save(
        {"object": output.object_logits, "quality": output.quality_logits}, stream
    )
    return stream.getvalue()


def test_same_seed_produces_byte_identical_predictions() -> None:
    assert _serialized_prediction(17) == _serialized_prediction(17)
