import torch
from torch import nn

from scripts.microquery_end2end_runtime import configure_trainable_modules


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(2, 2)
        self.prompt_encoder = nn.Linear(2, 2)
        self.mask_decoder = nn.Linear(2, 2)
        self.unused = nn.Linear(2, 2)


def test_only_prompt_and_decoder_are_trainable():
    model = TinyModel()
    configure_trainable_modules(model)
    assert not any(p.requires_grad for p in model.image_encoder.parameters())
    assert not any(p.requires_grad for p in model.unused.parameters())
    assert all(p.requires_grad for p in model.prompt_encoder.parameters())
    assert all(p.requires_grad for p in model.mask_decoder.parameters())

