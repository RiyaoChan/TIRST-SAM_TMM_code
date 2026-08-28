from types import SimpleNamespace

import torch

from efficient_sam.microquery_end2end import EndToEndHeadOutput


class DummyModel:
    def __init__(self, training=False):
        self.training = training


class DummyHead:
    def __init__(self, logits=None):
        self.logits = logits

    def __call__(self, descriptors, valid):
        batch, count = valid.shape
        logits = self.logits
        if logits is None:
            logits = torch.tensor([0.0, 2.0], dtype=descriptors.dtype).repeat(batch, count, 1)
        logits = logits.to(descriptors.device, descriptors.dtype)
        hidden = torch.zeros(batch, count, 256, device=descriptors.device, dtype=descriptors.dtype)
        return EndToEndHeadOutput(hidden, logits, hidden)


def deployable(count=2):
    return {
        "image": torch.ones(1, 3, 4, 4),
        "candidate_xy": torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])[:, :count],
        "candidate_scores": torch.ones(1, count),
        "candidate_valid": torch.ones(1, count, dtype=torch.bool),
    }


def patch_runtime(monkeypatch):
    import scripts.microquery_end2end_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "encode_frozen_image",
        lambda model, images: (torch.zeros(1, 1, 1, 1),) * 3,
    )
    monkeypatch.setattr(
        runtime,
        "extract_candidate_roi_features",
        lambda shallow, neck, xy, scores, valid, **kwargs: torch.zeros(
            *valid.shape, 451, dtype=xy.dtype
        ),
    )

    def decode(model, neck, interm, points, labels, **kwargs):
        count = points.shape[1]
        values = torch.arange(1, count + 1, dtype=torch.float32).reshape(1, count, 1, 1)
        return values.expand(1, count, 4, 4), torch.zeros(1, count)

    monkeypatch.setattr(runtime, "_decode_chunks", decode)
    return runtime
