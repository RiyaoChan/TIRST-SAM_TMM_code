import torch

from efficient_sam.prompt_training import (
    SpatialProbeHead,
    component_center_heatmap,
    prompt_probe_loss,
)


def test_center_heatmap_has_one_peak_per_component():
    masks = torch.zeros(1, 32, 32)
    masks[0, 2:4, 3:5] = 1
    masks[0, 20:25, 21:26] = 1
    target = component_center_heatmap(masks)
    assert target.shape == (1, 1, 32, 32)
    assert target[0, 0, 2:4, 3:5].max() > 0.9
    assert target[0, 0, 20:25, 21:26].max() > 0.9


def test_probe_loss_is_finite_and_backpropagates():
    head = SpatialProbeHead(16, hidden_channels=16)
    features = torch.randn(2, 16, 8, 8)
    masks = torch.zeros(2, 32, 32)
    masks[0, 5:7, 6:8] = 1
    masks[1, 20, 21] = 1
    logits = head(features, output_size=(32, 32))
    losses = prompt_probe_loss(logits, masks)
    assert set(losses) == {"total", "center_focal", "foreground_dice", "component_peak"}
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_empty_mask_component_loss_remains_finite():
    head = SpatialProbeHead(8, hidden_channels=8)
    logits = head(torch.randn(1, 8, 4, 4), output_size=(16, 16))
    losses = prompt_probe_loss(logits, torch.zeros(1, 16, 16))
    assert torch.isfinite(losses["total"])
