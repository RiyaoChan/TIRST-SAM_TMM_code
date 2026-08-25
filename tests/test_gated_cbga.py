import unittest

import torch

from efficient_sam.text_conditioner import (
    build_gated_backbone_bifusion_block_adapter,
)


class GatedBackboneBiFusionTest(unittest.TestCase):
    def _build(self, *, delta_only: bool, gate_bias: float = -20.0):
        return build_gated_backbone_bifusion_block_adapter(
            num_layers=2,
            vision_dim=16,
            text_dim=12,
            hidden_dim=8,
            num_heads=2,
            gate_init_bias=gate_bias,
            delta_only=delta_only,
        )

    def test_delta_only_is_identity_when_gates_are_closed(self):
        torch.manual_seed(0)
        module = self._build(delta_only=True)
        # Non-zero output projections make this test independent of the
        # near-identity zero initialization.
        torch.nn.init.normal_(module.vision_out_proj.weight, std=0.1)
        torch.nn.init.normal_(module.text_out_proj.weight, std=0.1)
        torch.nn.init.constant_(module.vision_out_proj.bias, 0.5)
        torch.nn.init.constant_(module.text_out_proj.bias, -0.5)

        vision = torch.randn(2, 9, 16)
        text = torch.randn(2, 5, 12)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
            dtype=torch.bool,
        )
        vision_out, text_out, _ = module.forward_layer(vision, text, mask, 0)

        self.assertLess((vision_out - vision).abs().max().item(), 1e-6)
        valid_delta = (text_out - text) * mask.unsqueeze(-1)
        self.assertLess(valid_delta.abs().max().item(), 1e-6)
        # Padding tokens must remain untouched for every gated mode.
        padded_delta = (text_out - text) * (~mask).unsqueeze(-1)
        self.assertEqual(padded_delta.abs().max().item(), 0.0)

    def test_mode_switch_does_not_change_checkpoint_keys(self):
        legacy = self._build(delta_only=False)
        stable = self._build(delta_only=True)
        self.assertEqual(set(legacy.state_dict()), set(stable.state_dict()))
        stable.load_state_dict(legacy.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
