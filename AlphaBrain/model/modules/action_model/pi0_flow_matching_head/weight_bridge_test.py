from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.weight_bridge import (
    _adapt_action_projection,
    _fixup_vlm_keys,
    load_pi0_weights,
)


class _NativeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flow_matching_head = torch.nn.Linear(3, 2)
        self.register_buffer("action_mean", torch.zeros(2))
        self.register_buffer("action_std", torch.ones(2))


class _OpenPiModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flow_matching_head = torch.nn.Module()
        self.flow_matching_head.action_in_proj = torch.nn.Linear(3, 2)


class WeightBridgeTest(unittest.TestCase):
    def test_tied_lm_head_seeds_embedding_aliases(self) -> None:
        weight = torch.randn(5, 3)
        fixed = _fixup_vlm_keys({"vlm_interface.model.lm_head.weight": weight})

        self.assertIs(fixed["vlm_interface.model.language_model.embed_tokens.weight"], weight)
        self.assertIs(fixed["vlm_interface.model.embed_tokens.weight"], weight)

    def test_slices_action_input_projection(self) -> None:
        source = torch.arange(24).reshape(3, 8)
        target = torch.empty(3, 2)
        adapted = _adapt_action_projection(
            "flow_matching_head.action_in_proj.weight", source, target
        )
        self.assertTrue(torch.equal(adapted, source[:, :2]))

    def test_slices_action_output_projection(self) -> None:
        source_weight = torch.arange(24).reshape(8, 3)
        target_weight = torch.empty(2, 3)
        adapted_weight = _adapt_action_projection(
            "flow_matching_head.action_out_proj.weight", source_weight, target_weight
        )
        self.assertTrue(torch.equal(adapted_weight, source_weight[:2]))

        source_bias = torch.arange(8)
        target_bias = torch.empty(2)
        adapted_bias = _adapt_action_projection(
            "flow_matching_head.action_out_proj.bias", source_bias, target_bias
        )
        self.assertTrue(torch.equal(adapted_bias, source_bias[:2]))

    def test_rejects_unrelated_shape_mismatch(self) -> None:
        adapted = _adapt_action_projection(
            "other.weight", torch.empty(3, 8), torch.empty(3, 2)
        )
        self.assertIsNone(adapted)

    def test_loads_alphabrain_native_checkpoint_without_remapping(self) -> None:
        source = _NativeModel()
        with torch.no_grad():
            source.flow_matching_head.weight.fill_(0.25)
            source.flow_matching_head.bias.fill_(-0.5)
            source.action_mean.copy_(torch.tensor([1.0, 2.0]))
            source.action_std.copy_(torch.tensor([3.0, 4.0]))

        target = _NativeModel()
        with torch.no_grad():
            target.flow_matching_head.weight.zero_()
            target.flow_matching_head.bias.zero_()

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.safetensors"
            save_file(dict(source.state_dict()), checkpoint)
            summary = load_pi0_weights(target, str(checkpoint), verbose=False)

        self.assertEqual(summary["source_format"], "alphabrain_native")
        self.assertEqual(summary["direct_coverage"], 1.0)
        self.assertEqual(set(summary["matched"]), set(source.state_dict()))
        self.assertFalse(summary["missing"])
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value)

    def test_keeps_openpi_projection_mapping(self) -> None:
        target = _OpenPiModel()
        weight = torch.full_like(target.flow_matching_head.action_in_proj.weight, 0.75)
        bias = torch.full_like(target.flow_matching_head.action_in_proj.bias, -0.25)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.safetensors"
            save_file(
                {
                    "action_in_proj.weight": weight,
                    "action_in_proj.bias": bias,
                },
                checkpoint,
            )
            summary = load_pi0_weights(target, str(checkpoint), verbose=False)

        self.assertEqual(summary["source_format"], "openpi_bridge")
        torch.testing.assert_close(target.flow_matching_head.action_in_proj.weight, weight)
        torch.testing.assert_close(target.flow_matching_head.action_in_proj.bias, bias)


if __name__ == "__main__":
    unittest.main()
