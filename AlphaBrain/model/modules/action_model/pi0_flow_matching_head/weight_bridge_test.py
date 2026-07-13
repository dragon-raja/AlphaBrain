import unittest

import torch

from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.weight_bridge import (
    _adapt_action_projection,
    _fixup_vlm_keys,
)


class WeightBridgeTest(unittest.TestCase):
    def test_tied_lm_head_seeds_embedding_aliases(self) -> None:
        weight = torch.randn(5, 3)
        fixed = _fixup_vlm_keys({"vlm_interface.model.lm_head.weight": weight})

        self.assertIs(fixed["vlm_interface.model.language_model.embed_tokens.weight"], weight)
        self.assertIs(fixed["vlm_interface.model.embed_tokens.weight"], weight)

    def test_slices_action_input_projection(self) -> None:
        source = torch.arange(24).reshape(3, 8)
        target = torch.empty(3, 2)
        adapted = _adapt_action_projection("flow_matching_head.action_in_proj.weight", source, target)
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
        adapted = _adapt_action_projection("other.weight", torch.empty(3, 8), torch.empty(3, 2))
        self.assertIsNone(adapted)


if __name__ == "__main__":
    unittest.main()
