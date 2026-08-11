import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.weight_bridge import (
    _fixup_vlm_keys,
    load_pi0_weights,
)


class WeightBridgeTest(unittest.TestCase):
    def test_tied_lm_head_populates_embedding_aliases(self) -> None:
        weight = torch.randn(5, 3)
        fixed = _fixup_vlm_keys({"vlm_interface.model.lm_head.weight": weight})

        self.assertIs(fixed["vlm_interface.model.language_model.embed_tokens.weight"], weight)
        self.assertIs(fixed["vlm_interface.model.embed_tokens.weight"], weight)

    def test_loads_one_checkpoint_tensor_into_tied_embeddings(self) -> None:
        class DummyVLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.embed_tokens = nn.Embedding(5, 3)
                self.embed_tokens = self.language_model.embed_tokens
                self.lm_head = nn.Linear(3, 5, bias=False)
                self.lm_head.weight = self.embed_tokens.weight

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vlm_interface = nn.Module()
                self.vlm_interface.model = DummyVLM()

        embedding = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        checkpoint_state = {
            "paligemma_with_expert.paligemma.lm_head.weight": embedding,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            torch.save(checkpoint_state, checkpoint)
            model = DummyModel()
            summary = load_pi0_weights(model, checkpoint, strict=True, verbose=False)

        self.assertEqual(summary["missing"], [])
        self.assertEqual(summary["shape_mismatch"], [])
        self.assertTrue(torch.equal(model.vlm_interface.model.embed_tokens.weight, embedding))
        self.assertIs(
            model.vlm_interface.model.lm_head.weight,
            model.vlm_interface.model.embed_tokens.weight,
        )

    def test_strict_loading_rejects_shape_mismatch(self) -> None:
        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.flow_matching_head = nn.Module()
                self.flow_matching_head.state_proj = nn.Linear(2, 3, bias=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            torch.save({"state_proj.weight": torch.zeros(4, 8)}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "Shape mismatches"):
                load_pi0_weights(DummyModel(), checkpoint, strict=True, verbose=False)


if __name__ == "__main__":
    unittest.main()
