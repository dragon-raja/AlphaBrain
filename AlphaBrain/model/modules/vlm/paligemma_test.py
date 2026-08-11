import unittest

from transformers import GemmaConfig, SiglipVisionConfig

from AlphaBrain.model.modules.vlm.paligemma import PaliGemmaVLM


class PaliGemmaVLMTest(unittest.TestCase):
    def test_lm_head_and_input_embeddings_are_tied(self) -> None:
        gemma_config = GemmaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        vision_config = SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=14,
            patch_size=14,
        )
        model = PaliGemmaVLM(gemma_config, vision_config)

        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        self.assertIs(model.lm_head.weight, model.language_model.embed_tokens.weight)


if __name__ == "__main__":
    unittest.main()
