from __future__ import annotations

import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from AlphaBrain.model.modules.vlm.vision_low_rank_adapter import (
    LowRankAdaptedLinear,
    inject_siglip_low_rank_adapters,
)


def _tiny_siglip() -> SiglipVisionModel:
    return SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=2,
            image_size=4,
        )
    )


def test_adapter_is_identity_at_initialization_and_preserves_base_keys() -> None:
    model = _tiny_siglip().eval()
    inputs = torch.randn(2, 3, 4, 4)
    with torch.no_grad():
        expected = model(pixel_values=inputs).last_hidden_state

    summary = inject_siglip_low_rank_adapters(
        model,
        rank=2,
        alpha=2.0,
        freeze_base=True,
    )
    with torch.no_grad():
        actual = model(pixel_values=inputs).last_hidden_state

    torch.testing.assert_close(actual, expected)
    assert summary.adapted_layer_count == 4
    assert summary.adapter_parameter_count == 192
    keys = set(model.state_dict())
    assert "vision_model.encoder.layers.0.mlp.fc1.weight" in keys
    assert "vision_model.encoder.layers.0.mlp.fc1.adapter_A" in keys
    assert "vision_model.encoder.layers.0.mlp.fc1.adapter_B" in keys


def test_only_adapter_parameters_are_trainable_in_frozen_vision_tower() -> None:
    model = _tiny_siglip()
    inject_siglip_low_rank_adapters(
        model,
        rank=2,
        alpha=2.0,
        freeze_base=True,
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.endswith(("adapter_A", "adapter_B")) for name in trainable
    )


def test_low_rank_path_receives_gradient_and_round_trips_strictly() -> None:
    model = _tiny_siglip()
    inject_siglip_low_rank_adapters(
        model,
        rank=2,
        alpha=2.0,
        freeze_base=True,
    )
    first = next(
        module
        for module in model.modules()
        if isinstance(module, LowRankAdaptedLinear)
    )
    with torch.no_grad():
        first.adapter_B.normal_(std=0.01)
    loss = model(pixel_values=torch.randn(2, 3, 4, 4)).last_hidden_state.square().mean()
    loss.backward()
    assert torch.count_nonzero(first.adapter_A.grad).item() > 0
    assert torch.count_nonzero(first.adapter_B.grad).item() > 0

    restored = _tiny_siglip()
    inject_siglip_low_rank_adapters(
        restored,
        rank=2,
        alpha=2.0,
        freeze_base=True,
    )
    restored.load_state_dict(model.state_dict(), strict=True)
