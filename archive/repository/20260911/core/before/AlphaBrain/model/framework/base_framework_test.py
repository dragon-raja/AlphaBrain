from __future__ import annotations

import torch
from safetensors.torch import save_file

import AlphaBrain.model.framework.base_framework as base_framework
from AlphaBrain.model.framework.base_framework import (
    BaseFramework,
    _critical_state_dict_mismatches,
)


def test_critical_state_dict_mismatches_reject_camera_and_action_weights() -> None:
    missing, unexpected = _critical_state_dict_mismatches(
        {
            "camera_conditioner.fusion.weight",
            "flow_matching_head.action_out_proj.bias",
            "noncritical.buffer",
        },
        {
            "vlm_interface.model.multi_modal_projector.weight",
            "legacy.buffer",
        },
    )

    assert missing == [
        "camera_conditioner.fusion.weight",
        "flow_matching_head.action_out_proj.bias",
    ]
    assert unexpected == [
        "vlm_interface.model.multi_modal_projector.weight",
    ]


def test_critical_state_dict_mismatches_allows_known_noncritical_compatibility() -> None:
    assert _critical_state_dict_mismatches(
        {"legacy.position_ids"},
        {"old_alias.weight"},
    ) == ([], [])


def test_native_strict_checkpoint_rejects_any_missing_weight(
    monkeypatch,
    tmp_path,
) -> None:
    class TinyFramework(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.required = torch.nn.Linear(2, 2)
            self.optional = torch.nn.Linear(2, 2)

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = TinyFramework()
    incomplete = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("optional.")
    }
    save_file(incomplete, checkpoint / "model.safetensors")
    monkeypatch.setattr(
        base_framework,
        "read_mode_config",
        lambda _path: (
            {
                "trainer": {"pretrained_checkpoint": None},
                "framework": {},
            },
            {},
        ),
    )
    monkeypatch.setattr(
        base_framework,
        "build_framework",
        lambda cfg: TinyFramework(),
    )

    loaded = BaseFramework.from_pretrained(str(checkpoint))
    assert isinstance(loaded, TinyFramework)

    try:
        BaseFramework.from_pretrained(
            str(checkpoint),
            strict_checkpoint=True,
        )
    except RuntimeError as error:
        assert "exact state-dict keys and tensor shapes" in str(error)
    else:
        raise AssertionError("strict native load accepted an incomplete checkpoint")
