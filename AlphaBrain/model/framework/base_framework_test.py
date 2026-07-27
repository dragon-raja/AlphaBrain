from __future__ import annotations

from AlphaBrain.model.framework.base_framework import (
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
