from __future__ import annotations

import numpy as np
import pytest

from build_libero_bind_decision_view import (
    decision_anchor_index,
    decision_anchor_key,
)


def test_decision_anchor_indices_match_causal_action_changes() -> None:
    phases = np.asarray(
        ["episode_start", "approach_grasp", "lift", "lift", "transport"]
    )
    assert decision_anchor_index(phases, "source_select") == 0
    assert decision_anchor_index(phases, "target_select") == 3


def test_source_decision_requires_episode_start_at_frame_zero() -> None:
    with pytest.raises(ValueError, match="frame zero"):
        decision_anchor_index(
            np.asarray(["approach_grasp", "episode_start", "transport"]),
            "source_select",
        )


def test_decision_anchor_key_separates_source_and_target_states() -> None:
    source = decision_anchor_key("red-left", 3, "source_select", "action")
    target = decision_anchor_key("red-left", 3, "target_select", "action")
    assert source == "red-left__state_03__source_select__action"
    assert target == "red-left__state_03__target_select__action"
    assert source != target
