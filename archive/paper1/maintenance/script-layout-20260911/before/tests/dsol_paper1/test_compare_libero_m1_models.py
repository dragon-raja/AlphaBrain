from __future__ import annotations

import pytest

from scripts.dsol_paper1.compare_dsol_libero_m1_models import (
    grouped_model_difference,
    validate_same_protocol,
)
from scripts.dsol_paper1.summarize_dsol_libero_m1_visibility import (
    EXPECTED_CONDITIONS,
    PHYSICS_STATE_STAGE,
)


def _pair(pair_key: str, source: str, success: int) -> dict:
    return {
        condition: {
            "pair_key": pair_key,
            "condition": condition,
            "task_id": "task",
            "episode_id_source": source,
            "success": success,
            "initial_metrics": {
                "physics_state_sha256": pair_key,
                "physics_state_stage": PHYSICS_STATE_STAGE,
            },
        }
        for condition in EXPECTED_CONDITIONS
    }


def test_grouped_model_difference_clusters_frame_states_by_source_demo() -> None:
    left = {
        "a": _pair("a", "demo-1", 1),
        "b": _pair("b", "demo-1", 0),
        "c": _pair("c", "demo-2", 1),
    }
    right = {
        "a": _pair("a", "demo-1", 0),
        "b": _pair("b", "demo-1", 0),
        "c": _pair("c", "demo-2", 0),
    }
    values, rows = grouped_model_difference(
        left, right, condition="strong_info_both"
    )
    assert values.tolist() == [0.5, 1.0]
    assert [row["paired_state_count"] for row in rows] == [2, 1]


def test_validate_same_protocol_rejects_missing_state() -> None:
    runs = {
        "baseline": {"a": _pair("a", "demo-1", 0)},
        "candidate": {"b": _pair("b", "demo-1", 0)},
    }
    with pytest.raises(ValueError, match="protocol mismatch"):
        validate_same_protocol(runs)
