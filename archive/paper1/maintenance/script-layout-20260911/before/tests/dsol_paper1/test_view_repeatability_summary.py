from __future__ import annotations

import pytest

from scripts.dsol_paper1.summarize_view_repeatability import (
    source_group_from_pair_key,
    summarize,
)


def _candidate_rows(
    pair_key: str,
    candidate_id: str,
    outcomes: list[int],
    *,
    visibility: bool = False,
) -> list[dict]:
    return [
        {
            "episode_id": f"{pair_key}::{candidate_id}::{index}",
            "pair_key": pair_key,
            "selected_candidate_id": candidate_id,
            "task_id": "task_a",
            "diagnostic_role": "repeat::canonical_success_mixed",
            "selection_metadata": {
                "repeat_selection_roles": ["visibility_top1"] if visibility else ["canonical"],
                "discovery_success": bool(outcomes[0]),
            },
            "success": bool(outcome),
            "completion_steps": 10 + index,
            "policy_noise_seed": 100 + index,
        }
        for index, outcome in enumerate(outcomes)
    ]


def test_source_group_removes_only_stage_suffix() -> None:
    assert (
        source_group_from_pair_key("task::val::demo_1::stage-05::frame-00042")
        == "task::val::demo_1"
    )
    with pytest.raises(ValueError, match="stage suffix"):
        source_group_from_pair_key("task::val::demo_1")


def test_summary_reports_grouped_rescue_harm_and_bootstrap() -> None:
    rows = []
    definitions = (
        ("task_a::val::demo_1::stage-01::frame-00010", [0, 0, 0], [1, 1, 0]),
        ("task_a::val::demo_1::stage-05::frame-00050", [1, 1, 1], [0, 0, 1]),
        ("task_a::val::demo_2::stage-01::frame-00012", [1, 1, 1], [1, 1, 1]),
        ("task_a::val::demo_2::stage-05::frame-00052", [0, 0, 0], [0, 0, 0]),
    )
    for pair_key, canonical, visibility in definitions:
        rows.extend(_candidate_rows(pair_key, "canonical", canonical))
        rows.extend(
            _candidate_rows(pair_key, "visibility", visibility, visibility=True)
        )

    payload, candidate_rows = summarize(rows, expected_seeds=3)

    assert len(candidate_rows) == 8
    assert payload["states"] == 4
    assert payload["stable_rescue_state_count"] == 1
    assert payload["visibility_rescue_state_count"] == 1
    assert payload["visibility_harm_state_count"] == 1
    assert payload["canonical_mean_repeat_success_rate"] == pytest.approx(0.5)
    assert payload["visibility_mean_repeat_success_rate"] == pytest.approx(0.5)
    assert payload["task_summary"]["task_a"]["source_groups"] == 2
    bootstrap = payload["group_bootstrap_95ci"]
    assert bootstrap["independent_unit"] == "source_episode"
    assert bootstrap["source_groups"] == 2
    assert bootstrap["metrics"]["visibility_minus_canonical_pp"]["estimate"] == pytest.approx(0.0)
