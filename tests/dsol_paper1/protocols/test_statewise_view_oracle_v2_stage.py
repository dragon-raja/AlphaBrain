from __future__ import annotations

from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2_stage import (
    rank_key,
    select_noncanonical,
    summarize_rows,
)


def _row(candidate: str, repeat: int, success: bool, progress: float) -> dict[str, object]:
    return {
        "episode_id": f"{candidate}-{repeat}",
        "status": "complete",
        "explicit_flow_noise": True,
        "noise_bank_id": "O",
        "pair_key": "state",
        "selected_candidate_id": candidate,
        "policy_repeat_id": repeat,
        "success": success,
        "normalized_final_progress": progress,
        "completion_steps": 10 + repeat,
    }


def test_independent_stage_summary_and_selection() -> None:
    rows = []
    for repeat in range(2):
        rows.extend(
            [
                _row("canonical", repeat, repeat == 0, 0.5),
                _row("view-a", repeat, True, 1.0),
                _row("view-b", repeat, False, 0.75),
            ]
        )
    summaries = summarize_rows(rows, expected_bank="O", expected_repeats=2)
    assert summaries["state"]["view-a"]["mean_success"] == 1.0
    assert summaries["state"]["view-b"]["harm_probability"] == 0.5
    assert select_noncanonical(summaries, 1) == {"state": ["view-a"]}


def test_rank_uses_progress_before_harm_and_steps() -> None:
    better_progress = {
        "candidate_id": "z",
        "mean_success": 0.5,
        "mean_progress": 0.75,
        "harm_probability": 1.0,
        "mean_success_steps": 100,
    }
    lower_progress = {
        "candidate_id": "a",
        "mean_success": 0.5,
        "mean_progress": 0.5,
        "harm_probability": 0.0,
        "mean_success_steps": 1,
    }
    assert rank_key(better_progress) < rank_key(lower_progress)
