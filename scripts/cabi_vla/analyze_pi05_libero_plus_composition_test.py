from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_pi05_libero_plus_composition import (
    CONDITIONS,
    build_report,
    read_composition_group_scores,
)


def _scores(*, combined: float) -> dict[str, dict[str, float]]:
    return {
        f"suite::task-{index}": {
            "canonical": 1.0,
            "camera_only": 1.0,
            "background_only": 1.0,
            "camera_background": combined,
        }
        for index in range(4)
    }


def test_report_confirms_residual_composition_gap() -> None:
    report = build_report(
        {"official": _scores(combined=0.0), "strong": _scores(combined=0.0)},
        reference_name="official",
        candidate_name="strong",
    )
    assert report["decision"] == "RESIDUAL_CAMERA_SCENE_COMPOSITION_GAP_CONFIRMED"
    assert report["runs"]["strong"]["effects"][
        "negative_composition_interaction"
    ]["mean"] == 1.0


def test_report_accepts_sufficient_multiview_result() -> None:
    report = build_report(
        {"official": _scores(combined=0.0), "strong": _scores(combined=1.0)},
        reference_name="official",
        candidate_name="strong",
    )
    assert report["decision"] == "MULTIVIEW_DATA_SUFFICIENT_ON_TESTED_PLUS_COMPOSITION"
    assert report["candidate_minus_reference"]["camera_background_success"]["mean"] == 1.0


def test_reader_requires_four_conditions_and_matching_physics(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    path = output / "episodes-shard-00.jsonl"
    rows = []
    for index, condition in enumerate(CONDITIONS):
        rows.append(
            {
                "episode_id": f"episode-{index}",
                "pair_key": "composition::suite::task::init0",
                "condition": condition,
                "suite": "suite",
                "base_task": "task",
                "success": True,
                "initial_metrics": {"physics_state_sha256": "same"},
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    scores = read_composition_group_scores(output)
    assert scores["suite::task"]["camera_background"] == 1.0

    rows[-1]["initial_metrics"]["physics_state_sha256"] = "different"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="physical initial state mismatch"):
        read_composition_group_scores(output)
