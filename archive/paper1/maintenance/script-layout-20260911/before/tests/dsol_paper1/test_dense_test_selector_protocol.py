from __future__ import annotations

from pathlib import Path

from scripts.dsol_paper1.build_dense_test_selector_protocol import (
    SELECTOR_METHODS,
    build,
)


def _visibility(score_a: float, score_b: float) -> dict:
    return {
        "score": (score_a + score_b) / 2,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "entity_names": ["target"],
        "per_camera": {
            "agentview": {
                "entities": {"target": {"visible_fraction": score_a}}
            },
            "robot0_eye_in_hand": {
                "entities": {"target": {"visible_fraction": score_b}}
            },
        },
    }


def _spec(pair_key: str, candidate: str) -> dict:
    return {
        "pair_key": pair_key,
        "selected_candidate_id": candidate,
        "task_id": "task-a",
        "episode_id_source": "suite::task::demo_1",
        "selection_metadata": {"catalog_group": "canonical"},
    }


def test_builder_freezes_outcome_blind_selector_methods(tmp_path: Path) -> None:
    dense_path = tmp_path / "dense.json"
    dense_path.write_text("{}")
    pair_key = "task-a::test::demo_1::stage-01::frame-00010"
    protocol = {
        "schema": "dsol_constructed_dense_view_oracle_protocol_v1",
        "status": "PASS",
        "split": "test",
        "candidate_count": 2,
        "catalog": "/tmp/catalog.json",
        "specs": [_spec(pair_key, "canonical"), _spec(pair_key, "view-b")],
    }
    scans = {
        pair_key: [
            {"pose_id": "canonical", "visibility": _visibility(0.1, 0.1)},
            {"pose_id": "view-b", "visibility": _visibility(0.3, 0.1)},
        ]
    }
    payload = build(
        protocol,
        scans,
        dense_protocol_path=dense_path,
        global_fixed_candidate="view-b",
        visibility_gain_threshold=0.005,
    )
    assert payload["selection_uses_test_policy_outcomes"] is False
    assert payload["selector_methods"] == list(SELECTOR_METHODS)
    assert payload["episode_count"] == len(SELECTOR_METHODS)
    assert payload["catalog"] == "/tmp/catalog.json"
    selected = payload["selected_states"][0]["selections"]
    assert selected["canonical"] == "canonical"
    assert selected["visibility_mean"] == "view-b"
    assert selected["visibility_gain_gated"] == "view-b"
