from __future__ import annotations

import copy

import pytest

from merge_kyc_camera_eval_shards import merge_payloads


def _row(edge: str, pose: str, state: int = 40, horizon: int = 3) -> dict:
    return {
        "edge_id": edge,
        "camera_pose": pose,
        "canonical_state_index": state,
        "execution_horizon": horizon,
        "success": False,
    }


def _payload(edges: list[str], *, status: str) -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "study": "test",
        "suite": "/suite",
        "camera_config_path": "/camera.json",
        "camera_config": {
            "name": "test",
            "poses": [{"name": "baseline"}, {"name": "shifted"}],
        },
        "split": "test",
        "state_indices": [40],
        "edges": edges,
        "poses": ["baseline", "shifted"],
        "execution_horizons": [3],
        "expected_episode_count": len(edges) * 2,
        "policy_identity": {"checkpoint_realpath": "/checkpoint", "horizon": 20},
        "rows": [_row(edge, pose) for edge in edges for pose in ("baseline", "shifted")],
    }


def test_merge_requires_exact_disjoint_episode_grid() -> None:
    first = _payload(["red-left"], status="partial")
    second = _payload(["white-left"], status="complete")

    merged = merge_payloads(
        [first, second],
        expected_edges=["red-left", "white-left"],
        expected_state_indices=[40],
        expected_execution_horizons=[3],
    )

    assert merged["status"] == "complete"
    assert merged["expected_episode_count"] == 4
    assert merged["edges"] == ["red-left", "white-left"]
    assert [
        (row["edge_id"], row["camera_pose"]) for row in merged["rows"]
    ] == [
        ("red-left", "baseline"),
        ("red-left", "shifted"),
        ("white-left", "baseline"),
        ("white-left", "shifted"),
    ]


def test_merge_rejects_duplicate_episode_keys() -> None:
    first = _payload(["red-left"], status="partial")
    duplicate = _payload(["red-left"], status="complete")

    with pytest.raises(ValueError, match="duplicate episode key"):
        merge_payloads(
            [first, duplicate],
            expected_edges=["red-left"],
            expected_state_indices=[40],
            expected_execution_horizons=[3],
        )


def test_merge_rejects_missing_episode_keys() -> None:
    fragment = _payload(["red-left"], status="complete")
    fragment["rows"].pop()

    with pytest.raises(ValueError, match="missing=1"):
        merge_payloads(
            [fragment],
            expected_edges=["red-left"],
            expected_state_indices=[40],
            expected_execution_horizons=[3],
        )


def test_merge_rejects_mixed_checkpoint_identity() -> None:
    first = _payload(["red-left"], status="partial")
    second = _payload(["white-left"], status="complete")
    second["policy_identity"] = copy.deepcopy(second["policy_identity"])
    second["policy_identity"]["checkpoint_realpath"] = "/other"

    with pytest.raises(ValueError, match="policy_identity"):
        merge_payloads(
            [first, second],
            expected_edges=["red-left", "white-left"],
            expected_state_indices=[40],
            expected_execution_horizons=[3],
        )
