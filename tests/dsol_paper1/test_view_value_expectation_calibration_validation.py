from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.dsol_paper1.protocols.build_view_value_expectation_calibration_stage import (
    validate_explicit_pairing,
)


def _spec(episode_id: str, candidate: str) -> dict:
    return {
        "episode_id": episode_id,
        "pair_key": "state-1",
        "selected_candidate_id": candidate,
        "policy_repeat_id": 0,
        "noise_bank_id": "B",
        "environment_seed": 41,
        "condition": f"candidate__{candidate}",
        "construction_spec_sha256": f"{1 if candidate == 'canonical' else 2:064x}",
        "source_group": "task::demo-1",
        "task_id": "task",
    }


def _row(spec: dict) -> dict:
    return {
        **spec,
        "status": "complete",
        "explicit_flow_noise": True,
        "noise_bank_manifest_sha256": "3" * 64,
        "inference_calls": 1,
        "initial_metrics": {
            "physics_state_sha256": "4" * 64,
            "post_wait_physics_state_sha256_exact": "4" * 64,
        },
        "policy_calls": [
            {
                "policy_repeat_id": 0,
                "replan_index": 0,
                "noise_seed": 123,
                "noise_sha256": "5" * 64,
                "action_chunk_sha256": "6" * 64,
            }
        ],
    }


def _fixture() -> tuple[list[dict], dict]:
    specs = [_spec("episode-canonical", "canonical"), _spec("episode-view", "view-1")]
    return [_row(spec) for spec in specs], {"status": "PASS", "stage": "B", "specs": specs}


def test_complete_protocol_exact_pairing_passes() -> None:
    rows, protocol = _fixture()
    validate_explicit_pairing(rows, "B", expected_protocol=protocol)


def test_missing_protocol_episode_fails_closed() -> None:
    rows, protocol = _fixture()
    with pytest.raises(ValueError, match="episode set is incomplete"):
        validate_explicit_pairing(rows[:1], "B", expected_protocol=protocol)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows[1].__setitem__("environment_seed", 42), "protocol field: environment_seed"),
        (
            lambda rows: rows[1]["initial_metrics"].__setitem__(
                "post_wait_physics_state_sha256_exact", "7" * 64
            ),
            "physics state changed",
        ),
        (
            lambda rows: rows[1]["policy_calls"][0].__setitem__("noise_sha256", "8" * 64),
            "explicit policy noise differs",
        ),
    ],
)
def test_pairing_or_restore_divergence_fails_closed(mutation, message: str) -> None:
    rows, protocol = _fixture()
    rows = deepcopy(rows)
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        validate_explicit_pairing(rows, "B", expected_protocol=protocol)
