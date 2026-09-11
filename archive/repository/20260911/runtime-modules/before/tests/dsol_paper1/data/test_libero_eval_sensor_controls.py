from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import (
    deployed_camera_names,
    masked_policy_observation,
    protocol_spec_at,
    protocol_spec_count,
    selected_spec_at,
    selected_spec_count,
    selected_specs,
)


def test_all_blackout_has_no_deployed_camera() -> None:
    assert deployed_camera_names("all_blackout") == ()


def test_all_blackout_zeroes_both_policy_images(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("evaluate_pi05_libero_plus_views")

    def prepare(*_args: object, **_kwargs: object):
        agent = np.full((2, 3, 3), 17, dtype=np.uint8)
        wrist = np.full((2, 3, 3), 29, dtype=np.uint8)
        return {
            "observation/image": agent.copy(),
            "observation/wrist_image": wrist.copy(),
        }, agent, wrist

    module.prepare_policy_observation = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    example, agent, wrist = masked_policy_observation(
        {},
        prompt="test",
        resize_size=2,
        eval_seed=1,
        camera_calibration={},
        sensor_control="all_blackout",
    )
    assert not agent.any()
    assert not wrist.any()
    assert not example["observation/image"].any()
    assert not example["observation/wrist_image"].any()


def _selected_specs_args(protocol: Path) -> argparse.Namespace:
    return argparse.Namespace(
        protocol=protocol,
        num_shards=1,
        shard_index=0,
        max_episodes=None,
    )


def test_selected_specs_accepts_per_spec_catalog(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"specs": [{"episode_id": "episode-1", "catalog": "/frozen/catalog.json"}]})
    )

    assert selected_specs(_selected_specs_args(protocol))[0]["catalog"] == "/frozen/catalog.json"


def test_selected_specs_prefers_top_level_catalog(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "catalog": "/top-level/catalog.json",
                "specs": [{"episode_id": "episode-1", "catalog": "/per-spec/catalog.json"}],
            }
        )
    )

    assert selected_specs(_selected_specs_args(protocol))[0]["catalog"] == "/top-level/catalog.json"


def test_selected_specs_requires_a_frozen_catalog(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"specs": [{"episode_id": "episode-1"}]}))

    with pytest.raises(ValueError, match="freeze a top-level or per-spec catalog"):
        selected_specs(_selected_specs_args(protocol))


def _compact_protocol() -> dict[str, object]:
    return {
        "schema": "dsol_compact_view_matrix_protocol_v1",
        "catalog": "/frozen/catalog.json",
        "episode_identity_prefix": "oracle-v2::O::wave-00",
        "diagnostic_role": "dense_statewise_oracle_discovery",
        "noise_bank_id": "O",
        "policy_repeat_ids": [0, 1],
        "state_blocks": [
            {
                "state": {"pair_key": "state-a", "task_id": "task-a"},
                "scene_construction": {"name": "scene-a"},
                "candidates": [
                    {"selected_candidate_id": "canonical", "pose": None},
                    {
                        "selected_candidate_id": "view-1",
                        "pose": {"position": [1, 2, 3]},
                    },
                ],
            },
            {
                "state": {"pair_key": "state-b", "task_id": "task-b"},
                "scene_construction": {"name": "scene-b"},
                "candidates": [
                    {"selected_candidate_id": "canonical", "pose": None},
                ],
            },
        ],
    }


def test_compact_protocol_expands_deterministically() -> None:
    protocol = _compact_protocol()
    assert protocol_spec_count(protocol) == 6
    first = protocol_spec_at(protocol, 0)
    third = protocol_spec_at(protocol, 2)
    last = protocol_spec_at(protocol, 5)
    assert (first["pair_key"], first["selected_candidate_id"], first["policy_repeat_id"]) == (
        "state-a",
        "canonical",
        0,
    )
    assert (third["pair_key"], third["selected_candidate_id"], third["policy_repeat_id"]) == (
        "state-a",
        "view-1",
        0,
    )
    assert (last["pair_key"], last["selected_candidate_id"], last["policy_repeat_id"]) == (
        "state-b",
        "canonical",
        1,
    )
    assert len({protocol_spec_at(protocol, index)["episode_id"] for index in range(6)}) == 6


def test_compact_protocol_direct_shard_access() -> None:
    protocol = _compact_protocol()
    args = argparse.Namespace(num_shards=4, shard_index=1, max_episodes=None)
    assert selected_spec_count(args, protocol) == 2
    assert selected_spec_at(args, protocol, 0) == protocol_spec_at(protocol, 1)
    assert selected_spec_at(args, protocol, 1) == protocol_spec_at(protocol, 5)
