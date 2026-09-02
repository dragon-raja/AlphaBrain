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
