from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts/dsol_paper1/build_prefreeze_bundle.py"
)
SPEC = importlib.util.spec_from_file_location("build_prefreeze_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_formal_output_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="formal root"):
        MODULE.assert_debug_output(tmp_path / "formal" / "paper1")


def test_task_enumeration_is_group_preserving(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    tasks = {"CloseBlenderLid", "CloseFridge"}
    for partition in ("seen_scene_pretrain", "heldout_scene_target"):
        for task in tasks:
            for seed in (0, 1):
                path = bank / partition / "snapshots" / task / f"seed_{seed:06d}"
                path.mkdir(parents=True)
                payload = {
                    "metadata": {
                        "task": task,
                        "scene_seed": seed,
                        "prompt": f"prompt for {task}",
                    }
                }
                (path / "manifest.json").write_text(json.dumps(payload))

    templates, summary = MODULE.enumerate_task_templates(bank, tasks)
    assert summary["manifest_count"] == 8
    assert summary["unique_snapshot_group_count"] == 8
    assert summary["partition_counts"] == {
        "heldout_scene_target": 4,
        "seen_scene_pretrain": 4,
    }
    assert {item["task"] for item in templates} == tasks
    assert all(item["expert_decision_set_status"] == "NOT_MATERIALIZED" for item in templates)


def test_duplicate_group_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "bank/seen/snapshots/CloseFridge/seed_000000/manifest.json"
    second = tmp_path / "bank/seen/snapshots/CloseFridge/seed_000001/manifest.json"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {"metadata": {"task": "CloseFridge", "scene_seed": 0, "prompt": "close"}}
            )
        )
    with pytest.raises(RuntimeError, match="duplicate snapshot group"):
        MODULE.enumerate_task_templates(tmp_path / "bank", {"CloseFridge"})
