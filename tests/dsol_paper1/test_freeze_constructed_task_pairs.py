from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "dsol_paper1"
sys.path.insert(0, str(SCRIPT_ROOT))

from freeze_constructed_task_pairs import freeze  # noqa: E402


def test_freeze_uses_episode_medians_and_selects_direction(tmp_path: Path) -> None:
    rows = []
    for episode, values in (("a", (0.006, 0.0)), ("b", (0.008, 0.001))):
        output = tmp_path / episode
        output.mkdir()
        (output / "scan.json").write_text(
            json.dumps(
                {
                    "initial_task_success": False,
                    "records": [
                        {
                            "group": "constructed_task_orbit",
                            "delta_visibility": values[0],
                            "pose": {"pair_id": "side", "pair_member": "negative"},
                        },
                        {
                            "group": "constructed_task_orbit",
                            "delta_visibility": values[1],
                            "pose": {"pair_id": "side", "pair_member": "positive"},
                        },
                    ],
                }
            )
        )
        rows.append(
            {
                "task_id": "task",
                "episode_id": episode,
                "split": "val",
                "status": "PASS",
                "stage_fraction": 0.2,
                "output_dir": str(output),
            }
        )
    result = freeze(rows, minimum_strong_delta=0.005, maximum_control_abs_delta=0.005)
    assert result["status"] == "PASS"
    assert result["tasks"]["task"]["selected"]["strong_member"] == "negative"
    assert result["tasks"]["task"]["selected"]["information_specificity"] == pytest.approx(0.0065)
