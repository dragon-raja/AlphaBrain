from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "dsol_paper1"
sys.path.insert(0, str(SCRIPT_ROOT))

from build_strong_information_gate_plan import build  # noqa: E402


def test_build_is_outcome_blind_and_early(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("{}")
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    source = {
        "_source_path": str(source_path),
        "records": [
            {"task_id": "task-a", "split": "val", "stage_fraction": 0.2},
            {"task_id": "task-a", "split": "test", "stage_fraction": 0.4},
            {"task_id": "task-b", "split": "test", "stage_fraction": 0.1},
        ],
    }

    result = build(source, {"task-a": spec}, maximum_stage=0.25)

    assert result["record_count"] == 1
    assert result["counts"] == {"val::task-a": 1}
    assert result["policy_outcomes_used_for_selection"] is False
    assert result["records"][0]["construction_spec"] == str(spec)
