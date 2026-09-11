from __future__ import annotations

from pathlib import Path

import pytest

from scripts.dsol_paper1.build_dense_test_scan_plan import build


def _protocol(split: str = "test") -> dict:
    return {
        "schema": "dsol_constructed_dense_view_oracle_protocol_v1",
        "status": "PASS",
        "split": split,
        "selected_state_count": 2,
        "selected_states": [{"pair_key": "state-1"}, {"pair_key": "state-3"}],
    }


def test_build_selects_only_frozen_test_states(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    protocol = tmp_path / "protocol.json"
    source.write_text("{}")
    protocol.write_text("{}")
    payload = build(
        {
            "status": "PASS",
            "records": [
                {"scan_id": "state-1", "split": "test", "task_id": "a"},
                {"scan_id": "state-2", "split": "test", "task_id": "a"},
                {"scan_id": "state-3", "split": "test", "task_id": "b"},
            ],
        },
        _protocol(),
        source_plan_path=source,
        protocol_path=protocol,
    )
    assert payload["record_count"] == 2
    assert [row["scan_id"] for row in payload["records"]] == ["state-1", "state-3"]
    assert payload["policy_outcomes_used_for_selection"] is False


def test_build_rejects_validation_protocol(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    protocol = tmp_path / "protocol.json"
    source.write_text("{}")
    protocol.write_text("{}")
    with pytest.raises(ValueError, match="test protocol"):
        build(
            {"status": "PASS", "records": []},
            _protocol(split="val"),
            source_plan_path=source,
            protocol_path=protocol,
        )
