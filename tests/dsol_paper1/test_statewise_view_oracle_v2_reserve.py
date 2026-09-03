from __future__ import annotations

import json

from scripts.dsol_paper1.clone_statewise_view_oracle_v2_reserve import clone


def test_reserve_clone_changes_only_bank_identity_and_keeps_candidates(tmp_path) -> None:
    source = {
        "schema": "dsol_compact_view_matrix_protocol_v1",
        "status": "PASS_FROZEN",
        "phase": "independent_confirmation",
        "noise_bank_id": "Q",
        "episode_identity_prefix": "oracle-v2::Q::test",
        "episode_count": 64,
        "state_blocks": [{"state": {"pair_key": "s"}, "candidates": [{"selected_candidate_id": "v"}]}],
    }
    path = tmp_path / "Q.json"
    path.write_text(json.dumps(source))
    reserve = clone(path)
    assert reserve["noise_bank_id"] == "R"
    assert reserve["state_blocks"] == source["state_blocks"]
    assert reserve["candidate_set_changed_from_Q"] is False
