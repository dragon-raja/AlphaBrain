from __future__ import annotations

import json
from pathlib import Path

from scripts.dsol_paper1.build_eval_throughput_benchmark import (
    build_benchmark_protocol,
)
from scripts.dsol_paper1.compare_eval_throughput_benchmarks import rollout_signature


def test_benchmark_protocol_preserves_episode_identity_fields(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("{}\n", encoding="utf-8")
    source = {
        "schema": "dsol_compact_view_matrix_protocol_v1",
        "episode_identity_prefix": "frozen-prefix",
        "policy_repeat_ids": [0, 1],
        "state_blocks": [
            {"state": {"pair_key": "a"}, "candidates": [{"selected_candidate_id": "x"}]},
            {"state": {"pair_key": "b"}, "candidates": [{"selected_candidate_id": "x"}]},
        ],
    }
    result = build_benchmark_protocol(source, source_path=source_path, state_count=1)
    assert result["episode_identity_prefix"] == "frozen-prefix"
    assert result["state_blocks"] == source["state_blocks"][:1]
    assert result["episode_count"] == 2
    assert result["excluded_from_scientific_estimands"] is True


def test_rollout_signature_ignores_nonscientific_serialization_fields() -> None:
    row = {
        "status": "complete",
        "success": True,
        "completion_steps": 10,
        "inference_calls": 1,
        "normalized_final_progress": 1.0,
        "initial_metrics": {
            "physics_state_sha256": "before",
            "post_wait_physics_state_sha256_exact": "after",
        },
        "policy_calls": [
            {
                "replan_index": 0,
                "noise_seed": 1,
                "noise_sha256": "noise",
                "action_chunk_sha256": "action",
            }
        ],
        "irrelevant": json.dumps({"worker": 3}),
    }
    changed = {**row, "irrelevant": "different"}
    assert rollout_signature(row) == rollout_signature(changed)
