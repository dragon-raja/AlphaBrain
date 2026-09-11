from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.dsol_paper1.diagnostics.audit_statewise_view_oracle_v2_run import audit
from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import protocol_spec_at
from AlphaBrain.research.dsol.data.flow_noise import materialize_bank, tensor_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_outcome_blind_audit_accepts_complete_paired_matrix(tmp_path: Path) -> None:
    protocol = {
        "schema": "dsol_compact_view_matrix_protocol_v1",
        "catalog": "/catalog.json",
        "episode_identity_prefix": "test::S",
        "diagnostic_role": "smoke",
        "noise_bank_id": "S",
        "policy_repeat_ids": [0],
        "state_blocks": [
            {
                "state": {
                    "pair_key": "state",
                    "source_group": "source",
                    "task_id": "task",
                    "environment_seed": 7,
                    "construction_spec_sha256": "construction",
                },
                "scene_construction": {},
                "candidates": [
                    {"selected_candidate_id": "canonical", "pose": None},
                    {"selected_candidate_id": "view", "pose": {}},
                ],
            }
        ],
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    bank = materialize_bank(
        output_dir=tmp_path,
        bank_id="S",
        state_keys=["state"],
        repeat_count=1,
        max_replans=2,
        action_horizon=10,
        action_dim=7,
        root_seed=9,
    )
    bank_array = np.load(bank["noise_file"])
    manifest_path = tmp_path / "bank_S.manifest.json"
    rows = []
    for index in range(2):
        spec = protocol_spec_at(protocol, index)
        rows.append(
            {
                **spec,
                "status": "complete",
                "success": bool(index),
                "explicit_flow_noise": True,
                "inference_calls": 1,
                "initial_metrics": {
                    "physics_state_sha256": "physics",
                    "post_wait_physics_state_sha256_exact": "physics",
                },
                "policy_calls": [
                    {
                        "policy_repeat_id": 0,
                        "replan_index": 0,
                        "noise_seed": int(
                            __import__(
                                "AlphaBrain.research.dsol.data.flow_noise",
                                fromlist=["stable_uint64"],
                            ).stable_uint64("S::state::0::0", root_seed=9)
                        ),
                        "noise_sha256": tensor_sha256(bank_array[0, 0, 0]),
                        "action_chunk_sha256": "a" * 64,
                    }
                ],
            }
        )
    ledger = tmp_path / "episodes.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "protocol_sha256": _sha(protocol_path),
                "noise_bank_manifest_sha256": _sha(manifest_path),
                "require_explicit_noise": True,
            }
        )
    )
    result = audit(
        protocol_path=protocol_path,
        noise_manifest=manifest_path,
        run_manifest_path=run_manifest,
        ledger_patterns=[str(ledger)],
    )
    assert result["status"] == "PASS_COMPLETE"
    assert result["episode_count"] == 2
    assert result["outcome_values_aggregated"] is False
