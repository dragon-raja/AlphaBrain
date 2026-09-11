from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.dsol_paper1.audit_view_value_expectation_heldout_run import audit
from scripts.dsol_paper1.explicit_flow_noise import materialize_bank, sha256_file


def episode_id(pair: str, repeat: int, method: str) -> str:
    return hashlib.sha256(f"{pair}/{repeat}/{method}".encode()).hexdigest()[:24]


class ViewValueExpectationHeldoutRunAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        bank = materialize_bank(
            output_dir=self.root / "noise",
            bank_id="E",
            state_keys=["state-a", "state-b"],
            repeat_count=2,
            max_replans=2,
            action_horizon=2,
            action_dim=1,
            root_seed=123,
        )
        self.noise_manifest = Path(bank["manifest_path"])
        specs = []
        for pair in ("state-a", "state-b"):
            for method in ("canonical", "selector"):
                for repeat in range(2):
                    specs.append(
                        {
                            "episode_id": episode_id(pair, repeat, method),
                            "pair_key": pair,
                            "selector_method": method,
                            "condition": f"selector__{method}",
                            "policy_repeat_id": repeat,
                            "noise_bank_id": "E",
                            "checkpoint_seed": 41,
                            "environment_seed": 99 if pair == "state-a" else 100,
                        }
                    )
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(
            json.dumps(
                {
                    "schema": "dsol_view_value_expectation_heldout_protocol_v1",
                    "status": "PASS",
                    "bank_id": "E",
                    "episode_count": len(specs),
                    "policy_noise_repeats": 2,
                    "selector_methods": ["canonical", "selector"],
                    "specs": specs,
                }
            )
        )
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.run_manifest = self.run_dir / "run_manifest.json"
        self.run_manifest.write_text(
            json.dumps(
                {
                    "schema": "dsol_libero_hdf5_closed_loop_run_v1",
                    "protocol": str(self.protocol),
                    "protocol_sha256": sha256_file(self.protocol),
                    "noise_bank_manifest": str(self.noise_manifest),
                    "noise_bank_manifest_sha256": sha256_file(self.noise_manifest),
                    "require_explicit_noise": True,
                    "eval_worker_count": 2,
                }
            )
        )
        from scripts.dsol_paper1.explicit_flow_noise import ExplicitFlowNoiseBank

        noise = ExplicitFlowNoiseBank(self.noise_manifest)
        self.rows = []
        for index, spec in enumerate(specs):
            entry = noise.get(spec["pair_key"], spec["policy_repeat_id"], 0)
            row = {
                **spec,
                "status": "complete",
                "explicit_flow_noise": True,
                "noise_bank_manifest_sha256": sha256_file(self.noise_manifest),
                "initial_metrics": {
                    "physics_state_sha256": f"physics-{spec['pair_key']}",
                    "post_wait_physics_state_sha256_exact": f"physics-{spec['pair_key']}",
                },
                "inference_calls": 1,
                "policy_calls": [
                    {
                        "policy_repeat_id": spec["policy_repeat_id"],
                        "replan_index": 0,
                        "noise_seed": entry["noise_seed"],
                        "noise_sha256": entry["noise_sha256"],
                    }
                ],
            }
            self.rows.append((index % 2, row))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_rows(self, rows: list[tuple[int, dict]]) -> None:
        for shard in range(2):
            values = [row for row_shard, row in rows if row_shard == shard]
            (self.run_dir / f"episodes-shard-{shard:02d}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in values)
            )

    def run_audit(self, *, complete: bool = False) -> dict:
        return audit(
            protocol_path=self.protocol,
            run_dir=self.run_dir,
            run_manifest_path=self.run_manifest,
            noise_manifest_path=self.noise_manifest,
            require_complete=complete,
        )

    def test_partial_subset_reconstructs_noise(self) -> None:
        self.write_rows(self.rows[:4])
        result = self.run_audit()
        self.assertEqual(result["status"], "PASS_PARTIAL_SUBSET")
        self.assertEqual(result["result_episode_count"], 4)
        self.assertEqual(result["reconstructed_policy_calls"], 4)

    def test_complete_population_and_shards_pass(self) -> None:
        self.write_rows(self.rows)
        result = self.run_audit(complete=True)
        self.assertEqual(result["status"], "PASS_COMPLETE")
        self.assertEqual(result["complete_state_matrices"], 2)

    def test_protocol_field_drift_fails_closed(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0][1]["environment_seed"] += 1
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "protocol field environment_seed"):
            self.run_audit()

    def test_noise_drift_fails_closed(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0][1]["policy_calls"][0]["noise_sha256"] = "bad"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "noise SHA-256 mismatch"):
            self.run_audit()

    def test_complete_mode_rejects_partial_population(self) -> None:
        self.write_rows(self.rows[:4])
        with self.assertRaisesRegex(ValueError, "episode set is incomplete"):
            self.run_audit(complete=True)


if __name__ == "__main__":
    unittest.main()
