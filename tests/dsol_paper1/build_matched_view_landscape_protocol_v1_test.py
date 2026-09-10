from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import audit_statewise_view_oracle_v2_run as old_auditor
from explicit_flow_noise import sha256_file

from build_matched_view_landscape_protocol_v1 import (
    DEFAULT_SALT, DIAGNOSTIC_ROLE, EXPECTED_CANDIDATES, PREFIX, SMOKE_CANDIDATES,
    build, expanded_specs, make_protocol, matching_blocks, select_development_states,
    selection_digest, verify_reuse_partition, write_new_json,
)


def population_fixture():
    states = [
        {"task_id": f"task-{task}", "source_group": f"task-{task}::demo-{demo}",
         "pair_key": f"oracle-v2::development::task-{task}::demo-{demo}",
         "asset_source_pair_key": f"legacy::{task}::{demo}",
         "split": "development", "v2_role": "development",
         "environment_seed": task * 10 + demo, "construction_spec_sha256": "a" * 64,
         "source_state_index": 17, "static_assets": {"physics_state_sha256": "b" * 64}}
        for task in range(8) for demo in range(6)
    ]
    return {"status": "PASS", "population": {"development": {"states": states},
                                                "test": {"states": ["MUST_NOT_BE_READ"]}}}


def protocol_fixture(population):
    blocks = [{
        "state": deepcopy(state), "scene_construction": {"sha256": "c" * 64},
        "candidates": [{"selected_candidate_id": candidate, "pose": {"yaw": index},
                        "candidate_features": {"visibility_score": index / 97}}
                       for index, candidate in enumerate(EXPECTED_CANDIDATES)],
    } for state in population["population"]["development"]["states"]]
    return [{
        "schema": "dsol_compact_view_matrix_protocol_v1", "status": "PASS_FROZEN",
        "role": "development", "noise_bank_id": "O",
        "policy_repeat_ids": list(range(index * 4, index * 4 + 4)),
        "state_blocks": deepcopy(blocks), "catalog": "/fake/catalog.json",
        "catalog_sha256": "d" * 64, "population": "/fake/population.json",
        "population_sha256": "e" * 64, "sensor_control": "both",
    } for index in range(8)]


class MatchedViewLandscapeProtocolTest(unittest.TestCase):
    def setUp(self):
        self.population = population_fixture()
        self.states, self.rankings = select_development_states(self.population)
        self.templates = protocol_fixture(self.population)
        self.blocks = matching_blocks(self.templates, self.states)

    def protocol(self, repeats, candidates=EXPECTED_CANDIDATES, label="test"):
        return make_protocol(
            self.templates[0], self.blocks, repeats=repeats, label=label,
            selection_path=Path("/fake/selection.json"), selection_sha256="f" * 64,
            candidates=candidates,
        )

    def test_selection_is_eight_development_sources_ranked_without_outcomes(self):
        self.assertEqual(len(self.states), 8)
        self.assertEqual(len({state["task_id"] for state in self.states}), 8)
        self.assertTrue(all(state["split"] == "development" for state in self.states))
        for state, ranking in zip(self.states, self.rankings):
            self.assertEqual(state["pair_key"], ranking["ranked_sources"][0]["pair_key"])
            self.assertEqual(selection_digest(DEFAULT_SALT, state), ranking["ranked_sources"][0]["selection_sha256"])

    def test_selection_does_not_depend_on_row_order_or_extra_outcome_fields(self):
        altered = deepcopy(self.population)
        random.Random(42).shuffle(altered["population"]["development"]["states"])
        for index, state in enumerate(altered["population"]["development"]["states"]):
            state["success"] = index % 2 == 0
            state["accel"] = 1000 - index
        changed, _ = select_development_states(altered)
        self.assertEqual([state["pair_key"] for state in changed], [state["pair_key"] for state in self.states])

    def test_reject_test_role_or_duplicate_source(self):
        altered = deepcopy(self.population)
        altered["population"]["development"]["states"][0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "non-development"):
            select_development_states(altered)
        altered = deepcopy(self.population)
        rows = altered["population"]["development"]["states"]
        rows[1]["source_group"] = rows[0]["source_group"]
        with self.assertRaisesRegex(ValueError, "source groups"):
            select_development_states(altered)

    def test_original_state_aliases_and_candidates_are_preserved(self):
        full = self.protocol(range(4))
        self.assertEqual(full["state_blocks"], self.blocks)
        for state, block in zip(self.states, full["state_blocks"]):
            self.assertEqual(state, block["state"])
        self.assertEqual(full["diagnostic_role"], DIAGNOSTIC_ROLE)
        self.assertEqual(full["episode_identity_prefix"], PREFIX)
        self.assertNotIn("oracle-v2", full["episode_identity_prefix"])
        self.assertFalse(full["test_states_included"])

    def test_reject_wave_changed_physics_or_repeat(self):
        changed = deepcopy(self.templates)
        pair_key = self.states[0]["pair_key"]
        for block in changed[7]["state_blocks"]:
            if block["state"]["pair_key"] == pair_key:
                block["scene_construction"]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "differ across"):
            matching_blocks(changed, self.states)
        changed = deepcopy(self.templates)
        changed[4]["policy_repeat_ids"] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "partition"):
            matching_blocks(changed, self.states)

    def test_smoke_is_exact_dense_subset_and_residuals_are_a_disjoint_cover(self):
        full = self.protocol(range(4), label="wave-00")
        smoke = self.protocol([0, 1], SMOKE_CANDIDATES, label="smoke")
        remaining = [candidate for candidate in EXPECTED_CANDIDATES if candidate not in SMOKE_CANDIDATES]
        residual_a = self.protocol([0, 1], remaining, label="remainder-a")
        residual_b = self.protocol([2, 3], label="remainder-b")
        receipt = verify_reuse_partition(full, [smoke, residual_a, residual_b])
        self.assertEqual(receipt["full_episode_count"], 3104)
        self.assertEqual(receipt["part_episode_counts"], [48, 1504, 1552])
        dense_specs = expanded_specs(full)
        for episode_id, spec in expanded_specs(smoke).items():
            self.assertEqual(spec, dense_specs[episode_id])

    def test_reuse_rejects_overlap_and_changed_execution_fields(self):
        full = self.protocol(range(4))
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_reuse_partition(full, [full, full])
        altered = deepcopy(full)
        altered["sensor_control"] = "external_only"
        with self.assertRaisesRegex(ValueError, "different expanded"):
            verify_reuse_partition(full, [altered])

    def test_dense_eight_waves_have_24832_unique_ids(self):
        seen = set()
        for index in range(8):
            wave = self.protocol(range(index * 4, index * 4 + 4))
            ids = set(expanded_specs(wave))
            self.assertEqual(len(ids), 3104)
            self.assertFalse(seen & ids)
            seen.update(ids)
        self.assertEqual(len(seen), 24832)

    def test_no_overwrite_even_for_identical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_new_json(path, {"frozen": True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"frozen": True})
            self.assertEqual(path.read_bytes(), original)
            (Path(directory) / "protocols").mkdir()
            with self.assertRaisesRegex(FileExistsError, "protocol directory"):
                build(Path("/missing/v2"), Path(directory))

    def test_new_smoke_schema_passes_existing_integrity_auditor(self):
        smoke = self.protocol([0, 1], SMOKE_CANDIDATES, label="smoke")

        class FakeVerifiedNoiseBank:
            def __init__(self, *_args, **_kwargs):
                self.manifest = {"bank_id": "O", "noise_file_sha256": "7" * 64}

            def get(self, pair_key, repeat_id, replan_index):
                return {"noise_seed": repeat_id * 1000 + replan_index,
                        "noise_sha256": "8" * 64}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, noise_path = root / "smoke.json", root / "noise.json"
            run_path, ledger_path = root / "run_manifest.json", root / "episodes.jsonl"
            write_new_json(protocol_path, smoke)
            write_new_json(noise_path, {"fixture": "noise-manifest"})
            write_new_json(run_path, {
                "protocol_sha256": sha256_file(protocol_path),
                "noise_bank_manifest_sha256": sha256_file(noise_path),
                "require_explicit_noise": True,
            })
            with ledger_path.open("x", encoding="utf-8") as handle:
                for spec in expanded_specs(smoke).values():
                    row = {
                        **spec, "status": "complete", "explicit_flow_noise": True,
                        "inference_calls": 1, "success": False,
                        "initial_metrics": {
                            "physics_state_sha256": "9" * 64,
                            "post_wait_physics_state_sha256_exact": "9" * 64,
                        },
                        "policy_calls": [{
                            "replan_index": 0, "policy_repeat_id": spec["policy_repeat_id"],
                            "noise_seed": spec["policy_repeat_id"] * 1000,
                            "noise_sha256": "8" * 64, "action_chunk_sha256": "6" * 64,
                        }],
                    }
                    handle.write(json.dumps(row) + "\n")
            with patch.object(old_auditor, "ExplicitFlowNoiseBank", FakeVerifiedNoiseBank):
                receipt = old_auditor.audit(
                    protocol_path=protocol_path, noise_manifest=noise_path,
                    run_manifest_path=run_path, ledger_patterns=[str(ledger_path)],
                )
            self.assertEqual(receipt["status"], "PASS_COMPLETE")
            self.assertEqual(receipt["episode_count"], 48)
            self.assertEqual(receipt["state_count"], 8)
            self.assertEqual(receipt["candidate_counts"], [3])


if __name__ == "__main__":
    unittest.main()
