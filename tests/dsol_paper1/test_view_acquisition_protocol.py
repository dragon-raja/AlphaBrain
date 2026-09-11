"""Synthetic, stdlib-only tests. Frozen fixtures are NOT real release approvals."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from AlphaBrain.research.dsol.acquisition.protocol import (
    GATES,
    ProtocolValidationError,
    build_plan,
    load_protocol,
    protocol_digest,
    require_release,
    validate_protocol,
)


REPO = Path(__file__).resolve().parents[2]
DRAFT = REPO / "configs/dsol_paper1/view_acquisition_a1_draft_v1.json"
SCRIPT = REPO / "AlphaBrain/research/dsol/acquisition/protocol.py"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def resolved_fixture() -> dict:
    """Resolved declarations only: no simulator, dataset or checkpoint is accessed."""
    config = load_protocol(DRAFT)
    config["status"] = "FROZEN"
    config["unresolved"] = []
    config["release"] = {"approved_by": "synthetic unit test, not operational approval",
                         "frozen_at_utc": "2026-09-07T00:00:00Z"}
    config["gates"] = {gate: True for gate in GATES}
    for index, treatment in enumerate(config["treatments"]):
        treatment.update(checkpoint_path=f"/synthetic/{treatment['id']}/model",
                         checkpoint_sha256=sha(f"checkpoint-{index}"),
                         training_manifest_sha256=sha(f"training-manifest-{index}"),
                         initial_checkpoint_sha256=sha("same-initial"),
                         training_sources_sha256=sha("same-sorted-source-ids"),
                         action_labels_sha256=sha("same-action-labels"),
                         unique_states=100, image_exposures=2000, optimizer_updates=2000)
        treatment["optimizer_contract"] = {
            "name": "adamw", "schedule": "cosine", "peak_learning_rate": 0.0001,
            "end_learning_rate": 0.000001, "weight_decay": 0.01,
            "schedule_steps": 2000, "warmup_steps": 100,
        }
    for index, bank in enumerate(config["noise"]["banks"]):
        bank.update(seed=100 + index, sha256=sha(f"bank-{index}"))
    config["sources"] = [
        {"source_id": "dev-1", "cluster_id": "layout-1", "task_id": "task-a", "split": "development",
         "branches": ["z0", "z1"], "seen_in_prior_research": True},
        {"source_id": "dev-2", "cluster_id": "layout-2", "task_id": "task-b", "split": "development",
         "branches": ["z0", "z1"], "seen_in_prior_research": False},
        {"source_id": "cal-1", "cluster_id": "layout-3", "task_id": "task-a", "split": "calibration",
         "branches": ["z0", "z1"], "seen_in_prior_research": False},
        {"source_id": "test-1", "cluster_id": "layout-4", "task_id": "task-b", "split": "confirmation",
         "branches": ["z0", "z1"], "seen_in_prior_research": False},
    ]
    config["source_manifest_sha256"] = sha("synthetic-source-manifest")
    for split, count in (("development", 2), ("calibration", 1), ("confirmation", 1)):
        config["planning_counts"][split] = {"base_clusters": count, "sources": count, "branches_per_source": 2}
    config["environment"].update(control_frequency_hz=20, total_steps=200, trigger_step=5,
                                 max_hold_position_error_m=0.002, max_hold_rotation_error_rad=0.01)
    config["acquisition"]["speed_limits"] = {
        "linear_m_s": 1.0, "angular_rad_s": 1.0,
        "linear_acceleration_m_s2": 2.0, "angular_acceleration_rad_s2": 2.0,
    }
    config["acquisition"]["workspace_bounds"] = {"min_xyz": [-2, -2, 0], "max_xyz": [2, 2, 2]}
    for index, action in enumerate(config["acquisition"]["actions"]):
        if action["kind"] != "continue":
            action["duration_steps"] = 10
        if action["kind"] == "move":
            action["target_pose"] = {"position_xyz": [index / 10, 0, 1], "quaternion_wxyz": [1, 0, 0, 0]}
    config["evaluation_groups"].append({
        "id": "test-frozen-rules", "split": "confirmation", "kind": "rules", "bank_id": "Q",
        "member_ids": config["rule_ids"][:], "frozen_rule_manifest_sha256": sha("frozen-rules"),
        "rules_frozen_before_confirmation": True,
    })
    for contrast in config["primary_contrasts"]:
        contrast["minimum_effect"] = 0.03
    config["analysis"].update(interval_method="predeclared cluster bootstrap",
                              multiplicity="predeclared Holm primary family",
                              cost_weights={"elapsed_seconds": 0.01, "observation_queries": 0.001})
    config["budget"]["max_episodes"] = 2304
    return config


class ViewAcquisitionProtocolTests(unittest.TestCase):
    def assertCode(self, config: dict, code: str) -> None:
        issues = validate_protocol(config, for_release=True)
        self.assertIn(code, {x.code for x in issues}, issues)
        with self.assertRaises(ProtocolValidationError):
            require_release(config)

    def test_draft_plans_but_never_releases(self) -> None:
        config = load_protocol(DRAFT)
        plan = build_plan(config)
        self.assertTrue(plan["draft_schema_valid"], plan["issues"])
        self.assertFalse(plan["release_valid"])
        self.assertFalse(plan["launch_enabled"])
        self.assertEqual(plan["planned_episodes"], 10752)
        self.assertEqual([x["episodes"] for x in plan["groups"]], [9216, 1536])
        self.assertEqual(plan["resolved_unique_clusters"], 0)
        self.assertEqual(plan["count_basis"], "draft_planning_counts_only")
        self.assertEqual(plan["groups"][0]["base_clusters"], 16)
        self.assertEqual(plan["groups"][0]["branch_instances"], 32)
        with self.assertRaises(ProtocolValidationError):
            require_release(config)

    def test_complete_synthetic_fixture_passes_without_launch(self) -> None:
        config = resolved_fixture()
        self.assertEqual(validate_protocol(config, for_release=True), [])
        plan = require_release(config)
        self.assertTrue(plan["release_valid"])
        self.assertFalse(plan["launch_enabled"])
        self.assertEqual(plan["planned_episodes"], 2304)
        self.assertEqual(plan["resolved_unique_clusters"], 4)
        self.assertEqual(plan["resolved_source_records"], 4)

    def test_digest_is_stable_and_changes_with_contract(self) -> None:
        config = resolved_fixture()
        self.assertEqual(protocol_digest(config), protocol_digest(dict(reversed(list(config.items())))))
        changed = copy.deepcopy(config)
        changed["environment"]["trigger_step"] += 1
        self.assertNotEqual(protocol_digest(config), protocol_digest(changed))

    def test_missing_and_undeclared_unresolved_fields_fail_closed(self) -> None:
        config = resolved_fixture()
        del config["environment"]["total_steps"]
        self.assertCode(config, "MISSING_FIELD")
        config = resolved_fixture()
        config["treatments"][0]["checkpoint_path"] = None
        self.assertCode(config, "UNDECLARED_UNRESOLVED")

    def test_bad_or_stale_unresolved_pointers_rejected(self) -> None:
        for pointer, code in (("/does/not/exist", "BAD_UNRESOLVED_POINTER"),
                              ("/environment/total_steps", "STALE_UNRESOLVED_POINTER")):
            config = resolved_fixture()
            config["unresolved"] = [pointer]
            self.assertCode(config, code)

    def test_no_release_by_changing_only_draft_status(self) -> None:
        config = load_protocol(DRAFT)
        config["status"] = "FROZEN"
        self.assertCode(config, "UNRESOLVED_RELEASE")

    def test_audit_gates_each_fail_closed(self) -> None:
        for gate in GATES:
            with self.subTest(gate=gate):
                config = resolved_fixture()
                config["gates"][gate] = False
                self.assertCode(config, "INVALID_VALUE")

    def test_matching_training_inputs_and_schedule(self) -> None:
        for field, value in (("initial_checkpoint_sha256", sha("different")),
                             ("image_exposures", 1999), ("unique_states", 99)):
            config = resolved_fixture()
            config["treatments"][0][field] = value
            self.assertCode(config, "TREATMENT_MATCH")
        config = resolved_fixture()
        config["treatments"][0]["optimizer_contract"]["schedule_steps"] = 3000
        self.assertCode(config, "OPTIMIZER_MISMATCH")
        config = resolved_fixture()
        config["treatments"][0]["input_contract_id"] = "other-interface"
        self.assertCode(config, "INPUT_CONTRACT_MISMATCH")

    def test_different_manifest_hashes_are_not_optimizer_mismatch(self) -> None:
        config = resolved_fixture()
        self.assertNotEqual(config["treatments"][0]["training_manifest_sha256"],
                            config["treatments"][1]["training_manifest_sha256"])
        self.assertTrue(require_release(config)["release_valid"])

    def test_noise_seed_and_hash_reuse_and_keys(self) -> None:
        for field in ("seed", "sha256"):
            config = resolved_fixture()
            config["noise"]["banks"][2][field] = config["noise"]["banks"][0][field]
            self.assertCode(config, "NOISE_BANK_REUSE")
        config = resolved_fixture()
        config["noise"]["key_fields"][-1] = "global_rng_call"
        self.assertCode(config, "NOISE_KEY")
        config = resolved_fixture()
        config["treatments"][1]["noise_bank_ids"] = ["O"]
        self.assertCode(config, "NOISE_TREATMENT_MISMATCH")

    def test_source_and_cluster_split_isolation(self) -> None:
        config = resolved_fixture()
        config["sources"][-1]["cluster_id"] = "layout-1"
        self.assertCode(config, "CLUSTER_LEAKAGE")
        config = resolved_fixture()
        config["sources"][-1]["source_id"] = "dev-1"
        self.assertCode(config, "DUPLICATE_ID")
        config = resolved_fixture()
        config["sources"][-1]["seen_in_prior_research"] = True
        self.assertCode(config, "HISTORICAL_CONFIRMATION")

    def test_planning_counts_cannot_replace_resolved_sources(self) -> None:
        config = resolved_fixture()
        config["sources"] = []
        self.assertCode(config, "EMPTY_SOURCE_MANIFEST")
        config = resolved_fixture()
        config["evaluation_groups"] = []
        self.assertCode(config, "NO_EVALUATION_GROUPS")

    def test_branch_instances_never_count_as_independent_layouts(self) -> None:
        config = resolved_fixture()
        source = copy.deepcopy(config["sources"][0])
        source["source_id"] = "dev-1-another-frame"
        config["sources"].append(source)
        config["planning_counts"]["development"]["sources"] = 3
        config["budget"]["max_episodes"] = 2976
        plan = require_release(config)
        self.assertEqual(plan["resolved_source_records"], 5)
        self.assertEqual(plan["resolved_unique_clusters"], 4)
        self.assertEqual(plan["groups"][0]["base_clusters"], 2)
        self.assertEqual(plan["groups"][0]["branch_instances"], 6)

    def test_confirmation_bank_and_rule_freeze(self) -> None:
        config = resolved_fixture()
        config["evaluation_groups"][-1]["bank_id"] = "O"
        self.assertCode(config, "CONFIRMATION_BANK")
        config = resolved_fixture()
        config["evaluation_groups"][-1]["rules_frozen_before_confirmation"] = False
        self.assertCode(config, "INVALID_VALUE")

    def test_duration_and_continuation_budget(self) -> None:
        config = resolved_fixture()
        config["acquisition"]["actions"][-1]["duration_steps"] = 11
        self.assertCode(config, "DURATION_MISMATCH")
        config = resolved_fixture()
        config["acquisition"]["actions"][0]["duration_steps"] = 10
        self.assertCode(config, "CONTINUE_DELAY")
        config = resolved_fixture()
        config["environment"]["total_steps"] = 15
        self.assertCode(config, "HORIZON_EXHAUSTED")

    def test_action_count_pose_and_workspace(self) -> None:
        config = resolved_fixture()
        config["acquisition"]["actions"].pop()
        self.assertCode(config, "ACTION_SET")
        config = resolved_fixture()
        config["acquisition"]["actions"][-1]["target_pose"]["quaternion_wxyz"] = [0, 0, 0, 0]
        self.assertCode(config, "QUATERNION_NORM")
        config = resolved_fixture()
        config["acquisition"]["actions"][-1]["target_pose"]["position_xyz"] = [99, 0, 0]
        self.assertCode(config, "TARGET_OUTSIDE_WORKSPACE")

    def test_privileged_and_unknown_query_inputs_rejected(self) -> None:
        config = resolved_fixture()
        config["input_contract"]["selector_inputs"].append("hidden_branch")
        self.assertCode(config, "PRIVILEGED_INPUT")
        config = resolved_fixture()
        config["input_contract"]["selector_inputs"].append("unreviewed_feature")
        self.assertCode(config, "UNKNOWN_INPUT")
        config = resolved_fixture()
        config["input_contract"]["along_path_images"] = True
        self.assertCode(config, "INVALID_VALUE")
        config = resolved_fixture()
        config["acquisition"]["actions"][0]["read_unrequested_views"] = True
        self.assertCode(config, "UNKNOWN_ACTION_FIELD")

    def test_budget_and_evidence_blocks(self) -> None:
        config = resolved_fixture()
        config["budget"]["max_episodes"] -= 1
        self.assertCode(config, "EPISODE_CAP")
        config = resolved_fixture()
        config["evidence_blocks"].remove("matched_negative")
        self.assertCode(config, "EVIDENCE_BLOCKS")
        config = resolved_fixture()
        config["evaluation_groups"].append({**config["evaluation_groups"][0], "id": "duplicate-condition"})
        self.assertCode(config, "DUPLICATE_EVALUATION")

    def test_numeric_bools_nonfinite_and_missing_contrasts(self) -> None:
        config = resolved_fixture()
        config["environment"]["total_steps"] = True
        self.assertCode(config, "INVALID_VALUE")
        config = resolved_fixture()
        config["acquisition"]["speed_limits"]["linear_m_s"] = float("nan")
        self.assertCode(config, "NON_JSON_VALUE")
        config = resolved_fixture()
        config["primary_contrasts"] = []
        self.assertCode(config, "NO_PRIMARY_CONTRASTS")

    def test_malformed_shapes_report_errors_not_crashes(self) -> None:
        for key, value in (("sources", {}), ("evaluation_groups", [None]), ("noise", []),
                           ("planning_counts", []), ("treatments", [None])):
            with self.subTest(key=key):
                config = resolved_fixture()
                config[key] = value
                plan = build_plan(config)
                self.assertFalse(plan["release_valid"])
        self.assertEqual(validate_protocol([])[0].code, "ROOT_TYPE")

    def test_strict_json_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            for value in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": 1e999}', '[]'):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_protocol(path)

    def test_cli_draft_success_release_failure_without_launch(self) -> None:
        common = [sys.executable, str(SCRIPT), str(DRAFT)]
        for extra, expected in (([], 0), (["--require-release"], 2)):
            run = subprocess.run(common + extra, capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, expected, run.stderr + run.stdout)
            plan = json.loads(run.stdout)
            self.assertFalse(plan["launch_enabled"])
            self.assertEqual(plan["planned_episodes"], 10752)


if __name__ == "__main__":
    unittest.main()
