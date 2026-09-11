from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.dsol_paper1.audit_dsol_training_match import (
    MANIFEST_MATCH_FIELDS, compare_runs, config_differences, inspect_run, propose_canonical,
    sha256_file,
)


def fixture(root: Path, name="broad"):
    run = root / name
    (run / "final_model").mkdir(parents=True)
    data, initial = root / "data", root / "initial"
    data.mkdir(exist_ok=True)
    initial.mkdir(exist_ok=True)
    (data / "manifest.json").write_text('{"source": "same"}')
    (initial / "source_manifest.json").write_text('{"initial": "same"}')
    arm = "broad_unpaired_practical" if name == "broad" else "canonical_unique"
    config = {
        "seed": 41, "run_id": name, "output_dir": str(run), "output_root_dir": str(root),
        "framework": {"pi05": True, "state_input": False},
        "trainer": {"max_train_steps": 2000, "scheduler_total_steps": 2000,
                    "gradient_accumulation_steps": 16, "is_resume": False,
                    "pretrained_checkpoint": str(initial), "optimizer": {"beta": [0.9, 0.95]}},
        "datasets": {"vla_data": {"data_root_dir": str(data), "dsol_arm": arm,
                                    "examples_per_item": 1, "per_device_batch_size": 1}},
    }
    manifest = {
        "arm": arm, "seed": 41, "num_gpus": 2, "steps": 2000, "scheduler_total_steps": 2000,
        "global_model_examples_per_update": 32, "source_data_items_per_update": 32,
        "examples_per_data_item": 1, "gradient_accumulation_steps": 16,
        "data_root": str(data), "data_manifest_sha256": sha256_file(data / "manifest.json"),
        "pretrained_checkpoint_manifest": str(initial / "source_manifest.json"),
        "pretrained_checkpoint_manifest_sha256": sha256_file(initial / "source_manifest.json"),
        "python": "3.12.3", "calibration": {"enabled": False}, "skip_final_save": False,
        "critical_code_sha256": {"train.py": "0" * 64},
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "final_model/framework_config.yaml").write_text(yaml.safe_dump(config))
    (run / "final_model/model.safetensors").write_bytes(b"test only")
    (run / "metrics.jsonl").write_text(json.dumps({"step": 2000, "examples_seen": 64000, "learning_rate": 5e-6}) + "\n")
    return run


class TrainingMatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.broad = inspect_run(fixture(self.root))
        self.canonical = inspect_run(fixture(self.root, "canonical"))

    def test_matched_recipe_is_not_formal_release(self):
        report = compare_runs(self.broad, self.canonical)
        self.assertTrue(report["saved_recipe_match"])
        self.assertFalse(report["formal_release"])
        self.assertFalse(report["left"]["model_weights"]["content_sha256_verified"])

    def test_all_manifest_invariants_detect_changes(self):
        for key in MANIFEST_MATCH_FIELDS:
            with self.subTest(field=key):
                changed = copy.deepcopy(self.canonical)
                changed["manifest"][key] = "different"
                self.assertFalse(compare_runs(self.broad, changed)["saved_recipe_match"])

    def test_equal_updates_different_schedule_rejected(self):
        self.canonical["config"]["trainer"]["scheduler_total_steps"] = 3000
        report = compare_runs(self.broad, self.canonical)
        self.assertFalse(report["saved_recipe_match"])
        self.assertIn("/trainer/scheduler_total_steps", {x["path"] for x in report["config_differences"]})

    def test_initialization_interface_optimizer_changes_rejected(self):
        for key, value in (("pretrained_checkpoint", "old/canonical"),
                           ("is_resume", True), ("optimizer", {"beta": [0.8, 0.99]})):
            with self.subTest(key=key):
                modified = copy.deepcopy(self.canonical)
                modified["config"]["trainer"][key] = value
                self.assertFalse(compare_runs(self.broad, modified)["saved_recipe_match"])
        self.canonical["config"]["framework"]["state_input"] = True
        self.assertFalse(compare_runs(self.broad, self.canonical)["saved_recipe_match"])

    def test_missing_key_is_not_equal_to_null(self):
        self.assertTrue(config_differences({"x": None}, {}))

    def test_same_arm_is_not_treatment_pair(self):
        self.assertFalse(compare_runs(self.broad, self.broad)["saved_recipe_match"])

    def test_code_difference_is_separately_reported(self):
        self.canonical["manifest"]["critical_code_sha256"]["train.py"] = "1" * 64
        report = compare_runs(self.broad, self.canonical)
        self.assertTrue(report["saved_recipe_match"])
        self.assertFalse(report["critical_code_identical"])
        self.assertFalse(report["formal_release"])

    def test_actual_small_artifact_tamper_blocks_recipe(self):
        (self.root / "data/manifest.json").write_text("{}")
        changed = inspect_run(Path(self.broad["run_dir"]))
        self.assertTrue(changed["issues"])
        self.assertFalse(compare_runs(changed, self.canonical)["saved_recipe_match"])

    def test_actual_metric_budget_mismatch_detected(self):
        (Path(self.broad["run_dir"]) / "metrics.jsonl").write_text('{"step": 100, "examples_seen": 3200}')
        self.assertTrue(inspect_run(Path(self.broad["run_dir"]))["issues"])

    def test_proposal_preserves_original_initialization_and_never_creates_run(self):
        report = propose_canonical(self.broad, run_id="new-canonical", output_root=self.root / "runs", repo_root=self.root)
        self.assertEqual(report["status"], "DRAFT_NOT_TRAINED")
        self.assertFalse(report["formal_release"])
        self.assertFalse((self.root / "runs").exists())
        proposed = report["proposed_config"]
        self.assertEqual(proposed["trainer"], self.broad["config"]["trainer"])
        self.assertEqual(proposed["datasets"]["vla_data"]["dsol_arm"], "canonical_unique")
        self.assertFalse(config_differences(self.broad["config"], proposed))

    def test_bad_proposal_anchor_or_output_refused(self):
        with self.assertRaises(ValueError):
            propose_canonical(self.canonical, run_id="new", output_root=self.root, repo_root=self.root)
        for run_id in ("../escape", "broad", "with space", ""):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                propose_canonical(self.broad, run_id=run_id, output_root=self.root, repo_root=self.root)


if __name__ == "__main__":
    unittest.main()
