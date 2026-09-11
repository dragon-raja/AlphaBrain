from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import copy
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


REPO = REPOSITORY_ROOT
SHELL = REPO / "scripts/dsol_paper1/run_matched_view_landscape_v1.sh"


def controller_module():
    module = types.ModuleType("landscape_controller_test")
    source = SHELL.read_text().split("<<'PY_CONTROLLER'\n", 1)[1].rsplit("\nPY_CONTROLLER", 1)[0]
    with patch.dict(os.environ, {"DSOL_LANDSCAPE_REPO_ROOT": str(REPO)}):
        exec(compile(source, str(SHELL), "exec"), module.__dict__)
    return module


class MatchedViewControllerTest(unittest.TestCase):
    def setUp(self):
        self.module = controller_module()

    def test_route_is_exact_nonoverlapping_budget(self):
        rows = [{"name": f"dense-O-development-wave-{index:02d}.json", "episode_count": 3104} for index in range(8)]
        rows += [{"name": "smoke-O-development.json", "episode_count": 48},
                 {"name": "dense-O-development-wave-00-remainder-a.json", "episode_count": 1504},
                 {"name": "dense-O-development-wave-00-remainder-b.json", "episode_count": 1552}]
        route = self.module.route_specs({"protocols": rows})
        self.assertEqual(sum(row["episode_count"] for row in route), 24832)
        self.assertEqual(len(route), 10)
        self.assertNotIn("dense-O-development-wave-00.json", [row["name"] for row in route])

    def test_empty_output_can_start(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(self.module.reusable(Path(directory), {}, {}, {}))

    def test_partial_output_never_automatically_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "episodes-shard-0.jsonl").touch()
            with self.assertRaisesRegex(ValueError, "no automatic retry"):
                self.module.reusable(path, {}, {}, {})

    def test_failed_receipt_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.module.write_json(path / "segment-receipt.json", {"status": "FAIL", "identity": {}})
            with self.assertRaises(ValueError):
                self.module.reusable(path, {}, {}, {})

    def test_changed_identity_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.module.write_json(path / "segment-receipt.json", {"status": "PASS_COMPLETE", "identity": {"code": "old"}})
            with self.assertRaises(ValueError):
                self.module.reusable(path, {"code": "new"}, {}, {})

    def test_receipts_are_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            self.module.write_json(path, {}, exclusive=True)
            with self.assertRaises(FileExistsError):
                self.module.write_json(path, {"overwrite": True}, exclusive=True)
            self.assertEqual(self.module.read(path), {})

    def test_exhausted_budget_does_not_start_process(self):
        with patch.object(self.module.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                self.module.run_command(["unused"], {}, Path("unused"), 0)
            process.assert_not_called()

    def test_interrupt_only_signals_owned_timeout(self):
        process = Mock()
        process.poll.return_value = None
        self.module.ACTIVE = process
        self.module.stop(15, None)
        self.assertTrue(self.module.STOPPED)
        process.terminate.assert_called_once_with()

    def test_manifest_rejects_wrong_checkpoint_or_noise(self):
        release = {"canonical_checkpoint": {"path": "/checkpoint", "weights_sha256": "weights"},
                   "noise_bank": {"manifest_sha256": "noise"}, "runner_code_sha256": "code"}
        manifest = {"checkpoint": "/checkpoint", "checkpoint_sha256": "weights", "protocol_sha256": "protocol",
                    "policy_backend": "alphabrain", "noise_bank_manifest_sha256": "noise", "code_sha256": "code",
                    "require_explicit_noise": True, "gpu_count": 8, "eval_worker_count": 32,
                    "policy_server_copies_per_gpu": 2, "replan_steps": 5, "wait_steps": 0, "eval_seed": 20260818,
                    "video_episodes_per_worker": 0, "max_episodes_per_shard": None, "keepalive_mode": "managed", "run_analysis": False}
        self.module.validate_manifest(manifest, release, {"sha256": "protocol"})
        for key in ("checkpoint_sha256", "noise_bank_manifest_sha256", "code_sha256", "protocol_sha256"):
            changed = copy.deepcopy(manifest)
            changed[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.module.validate_manifest(changed, release, {"sha256": "protocol"})

    def test_outer_timeout_and_explicit_mode_are_present(self):
        source = SHELL.read_text()
        self.assertIn("timeout --signal=TERM --kill-after=120s 129600s", source)
        self.assertIn("add_mutually_exclusive_group()", source)
        self.assertNotIn("pkill ", source)
        self.assertNotIn("killall ", source)


if __name__ == "__main__":
    unittest.main()
