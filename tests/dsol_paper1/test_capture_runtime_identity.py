from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "dsol_paper1" / "capture_runtime_identity.py"
SPEC = importlib.util.spec_from_file_location("capture_runtime_identity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
capture_runtime_identity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture_runtime_identity
SPEC.loader.exec_module(capture_runtime_identity)


class CaptureRuntimeIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self._git("init", "--quiet")
        self._git("config", "user.name", "Runtime Identity Test")
        self._git("config", "user.email", "runtime-identity@example.invalid")
        (self.repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        self._git("add", "tracked.txt")
        self._git("commit", "--quiet", "-m", "initial")
        self._git("remote", "add", "origin", "https://example.invalid/alphabrain.git")

        self.artifact = self.root / "artifact.bin"
        self.artifact.write_bytes(b"paper-1-artifact\n")
        self.output = self.root / "receipt.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", "-C", os.fspath(self.repo), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _run(self, *extra_arguments: str) -> int:
        arguments = [
            "--output",
            os.fspath(self.output),
            "--repo",
            os.fspath(self.repo),
            *extra_arguments,
        ]
        gpu_info = {"gpus": [], "query_status": "unavailable"}
        with mock.patch.object(capture_runtime_identity, "_collect_gpu_info", return_value=gpu_info):
            return capture_runtime_identity.main(arguments)

    def _receipt(self) -> dict[str, object]:
        return json.loads(self.output.read_text(encoding="utf-8"))

    def test_clean_repository_and_present_artifact_pass(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0", "API_TOKEN": "must-not-be-captured"},
            clear=False,
        ):
            return_code = self._run("--artifact", f"weights={self.artifact}")

        self.assertEqual(return_code, 0)
        receipt = self._receipt()
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(receipt["issues"], [])
        self.assertTrue(receipt["repositories"][0]["clean"])
        self.assertEqual(receipt["repositories"][0]["remote_names"], ["origin"])
        self.assertNotIn("example.invalid", self.output.read_text(encoding="utf-8"))
        self.assertEqual(receipt["runtime"]["environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("API_TOKEN", receipt["runtime"]["environment"])
        self.assertNotIn("must-not-be-captured", self.output.read_text(encoding="utf-8"))

        artifact = receipt["artifacts"][0]
        self.assertEqual(artifact["path"], os.fspath(self.artifact.resolve()))
        self.assertEqual(artifact["size_bytes"], self.artifact.stat().st_size)
        self.assertEqual(artifact["sha256"], hashlib.sha256(self.artifact.read_bytes()).hexdigest())

    def test_dirty_repository_is_hold_even_with_explicit_exception_label(self) -> None:
        (self.repo / "tracked.txt").write_text("changed\n", encoding="utf-8")

        return_code = self._run(
            "--artifact",
            f"weights={self.artifact}",
            "--allow-dirty-label",
            "local-changes",
        )

        self.assertEqual(return_code, 1)
        receipt = self._receipt()
        self.assertEqual(receipt["status"], "HOLD")
        self.assertFalse(receipt["repositories"][0]["clean"])
        self.assertIn("repository_dirty", {issue["code"] for issue in receipt["issues"]})
        self.assertEqual(receipt["exceptions"][0]["labels"], ["local-changes"])
        self.assertEqual(receipt["exceptions"][0]["type"], "dirty_repository_hold_only")

    def test_missing_artifact_writes_hold_receipt(self) -> None:
        missing = self.root / "missing.bin"

        return_code = self._run("--artifact", f"weights={missing}")

        self.assertEqual(return_code, 1)
        receipt = self._receipt()
        self.assertEqual(receipt["status"], "HOLD")
        self.assertFalse(receipt["artifacts"][0]["exists"])
        self.assertIsNone(receipt["artifacts"][0]["size_bytes"])
        self.assertIsNone(receipt["artifacts"][0]["sha256"])
        self.assertIn("artifact_missing_or_unreadable", {issue["code"] for issue in receipt["issues"]})

    def test_sensitive_artifact_labels_and_paths_are_rejected_without_receipt(self) -> None:
        unsafe_arguments = (
            ("--artifact", f"auth_token={self.artifact}"),
            ("--artifact", f"weights={self.root / '.ssh' / 'id_rsa'}"),
            ("--artifact", f"weights={self.root / 'model_key.bin'}"),
        )
        for arguments in unsafe_arguments:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()) as stderr:
                return_code = self._run(*arguments)
            self.assertEqual(return_code, 2)
            self.assertEqual(stderr.getvalue(), "error: unsafe label or path rejected\n")
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
