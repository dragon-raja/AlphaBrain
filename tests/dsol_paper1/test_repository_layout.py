"""Checks for source preservation and the non-GPU maintenance entrypoints."""
import json
from pathlib import Path
from unittest.mock import patch

from tools.paper1 import check_layout as layout
from tools.paper1 import test as runner


def test_relocated_files_and_archive_bytes_are_preserved():
    # Future scientific edits are allowed; the one-time baseline comparison is
    # a separate acceptance command, not a permanent freeze of development.
    assert layout.check() == []


def test_retired_operations_are_not_runtime_dependencies():
    manifest = json.loads(layout.MANIFEST.read_text())
    retired = {layout.ROOT / m["old"] for m in manifest["moves"] if m["new"].startswith("archive/")}
    closure = layout.runtime_closure(layout.dependencies())
    assert not retired & closure
    assert all(p.is_file() for p in closure)
    assert layout.ROOT / "scripts/cabi_vla/serve_alphabrain_pi05_websocket.py" in closure


def test_moved_tests_remain_in_same_repository_relative_depth():
    manifest = json.loads(layout.MANIFEST.read_text())
    tests = [m for m in manifest["moves"] if m["new"].startswith("tests/")]
    assert len(tests) == 21
    assert all(len(Path(m["old"]).parts) == len(Path(m["new"]).parts) for m in tests)


def test_cpu_runner_sets_environment_without_starting_research():
    with patch.object(runner.subprocess, "call", return_value=0) as run:
        with patch.object(runner.sys, "argv", ["test.py", "--collect-only"]):
            assert runner.main() == 0
    args, kwargs = run.call_args
    assert args[0][1:3] == ["-m", "pytest"]
    assert args[0][-1] == "--collect-only"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["cwd"] == runner.ROOT
