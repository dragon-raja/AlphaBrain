"""Checks for source preservation and the non-GPU maintenance entrypoints."""
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
from unittest.mock import patch
from urllib.parse import unquote

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
    assert layout.ROOT / "scripts/vla_shared/serve_alphabrain_pi05_websocket.py" in closure


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


def test_archived_document_bytes_match_relocation_receipt():
    manifest = json.loads(layout.MANIFEST.read_text())
    records = manifest['documentation_archive']['records']
    assert len(records) == 95
    for record in records:
        assert not (layout.ROOT/record['old']).exists()
        from tools.repository.evidence import current_expected_hash
        relative, expected = current_expected_hash(record['new'], record['after_sha256'])
        target=layout.ROOT/relative
        assert hashlib.sha256(target.read_bytes()).hexdigest()==expected


def test_archival_does_not_introduce_broken_markdown_file_links():
    import pytest
    manifest=json.loads(layout.MANIFEST.read_text())
    baseline=manifest['shared_extraction']['baseline_commit']
    result=subprocess.run(['git','ls-tree','-r','--name-only',baseline],cwd=layout.ROOT,capture_output=True,text=True)
    if result.returncode:pytest.skip('Historical Git object unavailable')
    original_files=set(result.stdout.splitlines())
    original_paths=set(original_files)
    for name in original_files:
        original_paths.update(str(p) for p in Path(name).parents)
    pattern=re.compile(r'\]\(([^\n)]*)\)')
    def links(text):
        values=[]
        for match in pattern.finditer(text):
            target=match[1]
            token=target[1:target.index('>')] if target.startswith('<') and '>' in target else target.split(' ',1)[0]
            values.append(token.split('#',1)[0])
        return values
    def resolve(source,target):
        return Path(os.path.normpath(str(source.parent/unquote(target))))
    for record in manifest['documentation_archive']['records']:
        if not record['old'].endswith('.md'):continue
        before=subprocess.check_output(['git','show',baseline+':'+record['old']],cwd=layout.ROOT,text=True)
        after=(layout.ROOT/record['new']).read_text()
        assert pattern.sub('](LINK)',before)==pattern.sub('](LINK)',after)
        left,right=links(before),links(after)
        assert len(left)==len(right)
        for a,b in zip(left,right):
            if not a or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',a):continue
            old=resolve(layout.ROOT/record['old'],a)
            if not old.is_relative_to(layout.ROOT):continue
            if str(old.relative_to(layout.ROOT)) in original_paths:
                assert resolve(layout.ROOT/record['new'],b).exists(), (record['new'],b)
