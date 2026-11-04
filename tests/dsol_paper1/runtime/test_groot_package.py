"""Import-surface packaging must not modify the scientific implementation."""
from pathlib import Path
import hashlib
import pytest
from benchmarks.LIBERO import groot_package


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path/'upstream'; source.mkdir()
    for name in groot_package.IMPORT_SURFACE:
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Original copyright\nimport unrelated_driver\n')
    (source/'fluxvla/__init__.py').write_text('# Copyright\nassert True\nfrom .collators import *\n')
    head = source/'fluxvla/models/heads/flow_matching_inference_head.py'
    head.write_text('def native_step(x, v):\n    return x + 0.25 * v\n')
    monkeypatch.setattr(groot_package, 'UPSTREAM_SHA256', hashlib.sha256(head.read_bytes()).hexdigest())
    return source, head


def test_candidate_changes_only_initializers_and_preserves_original(tmp_path, monkeypatch):
    source, head = source_fixture(tmp_path, monkeypatch)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*.py')}
    dest = tmp_path/'candidate'
    result = groot_package.prepare(source, dest)
    assert head.read_bytes() == (dest/head.relative_to(source)).read_bytes()
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*.py')}
    assert len(result['changed_initializers']) == len(groot_package.IMPORT_SURFACE) + 1
    assert all(Path(p).name == '__init__.py' for p in result['changed_initializers'])
    assert result['gpu_admitted'] is False
    assert 'assert True' in (dest/'fluxvla/__init__.py').read_text()
    assert '# Original copyright' in (dest/'fluxvla/engines/__init__.py').read_text()
    with pytest.raises(ValueError, match='must be new'): groot_package.prepare(source, dest)


def test_source_revision_change_rejected_before_copy(tmp_path, monkeypatch):
    source, head = source_fixture(tmp_path, monkeypatch)
    head.write_text('raise RuntimeError("changed")\n')
    dest = tmp_path/'candidate'
    with pytest.raises(ValueError, match='Unreviewed'): groot_package.prepare(source, dest)
    assert not dest.exists()
