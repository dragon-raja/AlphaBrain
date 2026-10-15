import json
import pytest
from AlphaBrain.research.dsol.data.resource_catalog import audit, audit_file, publish_catalog, check_pointer, resource_release
from AlphaBrain.research.dsol.data.resource_import import digest, run


def test_catalog_preserves_original(tmp_path):
    path = tmp_path / 'release.json'
    path.write_text('{"old_path": "/retired/path"}\n')
    before = path.read_bytes()
    result = audit({'entries': [{'id': 'x', 'benchmark': 'test', 'role': 'frozen',
                                'root': str(tmp_path), 'anchors': ['release.json']}]})
    assert result['status'] == 'PASS'
    assert result['entries'][0]['metadata'][0]['sha256'] == digest(path)
    assert path.read_bytes() == before


def test_missing_anchor_is_not_admitted(tmp_path):
    result = audit({'entries': [{'id': 'x', 'benchmark': 'test', 'role': 'frozen',
                                'root': str(tmp_path), 'anchors': ['absent.json']}]})
    assert result['status'] == 'INCOMPLETE'
    assert result['errors']


def test_release_binds_receipts(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'file').write_bytes(b'data')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'benchmark': 'test', 'resources': [
        {'id': 'one', 'kind': 'copy_tree', 'source': str(source), 'destination': 'raw'}]}))
    record = tmp_path / 'record'
    run(plan, tmp_path, record)
    release = resource_release(plan, [record])
    assert release['status'] == 'BYTE_VERIFIED'
    assert not release['runtime_validated']
    (tmp_path / 'raw/IMPORT_RECEIPT.json').write_text('{}')
    with pytest.raises(ValueError, match='receipt changed'):
        resource_release(plan, [record])


def test_catalog_mirror_drift_and_history(tmp_path):
    mirror = tmp_path / 'shared/catalog.json'
    path = tmp_path / 'catalog.json'
    config = {'entries': [], 'shared_catalog': str(mirror)}
    path.write_text(json.dumps(config))
    assert audit_file(path)['status'] == 'INCOMPLETE'
    publish_catalog(path)
    previous = mirror.read_bytes()
    previous_sha = digest(mirror)
    config['scope'] = 'new version'
    path.write_text(json.dumps(config))
    assert audit_file(path)['status'] == 'INCOMPLETE'
    publish_catalog(path)
    assert audit_file(path)['status'] == 'PASS'
    assert (mirror.parent / 'catalog_history' / (previous_sha + '.json')).read_bytes() == previous


def test_retired_active_path_rejected(tmp_path):
    result = audit({'historical_read_mapping': {str(tmp_path): '/new'},
                    'entries': [{'id': 'a', 'benchmark': 'test', 'role': 'x', 'root': str(tmp_path)}]})
    assert any('Invalid active root' in x for x in result['errors'])


def test_pointer_hash_mismatch(tmp_path):
    release = tmp_path / 'release.json'
    release.write_text('{"resources": {}}')
    pointer = tmp_path / 'current.json'
    pointer.write_text(json.dumps({'release': str(release), 'sha256': '0' * 64}))
    with pytest.raises(ValueError, match='Release hash'):
        check_pointer(pointer)
