import hashlib
import pytest
from AlphaBrain.research.dsol.data.hf_snapshot import validate_metadata, verify_file


def metadata(name='config.json'):
    blob = hashlib.sha1(b'blob 2\0{}').hexdigest()
    item = dict(repo_id='official/model', revision='a'*40, max_bytes=100)
    meta = dict(id=item['repo_id'], sha=item['revision'],
                siblings=[dict(rfilename=name, size=2, blobId=blob)])
    return item, meta


def test_pinned_identity_and_git_hash(tmp_path):
    item, meta = metadata()
    entry = validate_metadata(item, meta)[0]
    path = tmp_path / 'config.json'; path.write_bytes(b'{}')
    assert verify_file(path, entry)['sha256'] == hashlib.sha256(b'{}').hexdigest()
    path.write_bytes(b'[]')
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_file(path, entry)


def test_lfs(tmp_path):
    item, meta = metadata()
    entry = meta['siblings'][0]
    entry['lfs'] = dict(sha256=hashlib.sha256(b'{}').hexdigest())
    path = tmp_path/'weights'; path.write_bytes(b'{}')
    assert verify_file(path, entry)['upstream_hash'] == 'sha256'


@pytest.mark.parametrize('name', ['../x', '/x', 'a/../x', 'a\\x', 'IMPORT_RECEIPT.json'])
def test_unsafe_paths(name):
    item, meta = metadata(name)
    with pytest.raises(ValueError):
        validate_metadata(item, meta)


def test_revision_and_budget():
    item, meta = metadata()
    item['revision'] = 'main'
    with pytest.raises(ValueError):
        validate_metadata(item, meta)
    item, meta = metadata(); item['max_bytes'] = 1
    with pytest.raises(ValueError):
        validate_metadata(item, meta)
