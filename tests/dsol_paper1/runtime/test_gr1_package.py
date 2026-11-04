"""No network, model imports or GPU access."""
import hashlib
import io
import pytest
from benchmarks.Robocasa_tabletop.gr1_package import package_plan, fetch_file, audit_package


def row(path, data):
    return dict(Path=path, Type='blob', Sha256=hashlib.sha256(data).hexdigest(),
                Revision='a' * 40, Size=len(data))


def test_manifest_requires_independent_agreement():
    asset = row('checkpoint/params/data', b'abc')
    manifest = asset['Sha256'] + '  ' + asset['Path'] + '\n'
    catalog = {'Data': {'Files': [asset, row('PACKAGE_MANIFEST.sha256', manifest.encode())]}}
    assert len(package_plan(catalog, manifest)) == 2
    catalog['Data']['Files'][0]['Sha256'] = '0' * 64
    with pytest.raises(ValueError):
        package_plan(catalog, manifest)


def test_verified_fetch_reuses_without_network(tmp_path):
    asset = row('checkpoint/data', b'abc')
    result = fetch_file(tmp_path, asset, bytes_per_second=10**9,
                        opener=lambda *a, **k: io.BytesIO(b'abc'))
    assert result['status'] == 'FETCHED_VERIFIED'
    def forbidden(*a, **k):
        raise AssertionError('Must reuse verified file')
    assert fetch_file(tmp_path, asset, bytes_per_second=10**9, opener=forbidden)['status'] == 'REUSED_VERIFIED'
    assert audit_package(tmp_path, [asset])['status'] == 'PASS_FILES_ONLY'


def test_bad_transfer_preserved_and_not_retried(tmp_path):
    asset = row('data', b'abc')
    with pytest.raises(ValueError):
        fetch_file(tmp_path, asset, bytes_per_second=10**9,
                   opener=lambda *a, **k: io.BytesIO(b'bad'))
    assert not (tmp_path / 'data').exists()
    assert (tmp_path / 'data.partial').read_bytes() == b'bad'
    with pytest.raises(ValueError, match='explicit audit'):
        fetch_file(tmp_path, asset, bytes_per_second=10**9)


def test_destination_traversal_and_symlink_rejected(tmp_path):
    with pytest.raises(ValueError):
        fetch_file(tmp_path, row('../bad', b'x'), bytes_per_second=10**9)
    (tmp_path / 'escape').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        fetch_file(tmp_path, row('escape/data', b'x'), bytes_per_second=10**9)
