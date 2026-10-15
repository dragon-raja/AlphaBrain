import json
from pathlib import Path
import zipfile
import pytest
from AlphaBrain.research.dsol.data import resource_import as m


def test_copy_and_no_overwrite(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'data.bin').write_bytes(b'original')
    item = {'id': 'a', 'kind': 'copy_tree', 'source': str(source), 'destination': 'raw/a'}
    result = m.acquire(item, tmp_path, lambda *args: None, 100)
    receipt = json.loads((Path(result['destination']) / 'IMPORT_RECEIPT.json').read_text())
    assert receipt['files'][0]['sha256'] == m.digest(source / 'data.bin')
    assert not receipt['runtime_validated']
    with pytest.raises(FileExistsError):
        m.acquire(item, tmp_path, lambda *args: None, 100)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../../escape', ''])
def test_target_rejects_escape(tmp_path, name):
    with pytest.raises(ValueError):
        m.target(tmp_path, name)


def test_symlinks_rejected(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'link').symlink_to('/tmp')
    with pytest.raises(ValueError):
        m.inventory(source)
    with pytest.raises(ValueError):
        m.target(tmp_path, 'source/link/file')


def test_zip_validation_before_extraction(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape', b'bad')
    with pytest.raises(ValueError):
        m.extract_zip(archive, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_zip_hashes(tmp_path):
    archive = tmp_path / 'ok.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('task/seed.txt', '41\n42\n')
    rows = m.extract_zip(archive, tmp_path / 'out')
    assert rows == [{'path': 'task/seed.txt', 'bytes': 6,
                     'sha256': m.digest(tmp_path / 'out/task/seed.txt')}]


def test_tar_rejects_links_before_writing(tmp_path):
    import tarfile
    path = tmp_path / 'bad.tar.gz'
    with tarfile.open(path, 'w:gz') as f:
        entry = tarfile.TarInfo('link'); entry.type = tarfile.SYMTYPE; entry.linkname = '/tmp'
        f.addfile(entry)
    with pytest.raises(ValueError, match='Unsafe TAR'):
        m.extract_tar(path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_tar_hashes(tmp_path):
    import io
    import tarfile
    path = tmp_path / 'ok.tar.gz'
    with tarfile.open(path, 'w:gz') as f:
        entry = tarfile.TarInfo('checkpoint/metadata.json'); entry.size = 2
        f.addfile(entry, io.BytesIO(b'{}'))
    rows = m.extract_tar(path, tmp_path / 'out')
    assert rows[0]['sha256'] == m.digest(tmp_path / 'out/checkpoint/metadata.json')


def test_failure_receipt(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'resources': [{'id': 'missing', 'kind': 'copy_tree',
                     'source': str(tmp_path / 'absent'), 'destination': 'data'}]}))
    record = tmp_path / 'record'
    with pytest.raises(ValueError):
        m.run(plan, tmp_path, record)
    assert json.loads((record / 'status.json').read_text())['status'] == 'FAILED'


def test_overlap_rejected(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'resources': [{'id': 'a', 'destination': 'data'},
                                             {'id': 'b', 'destination': 'data/nested'}]}))
    with pytest.raises(ValueError):
        m.run(plan, tmp_path, tmp_path / 'record')
    assert not (tmp_path / 'record').exists()


def test_explicit_exclusion_of_source_link(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'code.py').write_text('pass\n')
    (source / 'assets').symlink_to('/tmp')
    assert list(m.inventory(source, ['assets'])) == ['code.py']


def test_official_hash_failure_never_publishes(tmp_path, monkeypatch):
    def fake_download(args, **kwargs):
        Path(args[args.index('--output') + 1]).write_bytes(b'not the official ZIP')
    monkeypatch.setattr(m.subprocess, 'run', fake_download)
    item = {'id': 'bad', 'kind': 'official_zip', 'destination': 'raw',
            'url': 'https://example.invalid/pinned.zip', 'bytes': 20, 'sha256': '0' * 64}
    with pytest.raises(ValueError, match='SHA256'):
        m.acquire(item, tmp_path, lambda *args: None, 100)
    assert not (tmp_path / 'raw').exists()
    assert (tmp_path / 'raw.partial/original.zip').exists()


def test_cli_is_dry_by_default(tmp_path, monkeypatch, capsys):
    from scripts.dsol_paper1.operations.import_resources import main
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'resources': []}))
    root, record = tmp_path / 'data', tmp_path / 'run'
    monkeypatch.setattr('sys.argv', ['import_resources', '--plan', str(plan),
                                    '--root', str(root), '--record', str(record)])
    main()
    assert json.loads(capsys.readouterr().out) == {'resources': []}
    assert not root.exists() and not record.exists()
