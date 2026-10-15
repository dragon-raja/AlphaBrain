from pathlib import Path
import pytest

from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.data.recovery_archive import backup


def test_backup_preserves_versions_symlinks_and_excludes_credentials(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'code.py').write_text('one')
    (source/'link').symlink_to('code.py');(source/'auth.json').write_text('secret')
    config=tmp_path/'config.json';target=tmp_path/'target'
    write(config,dict(host='test',sources=[str(source)],destination=str(target),exclude_patterns=['auth.json']))
    assert backup(config)['status']=='PASS'
    records=read(target/'manifest.json')['files'];assert str(source/'auth.json') not in records
    saved=Path(records[str(source/'code.py')]['backup']);assert saved.read_text()=='one'
    assert Path(records[str(source/'link')]['backup']).is_symlink()
    assert backup(config)['unchanged_files']==2
    (source/'code.py').write_text('two updated')
    assert backup(config)['status']=='PASS';assert saved.read_text()=='two updated'
    assert [p.read_text() for p in (target/'versions').rglob('code.py')]==['one']
    (source/'code.py').unlink();backup(config)
    assert saved.read_text()=='two updated'


def test_recursive_or_missing_source_rejected(tmp_path):
    config=tmp_path/'config.json'
    write(config,dict(host='x',sources=[str(tmp_path)],destination=str(tmp_path/'backup')))
    with pytest.raises(ValueError):backup(config)


def test_final_verification_detects_backup_corruption(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'data').write_text('original')
    config=tmp_path/'config.json';target=tmp_path/'target'
    cfg=dict(host='test',sources=[str(source)],destination=str(target),verify_unchanged=True)
    write(config,cfg);backup(config)
    entry=read(target/'manifest.json')['files'][str(source/'data')]
    Path(entry['backup']).write_text('corrupt')
    with pytest.raises(IOError,match='content changed'):backup(config)
