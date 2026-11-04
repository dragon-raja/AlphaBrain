"""New modules must be packaged, and parent workspace imports cannot mask gaps."""
import sys
import pytest
from AlphaBrain.research.dsol.execution.source_release import python_sources,isolated_identity
from AlphaBrain.research.dsol.execution.source_release import materialize_verified_release
import hashlib


def mirror_source(tmp_path):
    root=tmp_path/'source';root.mkdir();p=root/'code.py';p.write_text('value=3\n')
    return dict(release=str(root),source_identity='frozen',archive_sha256='archive',
                artifacts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())])


def test_mirror_preserves_bytes_identity_and_refuses_replacement(tmp_path):
    source=mirror_source(tmp_path);dest=tmp_path/'local';result=materialize_verified_release(source,dest)
    assert result['source_identity']==source['source_identity']
    assert result['archive_sha256']==source['archive_sha256']
    assert result['artifacts'][0]['sha256']==source['artifacts'][0]['sha256']
    assert (dest/'code.py').read_bytes()==(tmp_path/'source/code.py').read_bytes()
    assert result['release']==str(dest) and source['release']!=str(dest)
    with pytest.raises(ValueError,match='replaced'):materialize_verified_release(source,dest)


@pytest.mark.parametrize('fault',['hash','duplicate','traversal','symlink','empty'])
def test_invalid_mirror_never_published(tmp_path,fault):
    s=mirror_source(tmp_path);entry=s['artifacts'][0]
    if fault=='hash':entry['sha256']='bad'
    elif fault=='duplicate':s['artifacts'].append(dict(entry))
    elif fault=='traversal':entry['path']=str(tmp_path/'source/../code.py')
    elif fault=='symlink':
        p=tmp_path/'source/code.py';p.unlink();p.symlink_to(tmp_path/'outside')
    elif fault=='empty':s['artifacts']=[]
    with pytest.raises(ValueError):materialize_verified_release(s,tmp_path/'local')
    assert not (tmp_path/'local').exists()


def test_publish_collision_preserves_other_owner_and_cleans_temporary(tmp_path):
    s=mirror_source(tmp_path);lock=tmp_path/'local.publish-lock';lock.write_text('other')
    with pytest.raises(FileExistsError):materialize_verified_release(s,tmp_path/'local')
    assert lock.read_text()=='other' and not (tmp_path/'local').exists()
    assert not list(tmp_path.glob('.release-*'))


def test_new_module_collected_and_isolated_import_rejects_missing_dependency(tmp_path):
    root=tmp_path/'work';package=root/'package';package.mkdir(parents=True)
    (package/'__init__.py').write_text('from .new import value\ndef source_identity(): return value\n')
    assert python_sources(root,['package'])==[package.relative_to(root)/'__init__.py']
    with pytest.raises(ValueError,match='Isolated release import failed'):
        isolated_identity(root,sys.executable,'package')
    (package/'new.py').write_text('value = "new-source-identity"\n')
    assert len(python_sources(root,['package']))==2
    assert isolated_identity(root,sys.executable,'package')=='new-source-identity'


def test_source_collection_rejects_outside_files(tmp_path):
    root=tmp_path/'work';root.mkdir();outside=tmp_path/'outside.py';outside.write_text('x=1\n')
    (root/'escape.py').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):python_sources(root,['.'])
    with pytest.raises(ValueError,match='owned'):python_sources(root,['..'])


def test_identity_covers_added_nested_source_and_is_location_independent(tmp_path):
    from AlphaBrain.research.dsol.execution.source_release import source_manifest,source_tree_identity
    import shutil
    a=tmp_path/'a';p=a/'code';p.mkdir(parents=True)
    (p/'one.py').write_text('x=1\n');before=source_tree_identity(a,['code'])
    (p/'nested').mkdir();(p/'nested'/'two.py').write_text('y=2\n')
    after=source_tree_identity(a,['code'])
    assert before!=after and len(source_manifest(a,['code']))==2
    b=tmp_path/'b';shutil.copytree(a,b)
    assert source_tree_identity(b,['code'])==after
