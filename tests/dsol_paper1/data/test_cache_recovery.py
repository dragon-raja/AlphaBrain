import pytest

from AlphaBrain.research.dsol.data.artifacts import write, read, sha
from AlphaBrain.research.dsol.data.cache_recovery import reuse_initials


def test_reuse_requires_same_science_and_preserves_source(tmp_path):
    old, new = tmp_path / 'old.json', tmp_path / 'new.json'
    plan = dict(implementation_sha256={'renderer.py': 'same'},
                jobs=[dict(initial_id='a', snapshot_manifest_sha256='snapshot')],
                catalog=dict(views=[dict(view_id='canonical')]))
    write(old, plan); write(new, plan)
    source = tmp_path / 'source/initials/a/summary.json'
    write(source, dict(initial_id='a', snapshot_manifest_sha256='snapshot',
                      plan_sha256=sha(old), status='PASS', views=[{}], files={}))
    digest = sha(source)
    reuse_initials(old, new, tmp_path / 'source', tmp_path / 'dest')
    assert sha(source) == digest
    assert read(tmp_path / 'dest/initials/a/summary.json')['reuse_source']['sha256'] == digest
    plan['implementation_sha256']['renderer.py'] = 'changed'
    write(new, plan)
    with pytest.raises(ValueError, match='renderer'):
        reuse_initials(old, new, tmp_path / 'source', tmp_path / 'other')
