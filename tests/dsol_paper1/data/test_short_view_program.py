import pytest

from AlphaBrain.research.dsol.data.short_view_program import partition_missing


def test_missing_only_once_in_original_order():
    reused={('i1','v0'),('i2','v1')}
    chunks=partition_missing(['i1','i2'],['v0','v1','v2','v3'],reused,2)
    cells=[(c['initial'],v) for c in chunks for v in c['views']]
    assert cells==[('i1','v1'),('i1','v2'),('i1','v3'),('i2','v0'),('i2','v2'),('i2','v3')]
    assert len(set(cells))==len(cells)
    assert not set(cells)&reused


def test_outside_reuse_rejected():
    with pytest.raises(ValueError):partition_missing(['i'],['v'],{('other','v')})


def test_no_pending_cells():
    assert partition_missing(['i'],['v'],{('i','v')})==[]
