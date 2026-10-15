import pytest
from AlphaBrain.research.dsol.data.xr1_rebalance import validate_order


def test_partial_candidate_keeps_noise_addresses():
    order=dict(initial='i',views=['a','b'],repeats={'a':[2,3],'b':[0,1,2,3]},panel_sha256='frozen')
    validate_order(order,['i'],['a','b','c'],'frozen')
    assert order['repeats']['a']==[2,3]


@pytest.mark.parametrize('repeats',[[],[0,0],[3,2],[4],[-1]])
def test_invalid_repeats_rejected(repeats):
    order=dict(initial='i',views=['a'],repeats={'a':repeats},panel_sha256='frozen')
    with pytest.raises(ValueError):validate_order(order,['i'],['a'],'frozen')


def test_unknown_initial_rejected():
    with pytest.raises(ValueError):
        validate_order(dict(initial='other',views=['a'],repeats={'a':[0]},panel_sha256='frozen'),['i'],['a'],'frozen')
