import numpy as np
import pytest

from AlphaBrain.research.dsol.data.view_confirmation import frozen_targets


def test_targets_frozen_and_shared_when_identical():
    y=np.array([[[1,1,1,1],[0,0,0,0]],[[0,0,0,0],[1,1,1,1]]])
    a=np.array([[0.,1.],[0.,1.]])
    selected,trials=frozen_targets(y,a,['pretrain','target'],['a','b'],['canonical','v1'],[4,5])
    assert len(trials)==6
    assert selected[1]['methods']['seen_fixed']=='canonical'
    assert selected[1]['methods']['old_empirical_winner']=='v1'
    assert len({r['trial_id'] for r in trials})==len(trials)
    # Changing only target labels changes its empirical winner, not the rule controls.
    y[1]=y[1,::-1]
    other,_=frozen_targets(y,a,['pretrain','target'],['a','b'],['canonical','v1'],[4,5])
    for k in ('seen_fixed','minimum_accel','canonical'):
        assert selected[1]['methods'][k]==other[1]['methods'][k]


@pytest.mark.parametrize('repeats', [[0],[3,4],[4,4],[]])
def test_overlap_or_invalid_noise_address_rejected(repeats):
    with pytest.raises(ValueError):
        frozen_targets(np.zeros((1,2,4)),np.zeros((1,2)),['pretrain'],['a'],['canonical','v1'],repeats)
