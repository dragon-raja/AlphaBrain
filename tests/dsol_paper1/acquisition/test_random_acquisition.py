"""Random acquisition is independent of scores, candidates and global RNG."""
from dataclasses import replace
import numpy as np
import pytest
from AlphaBrain.research.dsol.acquisition.random_acquisition import UniformIncrementAcquisition
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from tests.dsol_paper1.helpers.local_search import synthetic_session


def rollout(invert=False):
    old,source=synthetic_session();original=source.query
    if invert:
        def query():
            m=original();return replace(m,value=replace(m.value,accel_mean=3.-m.value.accel_mean))
        source.query=query
    core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
        wrist=np.ones(4),score_scale=1.,seed=19)
    rng=UniformIncrementAcquisition(1809);obs=core.start()
    while core.status=='ACTIVE':obs,*_=core.step(rng.action(obs['delta_low'],obs['delta_high']))
    return core,source


def test_random_path_ignores_score_field_and_uses_common_minimum_return():
    a,sa=rollout();b,sb=rollout(invert=True)
    assert a.decision==b.decision==6
    np.testing.assert_array_equal([r[0] for r in a.records],[r[0] for r in b.records])
    assert a.best!=b.best
    for core,source in ((a,sa),(b,sb)):
        assert core.queries==source.calls and core.queries<=8
        assert core.records[core.best][1].accel_mean==min(v.accel_mean for _,v in core.records)
        np.testing.assert_array_equal(core.current,core.records[core.best][0])


def test_private_seed_and_legal_continuous_increments():
    a,b=UniformIncrementAcquisition(1),UniformIncrementAcquisition(1)
    for _ in range(40):
        np.random.random(17)
        x=a.action([-10,-1.25],[0,10]);y=b.action([-10,-1.25],[0,10])
        np.testing.assert_array_equal(x['delta_deg'],y['delta_deg'])
        assert not x['stop'] and -10<=x['delta_deg'][0]<=0 and -1.25<=x['delta_deg'][1]<=10
    with pytest.raises(ValueError):a.action([0,0],[-1,1])
