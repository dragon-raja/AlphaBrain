import numpy as np
import pytest

from benchmarks.Robocasa365.short_view_probe import prefix_means,verify_anchor,record_step_event
from benchmarks.Robocasa365.short_view_probe import score_fresh_anchor


def test_fresh_xr_anchor_uses_eight_initial_only_addressed_calls(tmp_path):
    class Policy:
        def __init__(self):self.calls=[]
        def infer(self,history,prompt,seed,capture):
            self.calls.append((len(history),seed,prompt,capture))
            rng=np.random.default_rng(seed)
            return {'velocity_trace':rng.normal(size=(1,5,16,12)), 'actions':rng.normal(size=(16,12))}
    policy=Policy()
    obs={'annotation.human.task_description':'task','state.test':np.zeros(3)}
    images=[np.full((2,2,3),i,np.uint8) for i in range(3)]
    request,meta,velocity,actions,scores=score_fresh_anchor(policy,'xr1','fresh',obs,
        {'view':{'view_id':'canonical'},'images':images},tmp_path)
    assert len(policy.calls)==8 and all(c[0]==1 for c in policy.calls)
    assert len({c[1] for c in policy.calls})==8
    assert scores.shape==(8,) and velocity.shape==(8,5,16,12) and actions.shape==(8,16,12)
    assert meta['prompt']=='task' and request['state.test'].shape==(3,)
    assert (tmp_path/'initial-anchor.json').is_file()


def test_prefix_is_same_noise_prefix_not_resampling():
    x=np.arange(40,dtype=float).reshape(8,5)
    got=prefix_means(x)
    for k in range(1,6): np.testing.assert_allclose(got[k-1],x[:,:k].mean())
    assert got[0]==x[:,0].mean()


def test_no_missing_prefix_imputation():
    x=np.ones((8,5));x[3,2]=np.nan
    with pytest.raises(ValueError):prefix_means(x)


def test_anchor_is_strict():
    x=np.ones((3,4));verify_anchor(x,x.copy())
    other=x.copy();other[0,0]+=1e-12
    with pytest.raises(ValueError):verify_anchor(x,other)
    with pytest.raises(ValueError):verify_anchor(x,np.ones((4,3)))


def test_historical_contiguous_mean_is_bitwise_equal():
    rng=np.random.default_rng(8)
    for _ in range(100):
        x=rng.random((8,5))
        assert prefix_means(x)[0]==x[:,0].copy().mean()


def test_success_at_step99_does_not_prevent_step100_score():
    record={}
    assert not record_step_event(record,99,True,False,False)
    assert not record_step_event(record,100,True,False,False)
    assert record=={'success_steps':[99,100]}


@pytest.mark.parametrize('terminated,truncated',[(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('success',[False,True])
def test_actual_environment_end_still_fails_closed(terminated,truncated,success):
    record={}
    assert record_step_event(record,99,success,terminated,truncated)
    assert record['early_terminal']['step']==99
