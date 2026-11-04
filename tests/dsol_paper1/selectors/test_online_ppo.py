"""Exact resumed optimizer/statistics and failure tests on synthetic CPU data."""
from contextlib import contextmanager
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
from AlphaBrain.research.dsol.data.reward_pilot import reward_config
from tests.dsol_paper1.helpers.local_search import synthetic_session


@contextmanager
def backend(row, output):
    old,source=synthetic_session()
    core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
        wrist=np.ones(4),score_scale=1.,reward_scale=.01214765438717935,seed=row['episode'])
    yield ContinuousSearchGym(4),{'session':core}


def make(path,kind,*,resume=False,warm_start=None):
    torch.set_num_threads(1);torch.manual_seed(19)
    return ContinuousPPO(ContinuousActorCritic(4,width=16),ContinuousPPOConfig(updates=2),
        [dict(episode=i) for i in range(64)],backend,path,
        binding=dict(source_kind='SYNTHETIC_CPU_NO_MODEL'),resume=resume,
        reward_config=reward_config(kind),warm_start=warm_start)


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.mark.parametrize('kind',['B','G','L'])
@pytest.mark.parametrize('pause',[17,32,39])
def test_exact_online_resume_without_double_statistics(tmp_path,kind,pause):
    with make(tmp_path/'full',kind) as t:
        t.run();expected=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
        assert expected['reward_state']['batches']==2
        assert len(expected['reward_state']['initials'])==64
    with make(tmp_path/'partial',kind) as t:
        for _ in range(pause):
            if len(t.pending)==32:t.optimize()
            t.collect_one()
    with make(tmp_path/'partial',kind,resume=True) as t:
        t.run();actual=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
        same(expected,actual)


def test_all_arms_fork_same_full_optimizer_and_rng(tmp_path):
    with make(tmp_path/'source','B') as t:
        t.run(through_update=1)
        source=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
    for arm in ('B','G','L'):
        with make(tmp_path/arm,arm,warm_start=source) as t:
            actual=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
            for k in ('policy','optimizer','rng'):same(actual[k],source[k])
            assert actual['update']==actual['cursor']==actual['queries']==0
            assert actual['reward_state']['initials']==[]


def test_failed_update_cannot_silently_recompute_scale(tmp_path,monkeypatch):
    with make(tmp_path/'failed','L') as t:
        for _ in range(32):t.collect_one()
        def fail(*args,**kwargs):raise OSError('injected optimizer checkpoint failure')
        monkeypatch.setattr(t.store,'save_checkpoint',fail)
        with pytest.raises(OSError):t.optimize()
    with pytest.raises(ValueError,match='no automatic replay'):make(tmp_path/'failed','L',resume=True)
