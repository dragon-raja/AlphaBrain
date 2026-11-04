"""Actual PPO CPU entrypoint with eight synthetic complete searches per update."""
from contextlib import contextmanager
import json
import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.selectors.continuous_ppo import (
    ContinuousPPO,ContinuousPPOConfig,initialize_adaptation_actor)
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
from tests.dsol_paper1.helpers.local_search import synthetic_session
from tests.dsol_paper1.helpers.adaptation_reward import config


@contextmanager
def backend(row,output):
    old,source=synthetic_session()
    core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
        wrist=np.ones(4),score_scale=1.,reward_scale=1.,seed=row['episode'])
    yield ContinuousSearchGym(4),{'session':core}


def make(path,*,resume=False):
    torch.set_num_threads(1);torch.manual_seed(79)
    return ContinuousPPO(ContinuousActorCritic(4,width=16),
        ContinuousPPOConfig(episodes_per_update=8,updates=5),
        [dict(episode=i) for i in range(40)],backend,path,
        binding=dict(source_kind='SYNTHETIC_CPU_NO_MODEL'),resume=resume,reward_config=config())


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('pause',[3,8,35])
def test_actual_entrypoint_and_exact_resume_after_window_eviction(tmp_path,pause):
    with make(tmp_path/'full') as t:
        t.run(); expected=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
        assert expected['reward_state']['count']==40
        assert len(expected['reward_state']['initials'])==32
    with make(tmp_path/'partial') as t:
        for _ in range(pause):
            if len(t.pending)==8: t.optimize()
            t.collect_one()
    with make(tmp_path/'partial',resume=True) as t:
        t.run(); actual=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
        equal(expected,actual)


def test_failed_optimizer_checkpoint_does_not_silently_replay(tmp_path,monkeypatch):
    with make(tmp_path/'failed') as t:
        for _ in range(8): t.collect_one()
        def fail(*args,**kwargs): raise OSError('injected checkpoint failure')
        monkeypatch.setattr(t.store,'save_checkpoint',fail)
        with pytest.raises(OSError): t.optimize()
    with pytest.raises(ValueError,match='no automatic replay'): make(tmp_path/'failed',resume=True)


def test_old_full_optimizer_warm_start_and_wrong_batch_rejected(tmp_path):
    for n,warm in [(32,None),(8,{})]:
        with pytest.raises(ValueError,match='explicit actor initialization'):
            ContinuousPPO(ContinuousActorCritic(4,width=16),
                ContinuousPPOConfig(episodes_per_update=n,updates=1),
                [dict(episode=i) for i in range(n)],backend,tmp_path/str(n),
                binding={},reward_config=config(),warm_start=warm)


def test_actor_transfer_keeps_fresh_critic_and_rejects_mismatch_before_mutation():
    torch.manual_seed(1); source=ContinuousActorCritic(4,width=16)
    torch.manual_seed(2); target=ContinuousActorCritic(4,width=16)
    before={k:v.clone() for k,v in target.state_dict().items()}
    receipt=initialize_adaptation_actor(target,source.state_dict())
    assert not receipt['optimizer_copied'] and not receipt['critic_copied']
    for k,v in target.state_dict().items():
        assert torch.equal(v,source.state_dict()[k] if k in receipt['copied_actor_keys'] else before[k])
    before={k:v.clone() for k,v in target.state_dict().items()}
    with pytest.raises(ValueError,match='Incompatible'):
        initialize_adaptation_actor(target,ContinuousActorCritic(6,width=16).state_dict())
    equal(before,target.state_dict())
