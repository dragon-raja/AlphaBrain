"""Full synthetic rollout, G settlement, optimizer and recovery for mechanisms."""
from contextlib import contextmanager
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.selectors.mechanism_policy import MechanismActorCritic
from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec,reward_for
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
from tests.dsol_paper1.helpers.local_search import synthetic_session

@contextmanager
def backend(row,output):
    old,source=synthetic_session()
    core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',wrist=np.ones(4),score_scale=1.,seed=row['episode'])
    yield ContinuousSearchGym(4),{'session':core}


def trainer(root,name,resume=False):
    torch.set_num_threads(1);torch.manual_seed(7);spec=mechanism_spec(name)
    p=MechanismActorCritic(core_spec('U12',width=16),mechanism=spec,visual_dim=4)
    return ContinuousPPO(p,ContinuousPPOConfig(updates=2),[dict(episode=i) for i in range(64)],backend,root,
        binding=dict(source_kind='SYNTHETIC_CPU_NO_MODEL'),reward_config=reward_for(spec),resume=resume)


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

@pytest.mark.parametrize('name',['cost005','cost020','no_score','fixed_query'])
def test_full_ppo_exact_resume(name,tmp_path):
    with trainer(tmp_path/'whole',name) as t:
        t.run();expected=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
    with trainer(tmp_path/'split',name) as t:
        for _ in range(32):t.collect_one()
        t.optimize()
    with trainer(tmp_path/'split',name,True) as t:
        t.run();actual=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
        assert actual['update']==2 and actual['reward_state']['batches']==2
    equal(expected,actual)
