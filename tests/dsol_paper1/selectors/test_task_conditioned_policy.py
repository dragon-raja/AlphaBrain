from copy import deepcopy
import pytest

torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.data.search_core import core_spec,VARIANTS
from AlphaBrain.research.dsol.data.native_observation import NativeStream
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.selectors.unified_search_policy import UnifiedActorCritic,SharedHistoryCore
from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
from AlphaBrain.research.dsol.data.native_features import layout_spec
from tests.dsol_paper1.helpers.task_condition import task_spec


def observation():
    x=dict(geometry=torch.zeros(2,3,12),visual=torch.randn(2,3,8),wrist=torch.randn(2,8),scores=torch.randn(2,3,2),flags=torch.zeros(2,3,4),joint_state=torch.randn(2,6),delta_low=torch.full((2,2),-5.),delta_high=torch.full((2,2),5.),action_mask=torch.ones(2,4,dtype=torch.bool))
    x['global']=torch.randn(2,6);x['flags'][:,:2,0]=1;x['flags'][:,1,2]=1
    return x


@pytest.mark.parametrize('variant',list(VARIANTS))
def test_zero_task_branch_preserves_base_and_trains_both_branches(variant):
    torch.set_num_threads(1);torch.manual_seed(19)
    a=UnifiedActorCritic(core_spec(variant,width=16),visual_dim=8)
    torch.manual_seed(19)
    b=UnifiedActorCritic(core_spec(variant,width=16),visual_dim=8,task_spec=task_spec())
    x=observation();tx=dict(x,task_features=torch.randn(2,8,requires_grad=True))
    for key,p in a.named_parameters():assert torch.equal(p,dict(b.named_parameters())[key])
    for branch in ['actor_history','critic_history']:
        aa=getattr(a,branch)(x);bb=getattr(b,branch)(tx)
        assert torch.equal(aa,bb)
        bb.square().sum().backward()
        assert getattr(b,branch).core.task_projection.weight.grad.abs().sum()>0
    assert tx['task_features'].grad is None
    assert not {p.data_ptr() for p in b.actor_history.parameters()} & {p.data_ptr() for p in b.critic_history.parameters()}
    with pytest.raises(ValueError):b.value(x)
    with pytest.raises(ValueError):a.value(tx)
    before={k:v.clone() for k,v in b.named_parameters()}
    with pytest.raises(ValueError):b.load_state_dict(a.state_dict())
    for k,v in b.named_parameters():assert torch.equal(v,before[k])


@pytest.mark.parametrize('family,channels',[('pi05',8),('groot_n15',12),('rldx',16)])
@pytest.mark.parametrize('wrist',[False,True])
def test_model_adapters_and_head_only_share_task_core(family,channels,wrist):
    torch.set_num_threads(1)
    streams=(NativeStream('head',2,'per_view',('head',),('movable',)),)
    if wrist:streams+=(NativeStream('wrist',2,'per_view',('wrist',),('context',)),)
    spec=dict(schema='UNIFIED_NATIVE_CONTINUOUS_POLICY_V1',core=core_spec('U123',width=16),layout=layout_spec(streams),projection_groups=['vision']*len(streams),producer_sha256='d'*64,task_spec=task_spec(family,channels))
    p=create_search_policy(spec);x=observation();raw=x.pop('visual');w=x.pop('wrist')
    x['feature_0']=raw;x['feature_mask']=torch.zeros(2,3,len(streams),dtype=torch.bool);x['feature_mask'][:,:2]=True
    if wrist:x['feature_1']=w[:,None].expand(-1,3,-1)
    x['task_features']=torch.randn(2,channels)
    assert p.value(x).shape==(2,)
    for branch in (p.actor_history,p.critic_history):assert type(branch.core) is SharedHistoryCore
    changed=deepcopy(spec);changed['task_spec']['model_sha256']='f'*64
    q=create_search_policy(changed)
    with pytest.raises(ValueError):q.load_state_dict(p.state_dict())


def test_legacy_initial_mapping_zero_task_and_uniform_fusion():
    old=ContinuousActorCritic(8,width=16)
    p=UnifiedActorCritic(core_spec('U123',width=16),visual_dim=8,task_spec=task_spec())
    p.load_legacy_initial(old.state_dict())
    for branch in (p.actor_history,p.critic_history):
        assert torch.count_nonzero(branch.core.task_projection.weight)==0
        assert torch.count_nonzero(branch.core.fusion.weight)==0


def test_complete_task_conditioned_ppo_and_mid_batch_recovery(tmp_path):
    pytest.importorskip('gymnasium')
    from contextlib import contextmanager
    import json,numpy as np
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
    from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
    from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
    from AlphaBrain.research.dsol.acquisition.task_condition import TaskConditionedEnv
    from AlphaBrain.research.dsol.data.reward_pilot import reward_config
    from tests.dsol_paper1.helpers.local_search import synthetic_session
    @contextmanager
    def backend(row,output):
        old,source=synthetic_session()
        core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',wrist=np.ones(4),score_scale=1.,seed=row['episode'])
        env=TaskConditionedEnv(ContinuousSearchGym(4),np.full(8,1.+row['episode']%2,dtype=np.float32))
        yield env,{'session':core}
    def trainer(path,resume=False):
        torch.manual_seed(7);torch.set_num_threads(1)
        p=UnifiedActorCritic(core_spec('U123',width=16),visual_dim=4,task_spec=task_spec())
        return ContinuousPPO(p,ContinuousPPOConfig(updates=2),[dict(episode=i) for i in range(64)],backend,path,binding=dict(source_kind='SYNTHETIC_CPU'),reward_config=reward_config('G'),resume=resume)
    with trainer(tmp_path/'whole') as t:
        t.run();expected=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
    with trainer(tmp_path/'split') as t:
        for _ in range(17):t.collect_one()
    with trainer(tmp_path/'split',True) as t:
        t.run();actual=t.store.load(json.loads((t.store.root/'latest.json').read_text()))
    def same(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    same(expected,actual)
    assert actual['update']==2 and actual['cursor']==64
