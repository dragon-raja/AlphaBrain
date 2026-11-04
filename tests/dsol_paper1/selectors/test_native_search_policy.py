import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.selectors.native_search_policy import (
    NativeStream,NativeContinuousActorCritic,pool_native_grid,capture_fastwam_query)
from AlphaBrain.research.dsol.data.native_observation import CameraFrame,ObservationBundle,FeatureSpec


def observation(streams):
    x=dict(geometry=torch.zeros(2,3,12),scores=torch.zeros(2,3,2),flags=torch.zeros(2,3,4),
        global_=torch.zeros(2,6),action_mask=torch.ones(2,4,dtype=torch.bool),joint_state=torch.zeros(2,6),
        delta_low=torch.full((2,2),-5.,dtype=torch.float64),delta_high=torch.full((2,2),5.,dtype=torch.float64),
        feature_mask=torch.zeros(2,3,len(streams),dtype=torch.bool))
    x['global']=x.pop('global_');x['flags'][:,:2,0]=1;x['feature_mask'][:,:2]=True
    for i,s in enumerate(streams):x[f'feature_{i}']=torch.randn(2,3,4*s.channels,requires_grad=True)
    return x


@pytest.mark.parametrize('streams',[
    (NativeStream('head',5,'per_view',('head',),('movable',)),),
    (NativeStream('joint',7,'joint',('external','wrist'),('movable','context')),),
    (NativeStream('head',5,'per_view',('head',),('movable',)),
     NativeStream('left',3,'per_view',('left',),('context',)),
     NativeStream('right',9,'per_view',('right',),('context',))),
])
def test_real_cpu_objective_updates_search_only(streams):
    torch.set_num_threads(1);torch.manual_seed(9)
    p=NativeContinuousActorCritic(streams,width=16,heads=4,layers=1);obs=observation(streams)
    action=p.distribution(obs).sample()
    loss=p.objective(obs,action['stop'],action['latent'],action['log_prob'].detach(),
                     torch.tensor([1.,-1.]),torch.tensor([1.,0.]))
    loss.backward()
    assert all(obs[f'feature_{i}'].grad is None for i in range(len(streams)))
    assert p.actor_history.projections[0][0].weight.grad is not None
    before=p.critic_history.projections[0][0].weight.detach().clone()
    torch.optim.Adam(p.parameters(),lr=.001).step()
    assert not torch.equal(before,p.critic_history.projections[0][0].weight)


def test_absent_context_and_unvisited_features_cannot_change_policy():
    streams=(NativeStream('head',3,'per_view',('head',),('movable',)),
             NativeStream('wrist',4,'per_view',('wrist',),('context',)))
    p=NativeContinuousActorCritic(streams,width=16,layers=1).eval();x=observation(streams)
    x['feature_mask'][...,1]=False
    original=p.value(x).detach()
    with torch.no_grad():x['feature_1'].fill_(999);x['feature_0'][:,2].fill_(777)
    assert torch.equal(original,p.value(x))
    x['feature_mask'][0,0,0]=False
    with pytest.raises(ValueError,match='movable'):p.value(x)


def test_joint_grid_keeps_spatial_cells_and_rejects_wrong_query():
    observation_=ObservationBundle('q','clock',1.,2.,1.,0.,'a'*64,'b'*64,
        (CameraFrame('head','movable',True,2.,'c'*64),CameraFrame('wrist','context',True,2.,'d'*64)))
    spec=FeatureSpec('target_native','e'*64,'video.prepare','joint',('head','wrist'),(1,8,3),'float32','a'*64)
    tokens=torch.arange(24,dtype=torch.float32).reshape(1,8,3).requires_grad_()
    value=pool_native_grid(tokens,spec,observation_,query_id='q',history_sha256='b'*64,grid=(1,2,4))
    assert value.shape==(1,12) and not value.requires_grad
    assert torch.equal(value.reshape(4,3),tokens.detach().reshape(2,4,3).reshape(2,2,2,3).mean(2).reshape(4,3))
    with pytest.raises(ValueError):pool_native_grid(tokens,spec,observation_,query_id='stale',history_sha256='b'*64,grid=(1,2,4))
    with pytest.raises(ValueError):pool_native_grid(tokens,spec,observation_,query_id='q',history_sha256='b'*64,grid=(2,2,2))


def test_no_fake_camera_or_unreviewed_legacy_weight_load():
    stream=NativeStream('joint',3,'joint',('head','wrist'),('movable','context'))
    with pytest.raises(ValueError,match='twice'):NativeContinuousActorCritic([stream,NativeStream('wrist',3,'per_view',('wrist',),('context',))])
    p=NativeContinuousActorCritic([stream],width=16,layers=1)
    with pytest.raises(ValueError,match='mapping'):p.load_legacy_encoder({})


def test_capture_pool_policy_actual_cpu_path_one_forward_output_and_rng_unchanged():
    class Expert:
        calls=0
        def prepare(self,**kw):
            self.calls+=1
            return (torch.rand(1,8,3),)+(None,)*9
    e=Expert()
    def infer():
        return e.prepare(x=torch.zeros(1,2,1,2,4),timestep=torch.zeros(1),action=None)
    ob=ObservationBundle('q','clock',1.,2.,1.,0.,'a'*64,'b'*64,
        (CameraFrame('head','movable',True,2.,'c'*64),))
    spec=FeatureSpec('target_native','e'*64,'video.prepare','joint',('head',),(1,8,3),'float32','a'*64)
    torch.manual_seed(43);expected=infer();rng=torch.get_rng_state()
    torch.manual_seed(43);result,pooled=capture_fastwam_query(e,infer,spec,ob,grid=(1,2,4))
    assert e.calls==2 and torch.equal(result[0],expected[0]) and torch.equal(rng,torch.get_rng_state())
    p=NativeContinuousActorCritic([NativeStream('joint',3,'joint',('head',),('movable',))],width=16,layers=1)
    x=observation(p.actor_history.streams);x['feature_0']=pooled[:,None,:].expand(2,3,12)
    assert p.distribution(x).sample(deterministic=True)['delta_deg'].shape==(2,2)
    with pytest.raises(ValueError):capture_fastwam_query(e,infer,spec,ob,grid=(2,2,2))
    assert e.calls==2
