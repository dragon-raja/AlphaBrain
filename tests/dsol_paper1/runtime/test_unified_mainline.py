"""Real policy factory and immutable development checkpoint selection, CPU only."""
from copy import deepcopy
import json
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.unified_mainline import select_checkpoint, VERSIONS
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.execution.manifest import digest


def panel():
    return [dict(version=v,seed=s,suite='suite',task=t,state_key=f'{t}-{i}',start=a,
        repeat=n,split='validation',success=(v==64 or n==0),queries=5)
        for v in VERSIONS for s in (0,1,2) for t in range(8) for i in range(2)
        for a in range(2) for n in range(2)]


def test_common_checkpoint_uses_complete_balanced_development_only():
    rows=panel();assert select_checkpoint(rows)['version']==64
    for r in rows:r['success']=True
    assert select_checkpoint(rows)['version']==16
    for r in rows:
        if r['version']==48:r['queries']=4
    assert select_checkpoint(rows)['version']==48
    for invalid in [rows[:-1],rows+[rows[0]],[dict(r,split='unseen_test') for r in rows]]:
        with pytest.raises(ValueError):select_checkpoint(invalid)
    changed=deepcopy(rows);changed[0]['state_key']='changed'
    with pytest.raises(ValueError):select_checkpoint(changed)


@pytest.mark.parametrize('channels',[2048,4096])
def test_actual_factory_native_actor_critic_share_same_core_and_receive_gradients(tmp_path,channels):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from benchmarks.LIBERO.prefix_pilot import policy_for
    from benchmarks.LIBERO.rldx_search_runtime import visual_feature_spec
    from AlphaBrain.research.dsol.selectors.unified_search_policy import SharedHistoryCore
    producer=visual_feature_spec()
    for s in producer['layout']['streams']:s['channels']=channels
    core=core_spec('U12');spec=dict(schema='UNIFIED_NATIVE_CONTINUOUS_POLICY_V1',core=core,
        layout=producer['layout'],projection_groups=['visual','visual'],producer_sha256=digest(producer))
    path=tmp_path/'protocol.json';path.write_text('{}')
    cell=dict(core_spec=core,native_policy=spec,model_hashes={'s':'a'*64})
    p=policy_for(dict(protocol=str(path)),cell)
    assert p.native_spec==spec
    assert type(p.actor_history.core) is type(p.critic_history.core) is SharedHistoryCore
    assert not {x.data_ptr() for x in p.actor_history.parameters()} & {x.data_ptr() for x in p.critic_history.parameters()}
    obs=dict(geometry=torch.randn(2,3,12),scores=torch.randn(2,3,2),flags=torch.zeros(2,3,4),
        joint_state=torch.randn(2,6),delta_low=torch.full((2,2),-5.),delta_high=torch.full((2,2),5.),
        action_mask=torch.ones(2,4,dtype=torch.bool),feature_mask=torch.ones(2,3,2,dtype=torch.bool),
        feature_0=torch.randn(2,3,4*channels),feature_1=torch.randn(2,3,4*channels))
    obs['global']=torch.randn(2,6);obs['flags'][:,:,0]=1;obs['flags'][:,2,2]=1
    torch.nn.init.normal_(p.stop_head.weight);torch.nn.init.normal_(p.mean_head.weight)
    action=p.distribution(obs).sample();loss=p.objective(obs,action['stop'],action['latent'],action['log_prob'].detach(),torch.tensor([1.,-1.]),torch.tensor([.1,.9]))
    loss.backward()
    for branch in (p.actor_history,p.critic_history):
        assert branch.adapter.projections['visual'][0].weight.grad.abs().sum()>0
        assert branch.core.joints[0].weight.grad.abs().sum()>0
    foreign=deepcopy(p.state_dict());foreign['_extra_state']['producer_sha256']='b'*64
    with pytest.raises(ValueError):p.load_state_dict(foreign)
    cell['core_spec']=core_spec('U3')
    with pytest.raises(ValueError):policy_for(dict(protocol=str(path)),cell)


def test_real_camera_visual_spec_has_distinct_spatial_streams():
    from benchmarks.LIBERO.rldx_search_runtime import visual_feature_spec,feature_spec
    from AlphaBrain.research.dsol.data.native_features import read_layout,NativeFeatures,to_wire,from_wire
    spec=visual_feature_spec();streams=read_layout(spec['layout'])
    assert [s.camera_ids for s in streams]==[('external',),('wrist',)]
    packet=NativeFeatures(spec['layout'],'q','a'*64,'b'*64,'c'*64,
        (np.ones(16384,np.float32),np.full(16384,2,np.float32)),digest(spec))
    recovered=from_wire(to_wire(packet));assert len(recovered.values)==2
    assert not np.array_equal(*recovered.values)
    assert feature_spec()['layout']['streams'][0]['fusion_scope']=='joint'


def test_cross_host_lease_is_atomic_and_unknown_holds_all_cards(tmp_path):
    from AlphaBrain.research.dsol.execution.fleet_slots import FleetSlots
    from benchmarks.LIBERO.prefix_campaign import assign_runtime_hosts
    from benchmarks.LIBERO.controlled_campaign import resident_runtime_slots
    from AlphaBrain.research.dsol.execution.resources import worker_slots
    pool=FleetSlots({'a':[0,1,2],'b':[4,5,6]})
    blocks=[]
    for i in range(4):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps({'runtime':i}))
        for _ in range(2):blocks.append(dict(index=len(blocks),host='a',sim_config=str(p),rows=[{'immutable':i}]))
    expected=deepcopy(blocks)
    with pool.lease_distributed('one',['a','b'],count=4) as capacity:
        assert sum(map(len,capacity.values()))==4
        assign_runtime_hosts(blocks,capacity)
        slots={h:worker_slots(h,ids,workers_per_gpu=1) for h,ids in capacity.items()}
        assigned=resident_runtime_slots({'native_policy':{},'blocks':blocks},slots)
        assert set(assigned)==set(range(8))
        assert [b['rows'] for b in blocks]==[b['rows'] for b in expected]
        for b in blocks:assert all(s in slots[b['host']] for s in assigned[b['index']])
        assert sum(map(len,pool.snapshot()['available'].values()))==2
    assert sum(map(len,pool.snapshot()['available'].values()))==6
    with pytest.raises(RuntimeError,match='unknown'):
        with pool.lease_distributed('unknown',['a','b'],count=4):raise RuntimeError('unknown')
    assert sum(len(v['gpu_ids']) for v in pool.snapshot()['held'].values())==4
    with pytest.raises(RuntimeError,match='quarantined'):
        with pool.lease_distributed('retry',['a','b'],count=4):pass


def test_training_and_development_request_all_native_runtime_groups_before_lease(tmp_path):
    from contextlib import contextmanager
    from benchmarks.LIBERO.prefix_campaign import PrefixCampaign,admission_identity
    cell=dict(family='groot_n15',native_policy={'declared':True},core_spec=core_spec('U12'),
              training_routes={},evaluation_routes={})
    rows=[]
    for i in range(4):
        f=tmp_path/f'sim-{i}.json';f.write_text(json.dumps({'runtime':i}))
        cell['training_routes'][str(i)]=cell['evaluation_routes'][str(i)]=str(f)
        rows.extend([{'suite':str(i)}]*8)
    schedule=tmp_path/'rows.json';schedule.write_text(json.dumps(rows))
    class Pool:
        @contextmanager
        def lease(self,key,hosts,**kwargs):
            assert kwargs['count']==4
            raise RuntimeError('four-runtimes-reserved-before-build')
            yield
    ctx=PrefixCampaign.__new__(PrefixCampaign);ctx.protocol={'cells':{'g':cell}}
    ctx.cfg=dict(hosts={'groot_n15':['node']},training_slots=3,evaluation_slots_max=4)
    ctx.pool=Pool();ctx.proofs={};ctx.variant_proofs={admission_identity(cell):{f'node/{i}':'proof' for i in range(4)}}
    for kind in ('training','evaluation'):
        with pytest.raises(RuntimeError,match='four-runtimes-reserved'):
            ctx.execute('g',kind,str(schedule),version=0)
