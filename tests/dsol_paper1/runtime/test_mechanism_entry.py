"""Exercise actual evaluation entry with synthetic sensor and actor backends."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from tests.dsol_paper1.helpers.local_search import synthetic_session

@pytest.mark.parametrize('cap',[4,6,8,16])
def test_actual_search_enforces_cap_and_charges_return(tmp_path,monkeypatch,cap):
    from benchmarks.LIBERO import continuous_evaluation as entry,local_view_runtime
    from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec
    core,source=synthetic_session();source.mount=core.mount;source.physical='synthetic'
    source.current=np.array([0.,0.]);source.last_example={};source.last_values=[dict(wrist=np.ones(4))]
    original=source.query
    def query():
        result=original();source.queries=source.calls;return result
    source.query=query;source.queries=0
    source.move_angles=lambda angles:source.move(source.mount.pose(angles))
    monkeypatch.setattr(local_view_runtime,'RenderedViewSource',lambda *a,**kw:source)
    monkeypatch.setattr(entry,'actor_action',lambda *a,**kw:dict(stop=False,delta_deg=np.array([1.125,0.])))
    actor=SimpleNamespace(mechanism=mechanism_spec('Q'+str(cap)) if cap!=8 else {})
    info,_=entry.search(dict(method='continuous_ppo',angles_deg=[0.,0.],proposal_seed=19,query_cap=cap),None,None,None,tmp_path,actor,1.)
    assert info['queries']==source.calls==cap and info['reason']=='budget'
    np.testing.assert_array_equal(info['selected_angles'],[0.,0.])


def test_campaign_deployment_only_does_not_train(tmp_path):
    from benchmarks.LIBERO.prefix_campaign import PrefixCampaign
    from threading import RLock
    co=PrefixCampaign.__new__(PrefixCampaign);co.root=tmp_path;co.cfg={};co.lock=RLock();co.failures={}
    co.protocol=dict(design_stage='mechanism_screen',cells={'Q4':dict(evaluation_only=True,evaluation_schedules={'panel':dict(version=16)})})
    calls=[];co.execute=lambda n,k,s,**kw:calls.append((k,kw['version']));co.status=lambda *a,**kw:None
    co.cpu_update=lambda *a:pytest.fail('Must never update deployment-only actor')
    co.cell('Q4');assert calls==[('evaluation',16)]


@pytest.mark.parametrize('cap',[4,6,8,16])
def test_real_source_and_actual_actor_entry_exhaust_declared_cap(tmp_path,cap):
    """Do not replace RenderedViewSource: the real source previously stopped at 8."""
    torch=pytest.importorskip('torch')
    from benchmarks.LIBERO import continuous_evaluation as entry
    from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec
    pose=np.eye(4);sizes=[]
    def request(cmd):
        nonlocal pose
        if cmd['kind']=='move_pose':pose=np.array(cmd['camera_to_world'],copy=True)
        receipt=dict(camera_to_world=pose.copy(),physical_state_id='synthetic',settled=True)
        if cmd['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=receipt)
        if cmd['kind']=='move_pose':return dict(receipt=receipt)
        assert cmd['kind']=='query_pose'
        return dict(receipt=receipt,example=dict(lang='task',image=[np.zeros((2,2,3),np.uint8)]*2,
            state=np.zeros(7),camera_intrinsics=np.eye(3),camera_to_world_opencv=pose.copy()))
    class Runtime:
        calls=0
        def infer(self,example,*args,**kwargs):
            self.calls+=1
            score=1. if np.array_equal(example['camera_to_world_opencv'],np.eye(4)) else 2.
            return dict(score=score,actions=np.zeros((10,7)),trace=np.zeros((2,10,7)),external=np.ones(4),wrist=np.ones(4))
    def distribution(obs):
        sizes.append(int((obs['flags'][0,:,0]>.5).sum()))
        assert obs['geometry'].shape[1]==(15 if cap==16 else 7)
        return SimpleNamespace(sample=lambda **kw:dict(stop=torch.tensor([False]),delta_deg=torch.tensor([[1.125,0.]])))
    actor=SimpleNamespace(mechanism=mechanism_spec('Q'+str(cap)) if cap!=8 else {},distribution=distribution)
    runtime=Runtime()
    info,_=entry.search(dict(method='continuous_ppo',angles_deg=[0.,0.],proposal_seed=19,query_cap=cap),
        request,runtime,np.zeros((2,10,7)),tmp_path,actor,1.)
    assert info['queries']==cap and runtime.calls==2*cap
    assert len(list(tmp_path.glob('query-*.npz')))==cap
    assert info['reason']=='budget'
    np.testing.assert_array_equal(info['selected_angles'],[0.,0.])
    if cap==16:assert max(sizes)>8


def test_unbound_long_cap_rejected_before_sensor_or_model_access(tmp_path):
    from benchmarks.LIBERO.continuous_evaluation import search
    def forbidden(*args,**kwargs):pytest.fail('Unbound budget reached sensor/model')
    with pytest.raises(ValueError,match='bound deployment'):
        search(dict(query_cap=16),forbidden,forbidden,None,tmp_path,SimpleNamespace(),1.)


def test_capacity_admission_forces_full_history_even_when_actor_stops(tmp_path):
    torch=pytest.importorskip('torch')
    from benchmarks.LIBERO.prefix_admission import CapacityProbeActor
    from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec
    calls=[]
    def distribution(obs):
        calls.append(True)
        return SimpleNamespace(sample=lambda **kw:dict(stop=torch.tensor([True]),delta_deg=torch.zeros((1,2))))
    actor=SimpleNamespace(mechanism=mechanism_spec('Q16'),distribution=distribution)
    probe=CapacityProbeActor(actor)
    obs=dict(flags=torch.ones((1,15,4)),delta_low=torch.tensor([[-1.,-1.]]),delta_high=torch.tensor([[1.,1.]]))
    value=probe.distribution(obs).sample(deterministic=True)
    assert len(calls)==1 and probe.maximum_history==15 and not value['stop'].item()
    assert value['delta_deg'][0,0].item()==.125
    assert actor.distribution(obs).sample()['stop'].item()


def test_recovery_admits_only_explicit_active_variants():
    from benchmarks.LIBERO.prefix_campaign import PrefixCampaign
    co=PrefixCampaign.__new__(PrefixCampaign)
    co.protocol=dict(schema='UNIFIED_CORE_PILOT_V1',seeds=[0,1,2],cells={
        'old':dict(family='pi05',seed=0),'repair':dict(family='pi05',seed=0)})
    co.cfg=dict(active_cells=['repair']);calls=[]
    co.admit_cell=lambda *a:calls.append(a)
    co.admission('pi05','node')
    assert calls==[('repair','pi05','node')]
