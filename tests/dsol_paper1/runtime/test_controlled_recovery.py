"""Synthetic actual entry exercises motion, score isolation, execution and receipts."""
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO import controlled_recovery as adapter
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest


def save(path,value):path.write_text(json.dumps(value));return str(path)


def test_direct_entry_is_closed(tmp_path):
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    with pytest.raises(ProductionDispatchDisabled):adapter.run({},tmp_path/'out',guard=lambda:None)
    assert not (tmp_path/'out').exists()


def test_independent_random_never_loads_actor():
    from AlphaBrain.research.dsol.data.independent_random import SCHEMA
    assert adapter.load_actors(dict(schema=SCHEMA,actors={}))=={}
    with pytest.raises(ValueError):adapter.load_actors(dict(schema=SCHEMA,actors={'foreign':{}}))


def test_random_control_only_draws_visited_candidates_without_return_duplicates():
    info=dict(events=[dict(kind=k,angles=a) for k,a in [('initial',[1,2]),('search',[3,4]),('return',[1,2])]])
    values=[adapter.random_visited(info,i) for i in range(100)]
    assert {tuple(v[0]) for v in values}=={(1,2),(3,4)} and {v[2] for v in values}=={2}
    assert adapter.random_visited(info,8)==adapter.random_visited(info,8)


@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    pytest.importorskip('gymnasium')
    from benchmarks.LIBERO import search_live,search_runtime,local_search_rollout
    from AlphaBrain.research.dsol.data import flow_noise
    from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
    from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/controlled_recovery.py:run')
    def example(pose):return dict(image=[pose.copy(),np.ones((2,2))],state=np.zeros(7),
        camera_intrinsics=np.eye(3),camera_to_world_opencv=pose.copy(),instruction='synthetic',lang='synthetic')
    @contextmanager
    def simulator(path,config,initial,output):
        pose=np.eye(4)
        def receipt():return dict(camera_to_world=pose.copy(),physical_state_id='fixed',settled=True)
        def query(message):
            nonlocal pose
            kind=message['kind']
            if kind=='continuous_info':return dict(reference=np.eye(4),receipt=receipt())
            if kind=='move_pose':pose=np.array(message['camera_to_world']);return dict(receipt=receipt())
            if kind=='step':return dict(executed=len(message['actions']),success=True,example=example(pose))
            assert kind in ('query_pose','begin_execution')
            return dict(receipt=receipt(),example=example(pose))
        yield query
    class Runtime:
        def __init__(self,sim,output,limit):self.calls=0
        def infer(self,example,noise,identity,capture=True):
            self.calls+=1
            return dict(score=float(np.abs(example['image'][0]-np.eye(4)).sum()),external=np.ones(4608),
                wrist=np.ones(4608),actions=np.zeros((10,7)),trace=np.zeros((10,10,7)))
    class Bank:
        def __init__(self,*args,**kwargs):pass
        def get(self,*args):return dict(noise=np.zeros((10,7)),noise_sha256='synthetic')
    monkeypatch.setattr(search_live,'simulator',simulator)
    monkeypatch.setattr(search_runtime,'create_policy_runtime',Runtime)
    monkeypatch.setattr(flow_noise,'ExplicitFlowNoiseBank',Bank)
    monkeypatch.setattr(local_search_rollout,'validate_plan',lambda p:None)
    monkeypatch.setattr(adapter,'load_actors',lambda c:{})
    sim=save(tmp_path/'sim.json',{})
    plan=save(tmp_path/'plan.json',dict(evaluation=dict(execution_repeat_ids=[0,1],horizons=dict(libero_spatial=220)),
        operation_policy=dict(predicted_actions=10,executed_actions_per_replan=5,clip=[-1,1])))
    paired=save(tmp_path/'paired.json',dict(design=plan,cohorts=dict(A=dict(sim_config=sim,execution_noise='synthetic'))))
    noise=tmp_path/'noise.npz';np.savez(noise,noise=np.zeros((1,4,10,7)))
    query=tmp_path/'query.npz';first=example(PanTiltMount(np.eye(4)).pose([-15,5]))
    np.savez(query,external_image=first['image'][0],wrist_image=first['image'][1],robot_state=first['state'],
        camera_intrinsics=first['camera_intrinsics'],camera_to_world_opencv=first['camera_to_world_opencv'])
    refs=save(tmp_path/'refs.json',dict(views={'state::0':dict(input_sha256=local_search_rollout.input_identity(example(np.eye(4)))),
        'state::1':dict(query=str(query))}))
    row=dict(cohort='A',state_key='state',initial=0,task='synthetic',split='validation',view_id=1,
        angles_deg=[-15,5],method='local',query_cap=8,proposal_seed=13,random_seed=17,suite='libero_spatial')
    block=dict(index=0,host='node',rows=[row],limits=dict(episodes=2,score=18,paired_execution=88,nominal_execution=0))
    c=dict(schema='NATIVE_CONTROLLED_RECOVERY_V1',paired_config=paired,score_noise=str(noise),baseline_references=refs,actors={},score_scale=.16535300100378683,blocks=[block])
    path=Path(save(tmp_path/'campaign.json',c));monkeypatch.setattr(adapter,'validate_campaign',lambda p,**kw:json.loads(Path(p).read_text()))
    return path,c


@pytest.mark.parametrize('method',['local','random_from_local'])
def test_actual_entry_and_artifact_verification(synthetic,tmp_path,method):
    path,c=synthetic;c['blocks'][0]['rows'][0]['method']=method;save(path,c)
    request=adapter.request_for(path,0);output=tmp_path/'out';calls=[]
    adapter.run(request,output,guard=lambda:calls.append(1))
    result=json.loads((output/'results.json').read_text());search=result['searches'][0]
    assert len(result['tasks'])==2 and all(t['success'] for t in result['tasks'])
    assert not search['validation_used_for_selection'] and search['validation_queries']==1
    assert search['score_calls']==2*search['queries']+2<=18
    assert len(calls)==search['score_calls']+2
    assert json.loads((output/'search-000/nominal-return.json').read_text())['model_calls']==0
    artifacts={str(p.relative_to(output)):file_digest(p) for p in output.rglob('*') if p.is_file()}
    save(output/'complete.json',dict(status='COMPLETE',request_sha256=digest(request),artifacts=artifacts))
    assert adapter.validate_block(request,output)['calls']['execution']==2
    (output/'search-000/episode-0/result.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact'):adapter.validate_block(request,output)


def test_disconnect_stops_actual_entry_without_replaying(synthetic,tmp_path):
    path,_=synthetic;count=[]
    def guard():
        count.append(1)
        if len(count)==2:raise ConnectionError('synthetic disconnected controller')
    with pytest.raises(ConnectionError):adapter.run(adapter.request_for(path,0),tmp_path/'out',guard=guard)
    lines=(tmp_path/'out/model-attempts.jsonl').read_text().splitlines()
    assert len(lines)==1 and not (tmp_path/'out/status.json').exists()
    with pytest.raises(FileExistsError):adapter.run(adapter.request_for(path,0),tmp_path/'out',guard=lambda:None)


@pytest.mark.parametrize('independent',[False,True])
def test_paired_diagnostics_charge_four_calls_without_reselecting(synthetic,tmp_path,independent):
    from AlphaBrain.research.dsol.data.reward_pilot import RECOVERY_SCHEMA
    path,c=synthetic
    before=tmp_path/'old';adapter.run(adapter.request_for(path,0),before,guard=lambda:None)
    expected=json.loads((before/'results.json').read_text())
    c['schema']=RECOVERY_SCHEMA;c['blocks'][0]['limits']['score']=20;save(path,c)
    if independent:
        from AlphaBrain.research.dsol.data.independent_random import SCHEMA
        c['schema']=SCHEMA
        c['blocks'][0]['rows'][0].update(method='random_acquisition-seed-0',selector='random_acquisition',acquisition_seed=183)
        save(path,c)
    request=adapter.request_for(path,0);out=tmp_path/'paired'
    adapter.run(request,out,guard=lambda:None)
    result=json.loads((out/'results.json').read_text());s=result['searches'][0]
    if not independent:
        assert s['selected_angles']==expected['searches'][0]['selected_angles']
        assert s['queries']==expected['searches'][0]['queries']
    else:
        assert s['acquisition_rule']=='uniform_legal_increment' and s['selection_rule']=='minimum_visited'
        assert s['queries'] in (7,8)
    assert s['validation_queries']==2 and s['score_calls']==2*s['queries']+4
    assert len(s['reference_validation_scores'])==2
    assert s['diagnostic_reference_angles']==c['blocks'][0]['rows'][0]['angles_deg']
    if not independent:
        for a,b in zip(result['tasks'],expected['tasks']):
            for k in ('success','steps','replans','input_sha256','execution_noise_sha256'):assert a[k]==b[k]
    artifacts={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}
    save(out/'complete.json',dict(status='COMPLETE',request_sha256=digest(request),artifacts=artifacts))
    assert adapter.validate_block(request,out)['calls']['score']==s['score_calls']
