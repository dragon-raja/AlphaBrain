import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO.continuous_execution import ContinuousExecutionGate
from benchmarks.LIBERO.continuous_evaluation import EvaluationCalls, audit_block, search
from AlphaBrain.research.dsol.analysis.continuous_evaluation import summarize, METHODS
from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
from reports.paper1.continuous_evaluation_tracking import paired_rows


@pytest.mark.parametrize('config',[{}, {'continuous_execution':True}, {'continuous_execution':'training'}])
def test_training_cannot_execute(config):
    gate=ContinuousExecutionGate(config)
    for kind in ('continuous_info','move_pose','query_pose','close'):gate.allow(kind)
    for kind in ('step','begin_execution','reset','catalog'):
        with pytest.raises(ValueError):gate.allow(kind)


def test_one_way_execution_camera_lock():
    gate=ContinuousExecutionGate({'continuous_execution':'PAIRED_EVALUATION_ONLY'})
    with pytest.raises(ValueError):gate.allow('step')
    gate.begin();gate.allow('step');gate.allow('close')
    for kind in ('begin_execution','move_pose','query_pose','reset','catalog','continuous_info'):
        with pytest.raises(ValueError):gate.allow(kind)


def paired_fixture():
    tasks=[]; searches=[]
    for j, method in enumerate(METHODS):
        base=dict(cohort='A',initial=2,start=0,task='cheese',method=method,queries=1 if j==0 else 8)
        searches.append(base)
        for seed in range(4):
            success=([True,True,False,False],[False,True,True,False],[True,True,True,True])[j][seed]
            tasks.append(dict(base,repeat=seed,success=success,replans=1,execution_noise_sha256=[str(seed)]))
    return tasks,searches


def test_pair_denominators_missing_and_query_session_weighting(tmp_path):
    tasks, searches=paired_fixture()
    result=summarize(tasks,searches)
    local=result['scopes']['all'][METHODS[1]]
    assert local['success_rate']==.5 and local['harm']==1 and local['rescue']==1
    assert local['harm_rate']==.25 and local['conditional_harm_rate']==.5
    assert local['search_sessions']==1 and local['mean_queries']==8
    partial=summarize(tasks[:-1],searches)
    assert partial['complete_pairs']==3 and partial['incomplete_pairs']==1
    assert partial['scopes']['all']['continuous_ppo']['n']==3
    (tmp_path/'paired-00.json').write_text(json.dumps(result))
    row=paired_rows(tmp_path)[0]
    assert row['evaluation/complete_pairs']==4
    assert row['real_task/all/local_near_best_accel/harm_rate']==.25
    assert not any('reward' in k for k in row)


def test_gate_is_engineering_not_success_threshold():
    tasks,searches=paired_fixture()
    for row in tasks:row['success']=False
    assert audit_block(tasks,searches)['status']=='PASS_ENGINEERING_ONLY'
    tasks[-1]['execution_noise_sha256']=['wrong']
    with pytest.raises(ValueError,match='noise'):audit_block(tasks,searches)


def test_attempts_are_durable_and_stage_caps_separate(tmp_path):
    class Runtime:
        calls=0
        def infer(self,*args,**kwargs):
            self.calls+=1
            raise RuntimeError('injected failure')
    calls=EvaluationCalls(Runtime(),tmp_path,{'score':1,'execution':2})
    calls.stage='score';calls.identity={'method':'stay'}
    with pytest.raises(RuntimeError):calls.infer({'state':np.ones(8)},np.ones((10,7)),{},True)
    assert calls.counts=={'score':1,'execution':0}
    assert len((tmp_path/'model-attempts.jsonl').read_text().splitlines())==1
    with pytest.raises(ValueError):calls.infer({},np.ones((10,7)),{},True)


@pytest.mark.parametrize('method',METHODS)
def test_search_real_source_lifecycle_without_models(method,tmp_path,monkeypatch):
    pytest.importorskip('gymnasium')
    # Actual RenderedViewSource + core logic; injected CPU renderer/inference.
    mount=PanTiltMount(np.eye(4)); pose=np.eye(4)
    def request(command):
        nonlocal pose
        kind=command['kind']
        if kind=='move_pose':pose=command['camera_to_world'].copy()
        receipt=dict(camera_to_world=pose.copy(),physical_state_id='fixed',settled=True)
        if kind=='continuous_info':return dict(reference=np.eye(4),receipt=receipt)
        if kind=='move_pose':return dict(receipt=receipt)
        assert kind=='query_pose'
        example=dict(lang='fixture',language='fixture',image=[np.zeros((2,2,3),np.uint8)]*2,state=np.ones(8),
            camera_intrinsics=np.eye(3),camera_to_world_opencv=pose.copy())
        return dict(receipt=receipt,example=example)
    class Runtime:
        calls=0
        def infer(self,example,noise,identity,capture):
            self.calls+=1
            return dict(score=1+float(abs(example['camera_to_world_opencv'][0,2])),
                actions=np.zeros((10,7)),trace=np.zeros((10,10,7)),external=np.ones(4),wrist=np.ones(4))
    if method=='continuous_ppo':
        monkeypatch.setattr('benchmarks.LIBERO.continuous_evaluation.actor_action',
            lambda *args:dict(stop=False,delta_deg=np.array([1.125,-.25])))
    row=dict(method=method,angles_deg=[-17.5,9.5],proposal_seed=7,query_cap=1 if method=='stay' else 8)
    runtime=Runtime()
    info,example=search(row,request,runtime,np.zeros((2,10,7)),tmp_path,None,1.)
    assert info['status']=='COMPLETE' and runtime.calls==2*info['queries']
    assert info['queries']<=(1 if method=='stay' else 8)
    np.testing.assert_allclose(example['camera_to_world_opencv'],mount.pose(info['selected_angles']))
    assert len(list(tmp_path.glob('query-*.npz')))==info['queries']


def test_complete_192_task_driver_with_injected_cpu_backend(tmp_path,monkeypatch):
    from contextlib import contextmanager
    torch=pytest.importorskip('torch')
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    from benchmarks.LIBERO.continuous_evaluation import run
    plan=json.loads((Path(__file__).resolve().parents[3]/'configs/dsol_paper1/search/continuous_search_v1.json').read_text())
    actor=tmp_path/'actor.pt';torch.save(dict(update=16,episodes=512,state_dict=ContinuousActorCritic(4608).state_dict()),actor)
    controls=tmp_path/'controls.pt';torch.save([],controls)
    schedule=[]
    for task in plan['evaluation']['tasks']:
        for i in task['indices']:
            for start,angles in enumerate(plan['camera']['evaluation_start_angles_deg']):
                for method in METHODS:
                    schedule.append(dict(cohort=task['cohort'],initial=i,state_key=str(i),task=task['task'],
                        start=start,angles_deg=angles,method=method,proposal_seed=1,
                        query_cap=1 if method=='stay' else 8,execution_repeats=[0,1,2,3]))
    schedule_file=tmp_path/'schedule.json';schedule_file.write_text(json.dumps(schedule))
    sim=tmp_path/'sim.json';sim.write_text('{}')
    score=tmp_path/'score.npz';np.savez(score,noise=np.zeros((32,2,10,7)))
    cfg=dict(authorization='EXPLICIT_PAIRED_EVALUATION_ONLY',actor=str(actor),actor_controls=str(controls),
        schedule=str(schedule_file),limits=dict(score=544,execution=10560),input_hashes={str(actor):file_digest(actor)},
        cohorts={k:dict(sim_config=str(sim),score_noise=str(score),score_key='noise',execution_noise='fake') for k in ('A','B')})
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    monkeypatch.setattr('benchmarks.LIBERO.continuous_evaluation.validate_inputs',lambda path:(cfg,plan,{}))
    monkeypatch.setattr('benchmarks.LIBERO.search_live.check_simulator_dependencies',lambda cfg:None)
    class Bank:
        def __init__(self,*args,**kwargs):pass
        def get(self,state,repeat,replan):return dict(noise=np.zeros((10,7)),noise_sha256=f'{state}-{repeat}-{replan}')
    monkeypatch.setattr('AlphaBrain.research.dsol.data.flow_noise.ExplicitFlowNoiseBank',Bank)
    class Runtime:
        def __init__(self,*args):self.calls=0
        def infer(self,example,noise,identity,capture=True):
            self.calls+=1
            return dict(score=1+abs(float(example['camera_to_world_opencv'][0,2])),actions=np.zeros((10,7)),
                trace=np.zeros((10,10,7)),external=np.ones(4608),wrist=np.ones(4608))
    monkeypatch.setattr('benchmarks.LIBERO.search_runtime.create_policy_runtime',Runtime)
    @contextmanager
    def simulator(*args):
        pose=np.eye(4);gate=ContinuousExecutionGate({'continuous_execution':'PAIRED_EVALUATION_ONLY'})
        def request(command):
            nonlocal pose
            kind=command['kind'];gate.allow(kind)
            if kind=='move_pose':pose=command['camera_to_world'].copy()
            receipt=dict(camera_to_world=pose.copy(),physical_state_id='fixed',settled=True)
            example=dict(lang='fixture',language='fixture',image=[np.zeros((2,2,3),np.uint8)]*2,
                state=np.ones(8),camera_intrinsics=np.eye(3),camera_to_world_opencv=pose.copy())
            if kind=='continuous_info':return dict(reference=np.eye(4),receipt=receipt)
            if kind=='move_pose':return dict(receipt=receipt)
            if kind=='begin_execution':gate.begin()
            if kind=='step':return dict(example=example,success=True,executed=1)
            return dict(example=example,receipt=receipt)
        yield request
    monkeypatch.setattr('benchmarks.LIBERO.search_live.simulator',simulator)
    output=tmp_path/'results';run(path,output,execute=True)
    status=json.loads((output/'status.json').read_text())
    assert status['tasks']==192 and status['searches']==48 and status['training_updates']==0
    assert status['summary']['complete_pairs']==64 and status['calls']['execution']==192
    assert status['calls']['score']<=544
    assert json.loads((output/'first-block-gate.json').read_text())['tasks']==12
    assert len(paired_rows(output))==16 and not (output/'failure.json').exists()
