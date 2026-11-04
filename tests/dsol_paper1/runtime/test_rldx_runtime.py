from contextlib import contextmanager
import json
import numpy as np
import pytest
from benchmarks.LIBERO import rldx_runtime as r,controlled_rldx as a
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest


def example(value=0):
    image=np.arange(256*256*3,dtype=np.int64).reshape(256,256,3).astype(np.uint8)
    return dict(lang='task',fastwam_raw=dict(schema='LIBERO_RAW_RGB256_PROPRIO_V1',
        agentview_image=image,robot0_eye_in_hand_image=image+np.uint8(value),
        robot0_eef_pos=np.array([value,0,0]),robot0_eef_quat=np.array([0.,0.,0.,1.]),robot0_gripper_qpos=np.zeros(2)))


def test_native_history_camera_roles_and_real_frame_indices():
    e=example();h=r.LiberoHistory(e);obs=h.observation()
    assert obs['video']['front_view'].shape==(1,4,256,256,3)
    assert np.array_equal(obs['video']['front_view'][0,0],e['fastwam_raw']['agentview_image'][::-1,::-1])
    for v in range(1,9):h.append(example(v))
    out=h.observation()
    assert out['state']['eef_pos_absolute'][0,0,0]==8
    assert out['state']['gripper_close'].shape==(1,1,2)
    raw=e['fastwam_raw']['robot0_eye_in_hand_image'][::-1,::-1]
    for i,v in enumerate((2,4,6,8)):
        assert np.array_equal(out['video']['left_wrist_view'][0,i],raw+np.uint8(v))
    changed=example();changed['lang']='another'
    with pytest.raises(ValueError,match='Task changed'):h.append(changed)


def test_native_decode_gripper_and_padding_mask():
    decoded=dict(eef_pos_delta=np.ones((1,16,3)),eef_rot_delta=np.zeros((1,16,3)),gripper_close=np.linspace(0,1,16).reshape(1,16,1))
    before=decoded['gripper_close'].copy();actions=r.simulator_actions(decoded)
    assert actions.shape==(16,7) and actions[0,-1]==-1 and actions[-1,-1]==1
    assert np.array_equal(before,decoded['gripper_close'])
    trace=np.random.default_rng(1).normal(size=(4,1,16,64));score=r.accel_scores(trace)
    trace[:,:,8:,:]=1e6;trace[:,:,:8,7:]=1e6
    assert r.accel_scores(trace)==score
    with pytest.raises(ValueError,match='Degenerate'):r.accel_scores(np.zeros((4,1,16,64)))


def test_direct_real_entries_closed(tmp_path):
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    for fn in (lambda:a.run({},tmp_path/'out',guard=lambda:None),
               lambda:r.RLDXRuntime('x','y',tmp_path,1)):
        with pytest.raises(ProductionDispatchDisabled):fn()
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('failure',['none','unknown','disconnect','mismatch'])
def test_actual_numerical_entry_and_no_replay(tmp_path,monkeypatch,failure):
    from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries
    from benchmarks.LIBERO import search_live
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/controlled_rldx.py:run')
    sim=tmp_path/'sim.json';sim.write_text('{}')
    limits=dict(episodes=0,score=6,paired_execution=0,nominal_execution=0)
    c=dict(sim_config=str(sim),runtime_contract='x',runtime_contract_sha256='y',seeds=[11,12],
           limits=limits,blocks=[dict(index=0,host='fresh',limits=limits,rows=[dict(initial=0,state_key='s')])])
    p=tmp_path/'campaign.json';p.write_text(json.dumps(c));monkeypatch.setattr(a,'validate_campaign',lambda _:c)
    ex=example();ex.update(image=[np.zeros((2,2)),np.zeros((2,2))],state=np.zeros(7),camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4))
    @contextmanager
    def simulator(*args):yield lambda m:dict(example=ex)
    monkeypatch.setattr(search_live,'simulator',simulator)
    class Runtime:
        calls=0
        def __init__(self,cp,stamp,out,limit):self.out=out;self.limit=limit
        def infer(self,history,seed,identity,**kw):
            self.calls+=1
            if failure=='unknown' and self.calls==2:raise RuntimeError('unknown outcome')
            result={k:np.ones((1,)) for k in ('actions','normalized_actions','rng_cpu','rng_cuda','states','velocities','initial_noise','backbone_features','state_features')}
            if failure=='mismatch' and self.calls==3:result['actions']=np.zeros(1)
            with (self.out/'calls.jsonl').open('a') as f:f.write(json.dumps(dict(index=self.calls))+'\n')
            return dict(result,scores={'3':1.,'4':2.},seconds=0.,peak_allocated_mib=0.,peak_reserved_mib=0.)
    monkeypatch.setattr(r,'RLDXRuntime',Runtime)
    checked=[]
    def guard():
        checked.append(1)
        if failure=='disconnect' and len(checked)==4:raise ConnectionError('lost center')
    req=a.request_for(p,0);out=tmp_path/'out'
    if failure!='none':
        with pytest.raises((RuntimeError,ValueError,ConnectionError)):a.run(req,out,guard=guard)
        assert not (out/'admission.json').exists()
        assert not (out/'complete.json').exists()
        return
    a.run(req,out,guard=guard)
    artifacts={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(request_sha256=digest(req),artifacts=artifacts)))
    assert a.validate_block(req,out)['calls']==dict(score=6,execution=0)
    with pytest.raises(FileExistsError):a.run(req,out,guard=guard)


def test_rldx_operation_contract_requires_explicit_gate():
    from AlphaBrain.research.dsol.data.operation_contract import rldx_contract,simulator_action_contract
    contract=rldx_contract('a'*64)
    c=dict(operation_contract=contract,continuous_execution='PAIRED_EVALUATION_ONLY',
           raw_observation='LIBERO_RAW_RGB256_PROPRIO_V1',operation_model='rldx',
           models=dict(rldx=dict(implementation='rldx_native',runtime_contract_sha256='a'*64)))
    assert simulator_action_contract(c)==(8,'float32')
    c['models']['rldx']['runtime_contract_sha256']='b'*64
    with pytest.raises(ValueError):simulator_action_contract(c)


def test_operation_noise_is_paired_and_budget_is_complete():
    from benchmarks.LIBERO import controlled_rldx_operation as op
    assert op.seed_for('state',0,3)==op.seed_for('state',0,3)
    assert len({op.seed_for('state',m,i) for m in (0,1) for i in range(90)})==180
    assert op.limits([{}],'trajectory')==dict(episodes=2,score=0,paired_execution=180,nominal_execution=0)
    assert op.limits([{}],'selection')==dict(episodes=8,score=54,paired_execution=720,nominal_execution=0)


@pytest.mark.parametrize('failure',['none','unknown','disconnect'])
def test_full_rollout_real_history_and_halt(monkeypatch,failure):
    from benchmarks.LIBERO import controlled_rldx_operation as op
    class Runtime:
        calls=0
        def infer(self,history,seed,identity,**kw):
            self.calls+=1
            if failure=='unknown' and self.calls==2:raise RuntimeError('unknown')
            assert history.frames[-1]['eef_pos_absolute'][0]==(self.calls-1)*8
            return dict(normalized_actions=np.zeros((1,16,64)),actions=np.zeros((16,7)))
    stepped=[];guarded=[]
    def query(payload):
        assert payload['kind']=='step' and payload['actions'].shape==(1,7)
        stepped.append(1);return dict(executed=1,success=len(stepped)==11,example=example(len(stepped)),physics=str(len(stepped)))
    def guard():
        guarded.append(1)
        if failure=='disconnect' and len(guarded)==4:raise ConnectionError('disconnected')
    runtime=Runtime()
    if failure!='none':
        with pytest.raises((RuntimeError,ConnectionError)):op.rollout(query,runtime,example(),dict(state_key='s'),0,'trace',guard=guard)
        assert runtime.calls<=2 and len(stepped)<=8
        return
    result,arrays=op.rollout(query,runtime,example(),dict(state_key='s'),0,'trace',guard=guard)
    assert result['success'] and result['steps']==11 and result['replans']==2
    assert arrays['actions'].shape==(11,7) and len(arrays['physics'])==11
    assert len(guarded)==13
