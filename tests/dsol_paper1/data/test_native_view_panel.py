import pytest

from AlphaBrain.research.dsol.data.native_view_panel import blocks, views, validate_coverage
from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled


def states():
    return [dict(state_key=f's-{i}',task=f't-{i//2}',initial=i,max_steps=300 if i<4 else (220 if i<6 else 280)) for i in range(8)]


def test_complete_native_panel_budget_and_reuse():
    bs=blocks(states());rows=[r for b in bs for r in b['rows']]
    assert len(bs)==72 and len(rows)==216 and len(views())==27
    assert sum(b['limits']['episodes'] for b in bs)==768
    assert sum(b['limits']['paired_execution']+b['limits']['score'] for b in bs)==43072
    assert all(r['execution_repeats']==[] for r in rows if r['view_id'] in (0,25,26))
    assert sum(len(r['score_members']) for r in rows)==832
    validate_coverage(rows)
    with pytest.raises(ValueError):validate_coverage(rows[:-1])
    with pytest.raises(ValueError):blocks(states()[:-1])
    changed=states();changed[0]['max_steps']=301
    with pytest.raises(ValueError,match='budget'):blocks(changed)


def test_panel_direct_entry_is_closed_before_input_or_model_access(tmp_path):
    from benchmarks.LIBERO.native_view_panel import run
    with pytest.raises(ProductionDispatchDisabled):run({},tmp_path/'output',guard=lambda:None)
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('resident',[False,True])
def test_actual_panel_loop_with_cpu_simulator_and_model_fakes(tmp_path,monkeypatch,resident):
    import json
    import numpy as np
    from contextlib import contextmanager
    from pathlib import Path
    from benchmarks.LIBERO import native_view_panel as module
    from benchmarks.LIBERO.local_search_rollout import input_identity
    from AlphaBrain.research.dsol.execution.dispatch_policy import _live_model_scope
    from AlphaBrain.research.dsol.execution.manifest import digest,file_digest
    from AlphaBrain.research.dsol.execution.resident_runtime import ResidentRuntimeCache
    from tests.dsol_paper1.paths import ROOT
    plan=json.loads((ROOT/'configs/dsol_paper1/search/continuous_search_v1.json').read_text())
    sim=tmp_path/'sim.json';sim.write_text('{}')
    design=tmp_path/'design.json';design.write_text(json.dumps(plan))
    paired=tmp_path/'paired.json';paired.write_text(json.dumps(dict(design=str(design),cohorts=dict(A=dict(sim_config=str(sim),execution_noise='fake')))))
    noises=tmp_path/'noise.npz';np.savez(noises,noise=np.zeros((1,4,10,7),np.float32))
    baseline=dict(state=np.zeros(8),lang='fake',image=[np.zeros((2,2,3),np.uint8)]*2,camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4))
    c=dict(paired_config=str(paired),score_noise=str(noises),state_keys=['s'],reuse_inputs={'s::0':dict(input_sha256=input_identity(baseline))})
    row=dict(cohort='A',initial=0,state_key='s',task='fake',view_id=1,angles_deg=[10,0],suite='libero_goal',score_members=[0,1,2,3],execution_repeats=[0,1,2,3])
    req=dict(campaign='fake',block=dict(index=0,rows=[row],limits=dict(episodes=4,score=4,paired_execution=240,nominal_execution=0)))
    monkeypatch.setattr(module,'validate_campaign',lambda p:c)
    monkeypatch.setattr(module,'request_for',lambda *a:req)
    loaded=[]
    class Runtime:
        def __init__(self,config,output,limit):
            self.calls=0;self.output=output;self.limit=limit;loaded.append(self)
        def infer(self,example,noise,identity,capture=True):
            self.calls+=1
            value=dict(actions=np.zeros((10,7)),trace=np.zeros((10,10,7)))
            if capture:value['score']=.02
            return value
    class Bank:
        def __init__(self,*a,**k):pass
        def get(self,*a):return dict(noise=np.zeros((10,7),np.float32),noise_sha256='0'*64)
    monkeypatch.setattr('benchmarks.LIBERO.search_runtime.create_policy_runtime',Runtime)
    monkeypatch.setattr('AlphaBrain.research.dsol.data.flow_noise.ExplicitFlowNoiseBank',Bank)
    monkeypatch.setattr('benchmarks.LIBERO.search_live.check_simulator_dependencies',lambda c:None)
    opened=[]
    @contextmanager
    def simulator(*args):
        opened.append(str(args[-1]));pose=np.eye(4)
        def query(message):
            nonlocal pose
            if message['kind']=='move_pose':pose=message['camera_to_world']
            receipt=dict(camera_to_world=pose,physical_state_id='fixed',settled=True)
            example=dict(baseline,camera_to_world_opencv=pose)
            if message['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=receipt)
            if message['kind']=='step':return dict(example=example,executed=1,success=True)
            return dict(example=example,receipt=receipt)
        yield query
    monkeypatch.setattr('benchmarks.LIBERO.search_live.simulator',simulator)
    guards=[];guard=lambda:guards.append(1)
    launch=dict(operation='LIBERO_BLOCK',permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1'))
    out=tmp_path/'output'
    cache=ResidentRuntimeCache() if resident else None
    with _live_model_scope(launch,guard):module.run(req,out,guard=guard,runtime_cache=cache)
    assert len(opened)==5 and len(guards)>=10
    status=json.loads((out/'status.json').read_text());assert status['calls']==dict(score=4,execution=4)
    timing=json.loads((out/'timing.json').read_text())['phases']
    assert timing['authority_wait']['count']==8
    for phase,count in [('score_simulator',1),('execution_simulator',4)]:
        assert timing[phase+'_start']['count']==count
        assert timing[phase+'_close']['count']==count
    if resident:
        next_out=tmp_path/'next-output'
        with _live_model_scope(launch,guard):module.run(req,next_out,guard=guard,runtime_cache=cache)
        assert len(loaded)==1 and cache.loads==1 and cache.reuses==1
        assert json.loads((next_out/'status.json').read_text())['calls']==dict(score=4,execution=4)
        assert json.loads((next_out/'timing.json').read_text())['runtime_reused'] is True
        first=json.loads((out/'results.json').read_text())
        second=json.loads((next_out/'results.json').read_text())
        assert first['scores']==second['scores']
        for a,b in zip(first['tasks'],second['tasks']):
            assert {k:v for k,v in a.items() if k!='trajectory'}=={k:v for k,v in b.items() if k!='trajectory'}
    artifacts={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',request_sha256=digest(req),artifacts=artifacts)))
    assert module.validate_block(req,out)['calls']==dict(score=4,execution=4)
    (out/'results.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):module.validate_block(req,out)
