from contextlib import contextmanager
import json
import numpy as np
import pytest
from benchmarks.LIBERO import rldx_spatial_panel as p,rldx_runtime as r
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
from AlphaBrain.research.dsol.execution.resident_runtime import ResidentRuntimeCache


def test_uncontrolled_grid_entry_cannot_start(tmp_path):
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    with pytest.raises(ProductionDispatchDisabled):p.run({},tmp_path/'out',guard=lambda:None)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('failure',['none','unknown','disconnect'])
def test_real_grid_entry_residency_and_no_retry(tmp_path,monkeypatch,failure):
    from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries
    from benchmarks.LIBERO import search_live
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/rldx_spatial_panel.py:run','benchmarks/LIBERO/rldx_runtime.py:RLDXRuntime.bind_task')
    pixels=np.zeros((256,256,3),np.uint8)
    def example(step):return dict(lang='task',fastwam_raw=dict(agentview_image=pixels,robot0_eye_in_hand_image=pixels,
        robot0_eef_pos=np.array([step,0.,0.]),robot0_eef_quat=np.array([0.,0.,0.,1.]),robot0_gripper_qpos=np.zeros(2)))
    sessions=[]
    class Pool:
        @contextmanager
        def session(self,*args):
            sessions.append(args[2]);step=0;pose=np.eye(4)
            def query(m):
                nonlocal step,pose
                receipt=lambda:dict(camera_to_world=pose.copy(),physical_state_id='fixed',settled=True)
                if m['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=receipt())
                if m['kind']=='move_pose':pose=np.asarray(m['camera_to_world']);return dict(receipt=receipt())
                if m['kind'] in ('query_pose','begin_execution'):return dict(example=example(step),receipt=receipt())
                if m['kind']=='step':
                    step+=1;return dict(executed=1,success=step==3,physics=str(step),example=example(step))
                raise AssertionError(m)
            yield query
    monkeypatch.setattr(search_live,'simulator',lambda *a:(_ for _ in ()).throw(AssertionError('Must use explicit fresh-session pool')))
    made=[]
    class Runtime:
        bind_task=r.RLDXRuntime.bind_task
        def __init__(self,cp,stamp,out,limit):
            made.append(self);self.output=out;self.calls=0;self.limit=limit;self.failed=False
        def infer(self,history,seed,identity,**kw):
            self.calls+=1
            row=dict(index=self.calls,identity=identity)
            with (self.output/'model-attempts.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if failure=='unknown' and len(sessions)==2:self.failed=True;raise RuntimeError('Unknown model outcome')
            with (self.output/'calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            return dict(normalized_actions=np.zeros((1,16,64)),actions=np.zeros((16,7)))
    monkeypatch.setattr(r,'RLDXRuntime',Runtime)
    sim=tmp_path/'sim.json';sim.write_text('{}')
    rows=[dict(state_key='s',initial=0,suite='libero_spatial',view_id=i,angles_deg=[float(i),0.],repeat=0) for i in range(2)]
    c=dict(schema=p.SCHEMA,runtime_contract='cp',runtime_contract_sha256='a'*64,sim_config=str(sim),
           blocks=[dict(index=i,host='fresh',rows=[row],limits=p.limits([row])) for i,row in enumerate(rows)])
    campaign=tmp_path/'campaign.json';campaign.write_text(json.dumps(c));monkeypatch.setattr(p,'validate_campaign',lambda _:c)
    cache=ResidentRuntimeCache();checks=[]
    def guard():
        checks.append(1)
        if failure=='disconnect' and len(sessions)==2 and len(checks)>6:raise ConnectionError('Lost center')
    for i in range(2):
        request=p.request_for(campaign,i);out=tmp_path/f'out-{i}'
        if i==1 and failure!='none':
            with pytest.raises((RuntimeError,ConnectionError)):p.run(request,out,guard=guard,runtime_cache=cache,simulator_pool=Pool())
            assert cache.failed and not (out/'complete.json').exists()
            with pytest.raises(RuntimeError,match='Failed resident'): 
                with cache.lease('a'*64,lambda:None,lambda _:None):pass
            break
        p.run(request,out,guard=guard,runtime_cache=cache,simulator_pool=Pool())
        artifacts={str(f.relative_to(out)):file_digest(f) for f in out.rglob('*') if f.is_file()}
        (out/'complete.json').write_text(json.dumps(dict(request_sha256=digest(request),artifacts=artifacts)))
        assert p.validate_block(request,out)['calls']==dict(score=0,execution=1)
    assert len(made)==1 and cache.loads==1 and cache.reuses==1
    assert sessions==[0,0]  # every operation initializes a fresh environment
