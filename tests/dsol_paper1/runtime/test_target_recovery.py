from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO import controlled_target_recovery as a
from AlphaBrain.research.dsol.data.operation_contract import fastwam_contract
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest


def test_direct_entry_closed(tmp_path):
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    with pytest.raises(ProductionDispatchDisabled):a.run({},tmp_path/'out',guard=lambda:None)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('method',['stay','reference','local','random_from_local'])
@pytest.mark.parametrize('recover',[False,True])
def test_actual_paired_target_entry(tmp_path,monkeypatch,method,recover):
    pytest.importorskip('gymnasium')
    from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries
    from benchmarks.LIBERO import search_live,search_runtime
    from AlphaBrain.research.dsol.data import flow_noise
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/controlled_target_recovery.py:run')
    def example(pose):return dict(image=[pose.copy(),np.ones((2,2))],state=np.zeros(7),
        camera_intrinsics=np.eye(3),camera_to_world_opencv=pose.copy(),lang='fixture')
    @contextmanager
    def simulator(path,config,initial,output):
        pose=np.eye(4)
        def receipt():return dict(camera_to_world=pose.copy(),physical_state_id='fixed',settled=True)
        def query(m):
            nonlocal pose
            if m['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=receipt())
            if m['kind']=='move_pose':
                def angles(p):return np.rad2deg([np.arctan2(-p[2,0],p[0,0]),np.arctan2(-p[1,2],p[1,1])])
                target=np.array(m['camera_to_world'])
                assert np.max(np.abs(angles(target)-angles(pose)))<=10.+1e-9
                pose=target;return dict(receipt=receipt())
            if m['kind']=='step':return dict(executed=3,success=True,example=example(pose),physics='final')
            assert m['kind'] in ('query_pose','begin_execution')
            return dict(receipt=receipt(),example=example(pose))
        yield query
    fail=[recover]
    class Runtime:
        def __init__(self,sim,output,limit):self.calls=0;self.output=Path(output)
        def infer(self,example,noise,identity,capture=True):
            self.calls+=1
            with (self.output/'calls.jsonl').open('a') as f:f.write(json.dumps(dict(identity,index=self.calls))+'\n')
            if fail[0] and identity.get('stage')=='execution' and identity.get('repeat')==1:
                fail[0]=False
                raise RuntimeError('injected unknown model outcome')
            return dict(score=float(np.abs(example['image'][0]-np.eye(4)).sum())+.1,external=np.ones(4608),
                wrist=np.ones(4608),actions=np.zeros((10,7)),full_actions=np.zeros((32,7)),trace=np.zeros((10,32,7)))
    class Bank:
        def __init__(self,*a,**kw):pass
        def get(self,*a):return dict(noise=np.zeros((10,7)),noise_sha256='noise')
    monkeypatch.setattr(search_live,'simulator',simulator);monkeypatch.setattr(search_runtime,'create_policy_runtime',Runtime)
    monkeypatch.setattr(flow_noise,'ExplicitFlowNoiseBank',Bank);monkeypatch.setattr(a,'load_actors',lambda c,rows:{})
    sim=tmp_path/'sim.json';sim.write_text(json.dumps(dict(operation_contract=fastwam_contract('a'*64))))
    noise=tmp_path/'noise.npz';np.savez(noise,noise=np.zeros((1,2,10,7)))
    row=dict(state_key='fixed',initial=0,start=0,angles_deg=[-15.,5.],method=method,query_cap=8,
        proposal_seed=13,random_seed=17,suite='libero_spatial')
    limits=a.limits([row],fastwam_contract('a'*64))
    c=dict(sim_config=str(sim),execution_noise='fake',score_noise=str(noise),actors={},score_scale=.16535300100378683,
        blocks=[dict(index=0,host='fresh',rows=[row],limits=limits)])
    p=tmp_path/'campaign.json';p.write_text(json.dumps(c));monkeypatch.setattr(a,'validate_campaign',lambda p:c)
    request=a.request_for(p,0);out=tmp_path/'out'
    if recover:
        with pytest.raises(RuntimeError,match='unknown model outcome'):a.run(request,out,guard=lambda:None)
        proof=tmp_path/'audit.json'
        proof.write_text(json.dumps(dict(status='AUDITED_EXPLICIT_CONTINUATION',old_workers_exited=True,
            automatic_retry=False,uncertain_calls_refunded=False,
            records={digest(row):dict(case=str(out/'case-0000'))},
            artifacts={str(f):file_digest(f) for f in out.rglob('*') if f.is_file()})))
        c.update(explicit_recovery=str(proof),input_hashes={str(proof):file_digest(proof)})
        p=tmp_path/'resumed-campaign.json';p.write_text(json.dumps(c));request=a.request_for(p,0);out=tmp_path/'resumed'
    a.run(request,out,guard=lambda:None)
    artifacts={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(request_sha256=digest(request),artifacts=artifacts)))
    proof=a.validate_block(request,out);assert proof['calls']['execution']==(1 if recover else 2)
    tasks=json.loads((out/'results.json').read_text())['tasks'];assert len(tasks)==2 and all(t['steps']==3 for t in tasks)
    if recover or method in ('stay','reference'):assert proof['calls']['score']==0
    else:assert 0<proof['calls']['score']<=16
