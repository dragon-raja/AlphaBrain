from pathlib import Path
from contextlib import contextmanager
import json
import numpy as np
import pytest
from benchmarks.LIBERO import controlled_target_panel as a
from AlphaBrain.research.dsol.data.operation_contract import fastwam_contract,groot_contract
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest


def test_direct_panel_entry_closed(tmp_path):
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    with pytest.raises(ProductionDispatchDisabled):
        a.run({}, tmp_path/'out', guard=lambda: None)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('model',['fastwam','groot_n15'])
def test_actual_fixed_panel_entry(tmp_path, monkeypatch, failure, model):
    from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries
    from benchmarks.LIBERO import search_live, search_runtime
    from AlphaBrain.research.dsol.data import flow_noise
    allow_synthetic_boundaries(monkeypatch, 'benchmarks/LIBERO/controlled_target_panel.py:run')
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
    fail=[failure]
    class Runtime:
        def __init__(self,sim,output,limit):self.calls=0;self.output=Path(output)
        def infer(self,example,noise,identity,capture=True,*,warmup=False):
            self.calls+=1
            with (self.output/'calls.jsonl').open('a') as f:f.write(json.dumps(dict(identity,index=self.calls))+'\n')
            if fail[0] and identity.get('stage')=='execution' and identity.get('repeat')==1:
                fail[0]=False
                raise RuntimeError('injected unknown model outcome')
            return dict(score=float(np.abs(example['image'][0]-np.eye(4)).sum())+.1,external=np.ones(4608),
                wrist=np.ones(4608),actions=np.zeros((10,7)),full_actions=np.zeros((32,7)),trace=np.zeros((4,10,7)),
                normalized_actions=np.zeros((10,7)),initial_noise=np.zeros((10,7)),timesteps=np.arange(4)/4,
                scores={'3':.1,'4':.2})
    class Bank:
        def __init__(self,*a,**kw):pass
        def get(self,*a):return dict(noise=np.zeros((10,7)),noise_sha256='noise')
    monkeypatch.setattr(search_live,'simulator',simulator);monkeypatch.setattr(search_runtime,'create_policy_runtime',Runtime)
    monkeypatch.setattr(flow_noise,'ExplicitFlowNoiseBank',Bank)
    contract=(groot_contract if model=='groot_n15' else fastwam_contract)('a'*64)
    sim=tmp_path/'sim.json';sim.write_text(json.dumps(dict(operation_contract=contract)))
    noise=tmp_path/'noise.npz';np.savez(noise,noise=np.zeros((1,2,10,7)))
    rows=[dict(state_key='fixed',initial=0,view_id=v,angles_deg=angles,
               score_members=[0,1],execution_repeats=repeats,suite='libero_spatial')
          for v,angles,repeats in [(0,[0.,0.],[]),(1,[-15.,5.],[0,1])]]
    c=dict(sim_config=str(sim),execution_noise='fake',score_noise=str(noise),target_model=model,
        blocks=[dict(index=0,host='fresh',rows=rows,limits=a.limits(rows,contract))])
    p=tmp_path/'campaign.json';p.write_text(json.dumps(c))
    monkeypatch.setattr(a,'validate_campaign',lambda p:c)
    request=a.request_for(p,0);out=tmp_path/'out'
    if failure:
        with pytest.raises(RuntimeError,match='unknown model outcome'):
            a.run(request,out,guard=lambda:None)
        assert not (out/'complete.json').exists()
        attempts=[json.loads(x) for x in (out/'model-attempts.jsonl').read_text().splitlines()]
        assert len(attempts)==6+int(model=='groot_n15')
        assert not (out/'case-0001/episode-1/result.json').exists()
        return
    a.run(request,out,guard=lambda:None)
    artifacts={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(request_sha256=digest(request),artifacts=artifacts)))
    assert a.validate_block(request,out)['calls']==dict(score=4+int(model=='groot_n15'),execution=2)
    value=json.loads((out/'results.json').read_text())
    assert len(value['scores'])==2 and len(value['tasks'])==2
    assert all(t['angles_deg']==[-15.,5.] for t in value['tasks'])
    assert all(t['success'] for t in value['tasks'])
    with np.load(out/'case-0001/query-01.npz') as z:
        key='camera_to_world_opencv' if model=='groot_n15' else 'external_image'
        assert not np.array_equal(z[key],np.eye(4))
        if model=='groot_n15':
            assert z['trace_0'].shape==(4,10,7) and float(z['score_p4_0'])==.2


def test_panel_budget_and_independent_addresses(tmp_path):
    rows=[dict(state_key='s',view_id=0,score_members=[0,1],execution_repeats=[],suite='libero_10'),
          dict(state_key='s',view_id=1,score_members=[0,1],execution_repeats=[0,1],suite='libero_10')]
    cap=a.limits(rows,fastwam_contract('a'*64))
    assert cap==dict(episodes=2,score=4,paired_execution=140,nominal_execution=0)
    p=tmp_path/'campaign.json';p.write_text(json.dumps(dict(blocks=[dict(index=0,host='cy',rows=rows,limits=cap)])))
    req=a.request_for(p,0)
    assert len(req['block']['addresses'])==6
    assert len({digest(x) for x in req['block']['addresses']})==6
    assert all('cy' not in str(x) for x in req['block']['addresses'])


def test_groot_warmup_passes_actual_queue_manifest(tmp_path):
    from benchmarks.LIBERO.continuous_authority_plan import plans
    from AlphaBrain.research.dsol.execution.manifest import task_key
    rows=[dict(state_key='s',view_id=0,score_members=[0,1],execution_repeats=[0,1],suite='libero_spatial')]
    c=dict(target_model='groot_n15',blocks=[dict(index=0,host='gnu',rows=rows,limits=a.limits(rows,groot_contract('a'*64)))])
    p=tmp_path/'campaign.json';p.write_text(json.dumps(c));req=a.request_for(p,0)
    addresses=req['block']['addresses'];key=task_key('fixture',file_digest(p),addresses)
    binding=dict(campaign=str(p),budget=dict(stages=['formal'],campaign_sha256=file_digest(p)),tasks={key:dict(
        stage='formal',host='gnu',addresses=addresses,eligible_slots=[dict(host='gnu',id='gpu0-worker0',gpu_id=0)],
        output=str(tmp_path/'out'),identity_experiment_id='fixture',request=req)})
    bp=tmp_path/'binding.json';bp.write_text(json.dumps(binding))
    plan=plans(binding,bp)['formal']
    assert len(plan['tasks'][0]['addresses'])==5
    assert {x['stream'] for x in addresses}=={'score','execution'}
    assert sum(x['view'].startswith('graph-warmup-') for x in addresses)==1
