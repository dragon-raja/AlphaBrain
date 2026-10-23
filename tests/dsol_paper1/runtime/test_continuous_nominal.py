import json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pytest

from benchmarks.LIBERO.continuous_nominal import nominal_rows,run
from AlphaBrain.research.dsol.execution.manifest import file_digest


def inputs():
    plan=json.loads((Path(__file__).resolve().parents[3]/'configs/dsol_paper1/search/continuous_search_v1.json').read_text())
    rows=[dict(cohort=t['cohort'],initial=i,task=t['task'],state_key=f'{t["cohort"]}:{i}',start=start,
        method=method,angles_deg=plan['camera']['evaluation_start_angles_deg'][start],query_cap=1 if method=='stay' else 8,execution_repeats=[0,1,2,3]) for t in plan['evaluation']['tasks'] for i in t['indices']
        for start in range(2) for method in plan['evaluation']['methods']]
    return plan,rows


def test_nominal_counts_once_per_state_seed_not_per_perturbation_start():
    plan,rows=inputs();result=nominal_rows(plan,rows)
    assert len(result)==8 and sum(len(r['execution_repeats']) for r in result)==32
    assert all(r['angles_deg']==[0,0] and r['method']=='nominal' for r in result)
    with pytest.raises(ValueError):nominal_rows(plan,rows[6:])
    for row in rows:row['execution_repeats']=[0]
    with pytest.raises(ValueError):nominal_rows(plan,rows)


def test_32_task_nominal_driver_synthetic_backend_no_scoring(tmp_path,monkeypatch):
    from benchmarks.LIBERO import continuous_nominal as module
    plan,rows=inputs();schedule=tmp_path/'schedule.json';schedule.write_text(json.dumps(rows))
    sim=tmp_path/'sim.json';sim.write_text('{}')
    parent=tmp_path/'paired.json';parent.write_text('{}')
    cfg=tmp_path/'nominal.json';cfg.write_text(json.dumps(dict(authorization='EXPLICIT_NOMINAL_EVALUATION_ONLY',
        paired_config=str(parent),paired_config_sha256=file_digest(parent),execution_cap=1760)))
    paired=dict(schedule=str(schedule),cohorts={k:dict(sim_config=str(sim),execution_noise='synthetic') for k in ('A','B')})
    monkeypatch.setattr(module,'validate_inputs',lambda p:(paired,plan,{}))
    class Runtime:
        def __init__(self,*a):self.calls=0
        def infer(self,example,noise,identity,capture=True):
            assert capture is False;self.calls+=1;return dict(actions=np.zeros((10,7)))
    class Bank:
        def __init__(self,*a,**k):pass
        def get(self,state_key,repeat,replan):return dict(noise=np.zeros((10,7),np.float32),noise_sha256='0'*64)
    monkeypatch.setattr('benchmarks.LIBERO.search_runtime.create_policy_runtime',Runtime)
    monkeypatch.setattr('AlphaBrain.research.dsol.data.flow_noise.ExplicitFlowNoiseBank',Bank)
    monkeypatch.setattr('benchmarks.LIBERO.search_live.check_simulator_dependencies',lambda c:None)
    commands=[]
    @contextmanager
    def simulator(*args):
        def request(message):
            kind=message['kind'];commands.append(kind)
            example=dict(state=np.zeros(8),lang='synthetic',image=[np.zeros((2,2,3),np.uint8)]*2,
                camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4))
            if kind=='continuous_info':return dict(reference=np.eye(4))
            if kind=='step':return dict(example=example,executed=1,success=True)
            assert kind in ('query_pose','begin_execution')
            return dict(example=example,receipt=dict(camera_to_world=np.eye(4)))
        yield request
    monkeypatch.setattr('benchmarks.LIBERO.search_live.simulator',simulator)
    run(cfg,tmp_path/'result',execute=True)
    result=json.loads((tmp_path/'result/status.json').read_text())
    assert result['completed_tasks']==32 and result['calls']=={'execution':32}
    assert result['score_calls']==0 and result['search_queries']==0
    assert commands.count('step')==32 and 'move_pose' not in commands

    calls=0
    for index in range(16,24):
        out=tmp_path/f'block-{index}';run(cfg,out,execute=True,block_index=index)
        status=json.loads((out/'status.json').read_text());assert status['completed_tasks']==4
        calls+=status['calls']['execution']
    assert calls==32
