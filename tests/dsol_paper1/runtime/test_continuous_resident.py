from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')

from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.rollout_budget import RolloutBudget
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.selectors.continuous_parallel import validate_rollout
from benchmarks.LIBERO.continuous_parallel import worker
from tests.dsol_paper1.helpers.local_search import synthetic_session


def test_actual_worker_reuses_backend_across_versions_without_changing_rollouts(tmp_path,monkeypatch):
    torch.set_num_threads(2);torch.manual_seed(31)
    created=[]
    class FakeLive:
        def __init__(self,cfg,plan,root,global_budget=None,runtime=None):
            if runtime is None:
                runtime=SimpleNamespace();created.append(runtime)
            self.runtime=runtime;self.root=root;self.timings=[];self.budget=global_budget
        @contextmanager
        def __call__(self,row,output):
            old,source=synthetic_session(4608);original=source.query
            def query():
                v=original();budget,owner,version,prefix=self.budget
                for member in (0,1):
                    budget.attempt(f'{prefix}:{row["episode"]}',owner,version,source.calls,member)
                return v
            source.query=query
            core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
                wrist=np.ones(4608),score_scale=1.,seed=row['proposal_seed'])
            yield ContinuousSearchGym(4608),{'session':core}
    monkeypatch.setattr('benchmarks.LIBERO.continuous_training.LiveEpisodes',FakeLive)
    plan=tmp_path/'design.json';plan.write_text(json.dumps(dict(training=dict(seed=0))))
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(dict(authorization='EXPLICIT_ADMISSION_ONLY',
        design=str(plan),source_kind='SYNTHETIC_CPU_ONLY')))
    policy=ContinuousActorCritic(4608).eval()
    row=dict(episode=0,source_episode=0,initial=0,proposal_seed=9)
    budget=RolloutBudget.create(tmp_path/'budget',48);shared={};records=[]
    for label,version,resident in [('cold',0,None),('resident-first',0,shared),('resident-next',1,shared)]:
        snapshot=tmp_path/f'{label}.pt';torch.save(dict(version=version,state_dict=policy.state_dict()),snapshot)
        digest=file_digest(snapshot);owner='collector'
        budget.reserve([dict(key=f'{label}:0',owner=owner,version=digest,cap=16)])
        req=dict(authorization='EXPLICIT_COLLECTION_ONLY',config=str(cfg),policy=str(snapshot),
            policy_sha256=digest,version=version,owner=owner,prefix=label,rows=[row],
            query_root=str(tmp_path/f'{label}-queries'),budget_root=str(budget.root))
        path=tmp_path/f'{label}.json';path.write_text(json.dumps(req));out=tmp_path/label
        result=worker(path,out,execute=True,shared_runtime=resident)
        value=torch.load(out/'episode-0000.pt',map_location='cpu',weights_only=True)
        validate_rollout(policy,value,row,version)
        records.append(value)
    assert len(created)==2 and shared['runtime'] is created[1]
    assert budget.inspect()['reserved']==0
    for value in records[1:]:
        assert value['metrics']['queries']==records[0]['metrics']['queries']
        for a,b in zip(records[0]['episode']['transitions'],value['episode']['transitions']):
            assert a['log_prob']==b['log_prob'] and a['value']==b['value']
            torch.testing.assert_close(a['action']['delta_deg'],b['action']['delta_deg'],rtol=0,atol=0)


def test_resident_entry_refuses_execution_without_authorization(tmp_path):
    from benchmarks.LIBERO.continuous_resident import serve
    cfg=tmp_path/'binding.json';cfg.write_text('{}')
    with pytest.raises(ValueError,match='not authorized'):serve(cfg,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('fail',[False,True])
def test_four_real_resident_processes_queue_barrier_and_failure_cleanup(tmp_path,monkeypatch,fail):
    import sys
    from benchmarks.LIBERO import continuous_resident as module
    from benchmarks.Robocasa365.probe_queue import process_identity
    from AlphaBrain.research.dsol.execution.queue import load
    class CPUBackend(module.ResidentBackend):
        def admit(self,slot):return True
        def gpu_available(self,slot):return True
        def worker_command(self,task):
            code='''import sys,json
from pathlib import Path
from benchmarks.LIBERO import continuous_resident as m
def fake(path,output,**kwargs):
 r=json.loads(Path(path).read_text())
 if r.get("inject_failure"):raise RuntimeError("injected worker failure")
 p=Path(output);p.mkdir(parents=True)
 (p/"status.json").write_text(json.dumps(dict(status="COMPLETE",version=r["version"])))
m.worker=fake
m.serve(sys.argv[1],sys.argv[2],execute=True)
'''
            return [sys.executable,'-c',code,task['payload']['binding'],task['output']]
    monkeypatch.setattr(module,'ResidentBackend',CPUBackend)
    config=tmp_path/'cfg.json';config.write_text('{}')
    runtime=dict(host='cpu',source=str(Path(__file__).resolve().parents[3]),environment={},
        source_artifacts=[],slots=[dict(host='cpu',id=str(i),gpu_id=str(i)) for i in range(4)],
        batch_timeout_seconds=30)
    pool=module.ResidentPool(str(config),runtime,tmp_path/'pool',tmp_path/'control')
    try:
        handles=None
        for version in range(2):
            requests=[]
            for i in range(4):
                q=tmp_path/f'request-{version}-{i}.json'
                q.write_text(json.dumps(dict(config=str(config),version=version,owner=f'resident-collector-{i}',
                    resident_output=str(tmp_path/f'out-{version}-{i}'),inject_failure=fail and version==1 and i==0)))
                requests.append(str(q))
            if fail and version==1:
                with pytest.raises(RuntimeError,match='queue failed'):pool.collect(requests)
                break
            results=pool.collect(requests);assert len(results)==4
            _,state=load(tmp_path/'control')
            current=[r['handle'] for r in state['tasks'].values()]
            if handles is not None:assert [h['pid'] for h in handles]==[h['pid'] for h in current]
            handles=current
    finally:pool.close(failed=fail)
    _,state=load(tmp_path/'control')
    assert state['status']==('FAILED' if fail else 'COMPLETE')
    assert all(process_identity(h['pid']) is None for h in handles)
