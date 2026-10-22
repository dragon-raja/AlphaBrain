"""Real four-process CPU sampling, synthetic query backend, no VLA/simulator."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing as mp
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.selectors.continuous_parallel import seeded_episode,ingest,validate_rollout,RNG_RULE
from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
from AlphaBrain.research.dsol.data.continuous_training_store import pack
from AlphaBrain.research.dsol.execution.manifest import file_digest
from tests.dsol_paper1.helpers.local_search import synthetic_session


@contextmanager
def backend(row,output):
    old,source=synthetic_session()
    core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
        wrist=np.ones(4),score_scale=1.,seed=row['proposal_seed'])
    yield ContinuousSearchGym(4),{'session':core}


def collect(args):
    state,row=args;torch.set_num_threads(1)
    p=ContinuousActorCritic(4,width=16);p.load_state_dict(state)
    episode,metrics=seeded_episode(p,backend,row,None,version=0,seed=0,source_kind='SYNTHETIC_CPU')
    metrics.pop('elapsed_seconds')
    return pack(dict(episode=episode,metrics=metrics))


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_serial_and_four_processes_exact_then_one_update_and_resume(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(71);p=ContinuousActorCritic(4,width=16)
    rows=[dict(episode=i,source_episode=31-i,initial=i%16,proposal_seed=100+i) for i in range(32)]
    args=[(p.state_dict(),r) for r in rows]
    serial=[collect(a) for a in args]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        parallel=list(pool.map(collect,args[::-1]))[::-1]
    for a,b in zip(serial,parallel):equal(a,b)
    records={}
    for i,payload in enumerate(parallel):
        path=tmp_path/f'{i}.pt';torch.save(payload,path);records[i]=dict(path=str(path),sha256=file_digest(path))
    config=ContinuousPPOConfig(updates=1)
    binding=dict(source_kind='SYNTHETIC_CPU',sampling_rng=RNG_RULE)
    with ContinuousPPO(p,config,rows,None,tmp_path/'training',binding=binding) as trainer:
        with pytest.raises(ValueError,match='barrier'):ingest(trainer,{k:v for k,v in records.items() if k!=0})
        wrong=pack(dict(episode=serial[0]['episode'],metrics=serial[0]['metrics']))
        wrong['episode']['policy_update']=1
        with pytest.raises(ValueError,match='version'):validate_rollout(p,wrong,rows[0],0)
        # Explicit recovery of a fully collected batch must not trigger any backend.
        ingest(trainer,records);assert trainer.cursor==32 and trainer.update==0
    restored=ContinuousActorCritic(4,width=16)
    with ContinuousPPO(restored,config,rows,None,tmp_path/'training',binding=binding,resume=True) as trainer:
        assert len(trainer.pending)==32
        ingest(trainer,records);assert trainer.cursor==32 and len(trainer.pending)==32
        old={k:v.clone() for k,v in restored.state_dict().items()};trainer.optimize()
        assert trainer.update==1 and any(not torch.equal(old[k],v) for k,v in restored.state_dict().items())
        with pytest.raises(ValueError):trainer.optimize()


def test_collection_failure_restores_rng_and_never_produces_episode():
    @contextmanager
    def fail(row,path):
        torch.rand(50);raise RuntimeError('backend failed');yield
    torch.manual_seed(8);p=ContinuousActorCritic(4,width=16);old=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):seeded_episode(p,fail,dict(episode=0,proposal_seed=0),None,version=0,seed=0,source_kind='CPU')
    assert torch.equal(old,torch.get_rng_state())


def test_capacity_replicas_preserve_episode_rng_without_reusing_output_identity():
    from AlphaBrain.research.dsol.selectors.continuous_parallel import episode_seed
    for row in [dict(episode=3,proposal_seed=8),dict(episode=3,source_episode=91,proposal_seed=8)]:
        replica=dict(row,episode=1000,source_episode=row.get('source_episode',row['episode']))
        assert replica['episode']!=row['episode']
        assert episode_seed(0,replica)==episode_seed(0,row)
