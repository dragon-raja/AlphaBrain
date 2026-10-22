"""Actual CPU optimization on synthetic searches; never scientific training."""
from contextlib import contextmanager
import json
import numpy as np
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO, ContinuousPPOConfig
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
from AlphaBrain.research.dsol.data.continuous_training_store import write_json
from AlphaBrain.research.dsol.data.continuous_training_schedule import balanced_schedule
from tests.dsol_paper1.helpers.local_search import synthetic_session


@contextmanager
def backend(row, output):
    old, source = synthetic_session()
    core = ContinuousSearch(old.mount, source, initial_angles=[0,0],
        physical_state_id='synthetic', wrist=np.ones(4), score_scale=1., seed=row['episode'])
    yield ContinuousSearchGym(4), {'session':core}


def trainer(path, *, resume=False, updates=2, source=backend, binding=None):
    torch.set_num_threads(1);torch.manual_seed(71)
    policy = ContinuousActorCritic(4,width=16)
    schedule = balanced_schedule([dict(episode=i,initial=i//(updates*2)) for i in range(updates*32)],
                                 list(range(16)),updates=updates)
    return ContinuousPPO(policy,ContinuousPPOConfig(updates=updates),schedule,source,path,
        binding=binding or {'source_kind':'SYNTHETIC_CPU_NO_MODEL'},resume=resume)


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('pause',[7,32,39])
def test_full_training_and_exact_resume(tmp_path,pause):
    with trainer(tmp_path/'whole') as full:
        initial={k:v.clone() for k,v in full.policy.state_dict().items()}
        result=full.run()
        assert result['episodes']==64 and result['updates']==2
        assert 64<=result['queries']<=512
        assert any(not torch.equal(v,initial[k]) for k,v in full.policy.state_dict().items())
        expected=full.store.load(json.loads((full.store.root/'latest.json').read_text()))
        for record in full.store.episodes():
            episode=full.store.load(record)['episode']
            assert episode['policy_update']==record['index']//32
        stats=json.loads((full.store.root/'updates/001.json').read_text())
        assert len(stats['minibatch_sizes'])>=4 and sum(stats['minibatch_sizes'])==4*stats['transitions']
        assert all(np.isfinite(v) for v in stats['means'].values())
        assert any(s<64 for s in stats['minibatch_sizes'])
    with trainer(tmp_path/'resumed') as first:
        for i in range(pause):
            if len(first.pending)==32:first.optimize()
            first.collect_one()
    with trainer(tmp_path/'resumed',resume=True) as second:
        assert second.cursor==pause
        actual=second.run()
        equal(result,actual)
        equal(expected,second.store.load(json.loads((second.store.root/'latest.json').read_text())))
        for index in range(64):
            a=torch.load(tmp_path/'whole/episodes'/f'{index:04d}.pt',weights_only=True)
            b=torch.load(tmp_path/'resumed/episodes'/f'{index:04d}.pt',weights_only=True)
            equal(a,b)


def test_no_optimization_until_complete_batch_and_single_writer(tmp_path):
    with trainer(tmp_path/'run') as t:
        before={k:v.clone() for k,v in t.policy.state_dict().items()}
        t.collect_one()
        equal(before,t.policy.state_dict())
        with pytest.raises(ValueError,match='complete episode'):t.optimize()
        with pytest.raises(ValueError,match='Another trainer'):trainer(tmp_path/'run',resume=True)
    with pytest.raises(ValueError,match='Changed protocol'):
        trainer(tmp_path/'run',resume=True,binding={'source_kind':'CHANGED'})


def test_failure_stops_without_extra_query_or_update(tmp_path):
    calls=[]
    @contextmanager
    def failing(row,output):
        calls.append(row['episode'])
        raise RuntimeError('injected source failure')
        yield
    with trainer(tmp_path/'run',source=failing) as t:
        with pytest.raises(RuntimeError):t.collect_one()
        assert t.cursor==t.update==0 and calls==[0]
        with pytest.raises(ValueError,match='no automatic replay'):t.collect_one()
    with pytest.raises(ValueError,match='no automatic replay'):trainer(tmp_path/'run',resume=True)
    assert calls==[0]


def test_hash_tampering_and_uncommitted_episode_fail_closed(tmp_path):
    with trainer(tmp_path/'hash') as t:t.collect_one()
    with (tmp_path/'hash/episodes/0000.pt').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='identity mismatch'):trainer(tmp_path/'hash',resume=True)
    with trainer(tmp_path/'interrupted') as t:t.collect_one()
    (tmp_path/'interrupted/episodes/0001.pt').write_bytes(b'incomplete')
    with pytest.raises(ValueError,match='Uncommitted'):trainer(tmp_path/'interrupted',resume=True)


def test_retention_and_update_boundary_resume(tmp_path):
    with trainer(tmp_path/'run',updates=3) as t:
        assert t.run(through_update=1)['status']=='PAUSED_AT_UPDATE_BOUNDARY'
    with trainer(tmp_path/'run',updates=3,resume=True) as t:
        assert t.update==1 and t.cursor==32 and not t.pending
        t.run()
        assert sorted(p.name for p in (t.store.root/'checkpoints').glob('*.pt'))==[
            'update-000.pt','update-002.pt','update-003.pt']
        assert len(t.store.episodes())==96
        assert (t.store.root/'updates/003-retention.json').exists()


def test_changed_schedule_and_counter_detected(tmp_path):
    with trainer(tmp_path/'run') as t:t.collect_one()
    receipt=tmp_path/'run/episodes/0000.json'
    obj=json.loads(receipt.read_text());obj['metrics']['initial_identity']['initial']=99
    write_json(receipt,obj)
    with pytest.raises(ValueError,match='identity/accounting'):trainer(tmp_path/'run',resume=True)


def test_backend_rng_does_not_change_actor_samples(tmp_path):
    import random
    @contextmanager
    def noisy(row,output):
        torch.rand(50);np.random.rand(17);random.random()
        with backend(row,output) as pair:yield pair
        torch.rand(23);np.random.rand(3);random.random()
    with trainer(tmp_path/'reference') as t:
        t.collect_one();expected=t.store.load(t.store.episodes()[0])
    with trainer(tmp_path/'noisy',source=noisy) as t:
        t.collect_one();equal(expected,t.store.load(t.store.episodes()[0]))


def test_failed_commit_leaves_no_resumable_fake_success(tmp_path,monkeypatch):
    with trainer(tmp_path/'run') as t:
        def fail(*args):raise OSError('injected disk full')
        monkeypatch.setattr(t.store,'save_episode',fail)
        with pytest.raises(OSError):t.collect_one()
        assert not t.pending and t.cursor==t.update==0
    with pytest.raises(ValueError,match='no automatic replay'):trainer(tmp_path/'run',resume=True)


def test_frozen_512_episode_schedule_completes_without_extra_sampling(tmp_path):
    with trainer(tmp_path/'run',updates=16) as t:
        result=t.run()
        assert result['updates']==16 and result['episodes']==512
        assert result['scoring_trajectories']==2*result['queries']<=8192
        assert len(t.store.episodes())==512
        with pytest.raises(ValueError,match='exhausted'):t.collect_one()
        with pytest.raises(ValueError,match='complete episode'):t.optimize()


def test_backend_cleanup_failure_never_commits_episode(tmp_path):
    @contextmanager
    def bad_cleanup(row,output):
        with backend(row,output) as pair:yield pair
        raise RuntimeError('cleanup failed')
    with trainer(tmp_path/'run',source=bad_cleanup) as t:
        with pytest.raises(RuntimeError,match='cleanup failed'):t.collect_one()
        assert not t.store.episodes() and t.cursor==0
    with pytest.raises(ValueError,match='no automatic replay'):trainer(tmp_path/'run',resume=True)


def test_legacy_training_binding_cannot_resume_with_corrected_objective(tmp_path):
    with trainer(tmp_path/'run') as t:t.collect_one()
    path=tmp_path/'run/binding.json';binding=json.loads(path.read_text())
    binding.pop('exploration_rule');binding['engine']='continuous_ppo_complete_episode_v1'
    write_json(path,binding)
    with pytest.raises(ValueError,match='Changed protocol'):trainer(tmp_path/'run',resume=True)
