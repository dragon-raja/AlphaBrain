"""Explicit stage identity and exact CPU optimizer continuation, no real models."""
from contextlib import contextmanager
from copy import deepcopy
import json
import numpy as np
import pytest
from benchmarks.LIBERO.confirmed_pilot import protocol_limits
from AlphaBrain.research.dsol.execution.manifest import file_digest


def test_continuation_does_not_relax_the_original_four_update_protocol(tmp_path):
    def save(name,obj):
        p=tmp_path/name;p.write_text(json.dumps(obj));return str(p)
    schedule=save('old-schedule.json',list(range(128)))
    old=dict(schema='CONFIRMED_CHANGE_SMALL_PILOT_PROTOCOL_V1',maximum_updates=4,
             training_schedule=schedule,query_cap=8,query_cost=.01,maximum_new_candidate_queries=4,
             terminal_confirmation_queries=2,terminal_signed_gain=True,seeds=[0],
             models=['pi05','groot_n15','fastwam'],unchanged=['networks','STOP'])
    parent=save('old.json',old)
    new=dict(old,schema='CONFIRMED_CHANGE_CONTINUATION_V1',maximum_updates=16,
             initial_updates=4,episodes_per_update=32,automatic_retry=False,automatic_mainline=False,
             parent_protocol=dict(path=parent,sha256=file_digest(parent)),
             parent_schedule_sha256=file_digest(schedule),training_schedule=save('schedule.json',list(range(512))))
    path=save('new.json',new)
    assert protocol_limits(parent)[1:]==(0,4,'G-matched-v4')
    assert protocol_limits(path)[1:]==(4,16,'G-matched-v16')
    for key,value in [('query_cost',.001),('maximum_updates',64),('automatic_retry',True),('initial_updates',3)]:
        changed=dict(new);changed[key]=value;save('bad.json',changed)
        with pytest.raises(ValueError):protocol_limits(tmp_path/'bad.json')
    save('schedule.json',list(reversed(range(512))))
    with pytest.raises(ValueError,match='prefix'):protocol_limits(path)


def test_forked_stage_matches_uninterrupted_optimizer_and_preserves_source(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('gymnasium')
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig,training_binding
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
    from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
    from AlphaBrain.research.dsol.data.continuous_training_store import fork_complete_stage
    from tests.dsol_paper1.helpers.local_search import synthetic_session
    torch.set_num_threads(1)
    @contextmanager
    def backend(row,output):
        old,source=synthetic_session()
        core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
                              wrist=np.ones(4),score_scale=1.,seed=row['episode'])
        yield ContinuousSearchGym(4),{'session':core}
    rows=[dict(episode=i,initial=i//2) for i in range(64)]
    def policy():
        torch.manual_seed(71);return ContinuousActorCritic(4,width=16)
    binding=dict(source_kind='SYNTHETIC_CPU_NO_MODEL',config_sha256='old')
    with ContinuousPPO(policy(),ContinuousPPOConfig(updates=2),rows,backend,tmp_path/'whole',binding=binding) as full:
        full.run();expected=full.store.load(json.loads((full.store.root/'latest.json').read_text()))
    source=tmp_path/'old'
    with ContinuousPPO(policy(),ContinuousPPOConfig(updates=1),rows[:32],backend,source,binding=binding) as first:
        first.run()
    files={str(p.relative_to(source)):file_digest(p) for name in ('episodes','checkpoints','updates')
           for p in (source/name).iterdir() if p.is_file()}
    files.update({name:file_digest(source/name) for name in ('binding.json','latest.json')})
    newbinding=training_binding(policy(),ContinuousPPOConfig(updates=2),rows,dict(binding,config_sha256='new'))
    def fork(dest,**overrides):
        values=dict(previous_schedule=rows[:32],schedule=rows,source_files=files,expected_update=1);values.update(overrides)
        fork_complete_stage(source,dest,newbinding,**values)
    for fault in ('schedule','manifest','reward','optimizer','inflight'):
        dest=tmp_path/fault
        if fault=='schedule':
            with pytest.raises(ValueError):fork(dest,schedule=list(reversed(rows)))
        elif fault=='manifest':
            with pytest.raises(ValueError):fork(dest,source_files={})
        elif fault in ('reward','optimizer'):
            bad=deepcopy(newbinding)
            if fault=='reward':bad['reward_config']={'changed':True}
            else:bad['config']['learning_rate']=.1
            with pytest.raises(ValueError):fork_complete_stage(source,dest,bad,previous_schedule=rows[:32],schedule=rows,source_files=files,expected_update=1)
        else:
            (source/'inflight.json').write_text('{}')
            with pytest.raises(ValueError,match='Ambiguous'):fork(dest)
            (source/'inflight.json').unlink()
        assert not dest.exists()
    fork(tmp_path/'extended')
    with pytest.raises(ValueError,match='separate'):fork(tmp_path/'extended')
    with ContinuousPPO(policy(),ContinuousPPOConfig(updates=2),rows,backend,tmp_path/'extended',
                       binding=dict(binding,config_sha256='new'),resume=True) as actual:
        assert actual.update==1 and actual.cursor==32
        actual.run();got=actual.store.load(json.loads((actual.store.root/'latest.json').read_text()))
    def equal(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b
    equal(expected,got)
    assert all(file_digest(source/name)==stamp for name,stamp in files.items())
