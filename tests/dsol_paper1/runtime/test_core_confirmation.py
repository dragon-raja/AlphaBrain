"""Closed evidence scope, historical compatibility, and actual scheduling barriers."""
from threading import RLock
import pytest
from AlphaBrain.research.dsol.data.core_pilot import validate_protocol
from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.data.score_prefix import contract


def protocol():
    p=dict(schema='UNIFIED_CORE_PILOT_V1',design_stage='core_confirmation',updates=128,start_update=96,
           episodes_per_update=32,seeds=[0,1,2],maximum_model_calls=300000,query_cap=8,query_cost=.01,cells={})
    for variant in ('U12','U3','Q4','Q6','Q16'):
        for seed in range(3):
            deploy=variant.startswith('Q');core='U12' if deploy else variant
            name=f'pi05-{core}'+('-'+variant if deploy else '')+f'-seed-{seed}'
            c=dict(core_spec=core_spec(core),family='pi05',prefix=10,seed=seed,
                   score_contract=contract('pi05',10),native_policy=None,observation_scale=.16535300100378683,
                   evaluation_only=deploy,training_schedule='train',
                   evaluation_schedules={str(v):dict(purpose='development',version=v,method='ppo')
                       for v in ([96] if deploy else [0,4,8,112,128])})
            if deploy:c['mechanism']=mechanism_spec(variant)
            else:c['continuation']=dict(update=96,source_training='old',checkpoint_sha256='a'*64)
            p['cells'][name]=c
    return p


@pytest.mark.parametrize('field,value',[('updates',144),('start_update',16),('seeds',[0,1,2,3]),
    ('maximum_model_calls',300001),('query_cap',16),('query_cost',0)])
def test_scope_cannot_expand(field,value):
    p=protocol();validate_protocol(p);p[field]=value
    with pytest.raises(ValueError):validate_protocol(p)


def test_q16_cannot_train_or_change_score():
    p=protocol();p['cells']['pi05-U12-Q16-seed-0']['evaluation_only']=False
    with pytest.raises(ValueError):validate_protocol(p)
    p=protocol();p['cells']['pi05-U12-seed-0']['prefix']=3
    with pytest.raises(ValueError):validate_protocol(p)


@pytest.mark.parametrize('asynchronous',[False,True])
def test_continuation_queues_early_and_tail_once_without_replaying_training(tmp_path,monkeypatch,asynchronous):
    from benchmarks.LIBERO import prefix_campaign as module
    co=module.PrefixCampaign.__new__(module.PrefixCampaign)
    co.protocol=protocol();co.cfg={'async_development':asynchronous};co.root=tmp_path
    co.lock=RLock();co.failures={};calls=[];updates=[];co.status=lambda *a,**k:None
    def execute(n,kind,schedule,**kwargs):
        calls.append((kind,kwargs['version']));return {},'campaign','phase'
    co.execute=execute;co.cpu_update=lambda n,v,b:updates.append(v)
    actor=tmp_path/'actor';actor.write_bytes(b'actor');co.publication=lambda n,v:actor
    monkeypatch.setattr(module,'collection_records',lambda *a:{})
    co.cell('pi05-U12-seed-0')
    assert [v for k,v in calls if k=='training']==list(range(96,128))
    assert sorted(v for k,v in calls if k=='evaluation')==[0,4,8,112,128]
    assert updates==list(range(97,129))
    calls.clear();co.cell('pi05-U12-Q16-seed-0')
    assert calls==[('evaluation',96)] and not co.failures


def test_query_accounting_covers_real_long_search_and_admission(tmp_path):
    from benchmarks.LIBERO.prefix_pilot import limits
    from AlphaBrain.research.dsol.data.artifacts import write
    f=tmp_path/'plan.json';write(f,{'evaluation':{'horizons':{'spatial':220}}})
    sim={'pilot_execution_plan':str(f)}
    for cap in (4,6,8,16):
        rows=[{'suite':'spatial','query_cap':cap}]
        budget=limits(rows,sim,'evaluation','pi05',query_cap=cap)
        assert budget==dict(episodes=2,score=2*cap,paired_execution=88,nominal_execution=0)
        admission=limits(rows,sim,'admission','pi05',collector_smoke=True,query_cap=cap)
        assert admission['score']==88+3+2*cap
