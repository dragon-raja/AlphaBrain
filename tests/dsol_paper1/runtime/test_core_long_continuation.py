from copy import deepcopy
from threading import RLock

import pytest

from AlphaBrain.research.dsol.data.core_pilot import validate_protocol
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.data.score_prefix import contract


def protocol():
    p=dict(schema='UNIFIED_CORE_PILOT_V1',design_stage='long_continuation',updates=96,start_update=16,
           episodes_per_update=32,seeds=[0,1,2],maximum_model_calls=600000,query_cap=8,query_cost=.01,cells={})
    for v in ('U12','U3'):
        for s in range(3):
            p['cells'][f'pi05-{v}-seed-{s}']=dict(core_spec=core_spec(v),seed=s,family='pi05',prefix=10,
                score_contract=contract('pi05',10),native_policy=None,observation_scale=.16535300100378683,
                continuation=dict(update=16,source_training='old',checkpoint_sha256='a'*64),
                evaluation_schedules={str(i):dict(purpose='development',version=i,method='ppo') for i in (32,48,64,80,96)})
    return p


@pytest.mark.parametrize('field,value',[('updates',128),('start_update',0),('seeds',[0,1,2,3]),('maximum_model_calls',600001),('query_cost',0)])
def test_long_scope_rejects_silent_extension(field,value):
    p=protocol();validate_protocol(p);p[field]=value
    with pytest.raises(ValueError):validate_protocol(p)


def test_long_scope_requires_complete_matched_curves():
    p=protocol();del p['cells']['pi05-U3-seed-2']['evaluation_schedules']['48']
    with pytest.raises(ValueError):validate_protocol(p)


@pytest.mark.parametrize('asynchronous',[False,True])
def test_campaign_continues_without_replaying_prefix_and_evaluates_all_endpoints(tmp_path,monkeypatch,asynchronous):
    import benchmarks.LIBERO.prefix_campaign as module
    co=module.PrefixCampaign.__new__(module.PrefixCampaign)
    co.protocol=protocol();co.cfg={'async_development':asynchronous};co.root=tmp_path
    co.lock=RLock();co.failures={};name='pi05-U12-seed-0';co.protocol['cells'][name]['training_schedule']='train'
    calls=[];updates=[];co.status=lambda *a,**k:None
    def execute(n,kind,schedule,**kwargs):
        calls.append((kind,kwargs['version']));return {},'campaign','phase'
    co.execute=execute;co.cpu_update=lambda n,v,b:updates.append(v)
    actor=tmp_path/'actor';actor.write_bytes(b'actor');co.publication=lambda n,v:actor
    monkeypatch.setattr(module,'collection_records',lambda *a:{})
    co.cell(name)
    assert [v for kind,v in calls if kind=='training']==list(range(16,96))
    assert sorted(v for kind,v in calls if kind=='evaluation')==[32,48,64,80,96]
    assert updates==list(range(17,97))
    assert not co.failures
