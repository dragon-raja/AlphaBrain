from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.core_pilot import validate_protocol, validate_evaluation, SCHEMA
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.data.score_prefix import contract


def protocol():
    cells={}
    for v in ('U0','U1','U2','U3'):
        for s in (0,1):
            cells[f'pi05-{v}-seed-{s}']=dict(family='pi05',prefix=10,seed=s,
                score_contract=contract('pi05',10),native_policy=None,feature_producer=None,
                core_spec=core_spec(v),observation_scale=0.16535300100378683,
                evaluation_schedules={'dev':dict(purpose='development',version=16,method='ppo')})
    return dict(schema=SCHEMA,updates=16,episodes_per_update=32,seeds=[0,1],
                maximum_model_calls=120000,query_cap=8,query_cost=.01,cells=cells)


@pytest.mark.parametrize('field,value',[('updates',32),('maximum_model_calls',120001),('seeds',[0,1,2]),('query_cost',0.)])
def test_scope_cannot_silently_extend(field,value):
    p=protocol();validate_protocol(p);p[field]=value
    with pytest.raises(ValueError):validate_protocol(p)


def test_fixed_score_features_core_and_dev_only():
    p=protocol();validate_protocol(p)
    for change in ('core','feature','score','eval'):
        q=deepcopy(p);c=q['cells']['pi05-U0-seed-0']
        if change=='core':c['core_spec']=core_spec('U1')
        if change=='feature':c['feature_producer']={'new':'tap'}
        if change=='score':c['score_contract']=contract('pi05',3)
        if change=='eval':c['evaluation_schedules']['dev']['purpose']='final'
        with pytest.raises(ValueError):validate_protocol(q)
    c=dict(evaluation_version=16,schedule='dev',stay=False)
    cell=p['cells']['pi05-U0-seed-0']
    row=dict(query_cap=8,execution_repeats=[0,1],settle_steps=10)
    state=dict(split='validation',initialization=dict(settle_steps=10))
    validate_evaluation(c,cell,row,state)
    with pytest.raises(ValueError):validate_evaluation(c,cell,row,dict(state,split='test_unseen_task'))


def test_combinations_require_distinct_explicit_stage():
    from AlphaBrain.research.dsol.data.core_pilot import STAGES
    p=protocol(); original=p['cells']; p['cells']={}
    for variant in STAGES['combinations']:
        for seed in p['seeds']:
            c=deepcopy(original[f'pi05-U0-seed-{seed}'])
            c['core_spec']=core_spec(variant)
            p['cells'][f'pi05-{variant}-seed-{seed}']=c
    with pytest.raises(ValueError):validate_protocol(p)
    p['design_stage']='combinations';validate_protocol(p)
    p['cells'].update(original)
    with pytest.raises(ValueError):validate_protocol(p)


@pytest.mark.parametrize('stage',['unknown',None,''])
def test_unknown_design_stage_rejected(stage):
    p=protocol();p['design_stage']=stage
    with pytest.raises(ValueError):validate_protocol(p)


def test_task_factor_scope_keeps_condition_and_budget_explicit():
    from tests.dsol_paper1.helpers.task_condition import task_spec
    p=protocol();old=p['cells'];binding=dict(spec=task_spec(),cache='/synthetic/text.json',cache_sha256='e'*64)
    p.update(design_stage='task_condition',maximum_model_calls=90000,selected_base='U2',task_variants=['U0','U123','U2'],task_condition=binding,cells={})
    for v in p['task_variants']:
        for seed in p['seeds']:
            c=deepcopy(old[f'pi05-U0-seed-{seed}']);c.update(core_spec=core_spec(v),task_condition=deepcopy(binding),model_hashes={'suite':'a'*64})
            p['cells'][f'pi05-{v}T-seed-{seed}']=c
    validate_protocol(p)
    q=deepcopy(p);q['maximum_model_calls']=120000
    with pytest.raises(ValueError):validate_protocol(q)
    q=deepcopy(p);q['selected_base']='U1'
    with pytest.raises(ValueError):validate_protocol(q)
    q=deepcopy(p);q['cells']['pi05-U0T-seed-0']['task_condition']['spec']['family']='groot_n15'
    with pytest.raises(ValueError):validate_protocol(q)


def replication_protocol():
    from tests.dsol_paper1.helpers.task_condition import task_spec
    from AlphaBrain.research.dsol.data.core_pilot import STAGES
    p=protocol();base=deepcopy(p['cells']['pi05-U0-seed-0'])
    binding=dict(spec=task_spec(),cache='/synthetic/task.json',cache_sha256='e'*64)
    p.update(design_stage='seed_replication',seeds=[2],maximum_model_calls=170000,
             selected_base='U12',task_variants=['U0','U123','U12'],task_condition=binding,cells={})
    for v in (*STAGES['single_factors'],*STAGES['combinations'],'U0T','U123T','U12T'):
        c=deepcopy(base);c.update(seed=2,core_spec=core_spec(v.removesuffix('T')),model_hashes={'suite':'a'*64})
        if v.endswith('T'):c['task_condition']=deepcopy(binding)
        p['cells'][f'pi05-{v}-seed-2']=c
    return p


def test_replication_closed_budget_seed_and_task_matrix():
    p=replication_protocol();validate_protocol(p)
    for change in ('seed','budget','missing','task_on_base','missing_task'):
        q=deepcopy(p)
        if change=='seed':q['seeds']=[2,3]
        if change=='budget':q['maximum_model_calls']=170001
        if change=='missing':q['cells'].pop('pi05-U3-seed-2')
        if change=='task_on_base':q['cells']['pi05-U0-seed-2']['task_condition']=q['task_condition']
        if change=='missing_task':q['cells']['pi05-U0T-seed-2']['task_condition']=None
        with pytest.raises((ValueError,TypeError)):validate_protocol(q)


def test_admission_identity_separates_task_and_provenance_not_seed():
    from benchmarks.LIBERO.prefix_campaign import admission_identity
    p=replication_protocol();base=p['cells']['pi05-U0-seed-2'];task=p['cells']['pi05-U0T-seed-2']
    assert admission_identity(base)!=admission_identity(task)
    assert admission_identity(base)==admission_identity(dict(base,seed=9))
    assert admission_identity(base)!=admission_identity(dict(base,model_hashes={'suite':'b'*64}))


def test_replication_admission_enumerates_seed_two_not_missing_seed_zero():
    from benchmarks.LIBERO.prefix_campaign import PrefixCampaign
    co=PrefixCampaign.__new__(PrefixCampaign);co.protocol=replication_protocol();seen=[]
    co.admit_cell=lambda name,family,host:seen.append((name,family,host))
    co.admission('pi05','synthetic-host')
    assert len(seen)==11 and {x[0] for x in seen}==set(co.protocol['cells'])
