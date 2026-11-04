"""Scientific population, split and separate model-budget regression checks."""
from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.native_baseline import HORIZONS
from AlphaBrain.research.dsol.data.native_view_panel import views
from AlphaBrain.research.dsol.data.target_mainline import OperationSpec,compile_mainline


def inputs():
    tasks=[(suite,f'task-{i*10+j}') for i,suite in enumerate(HORIZONS) for j in range(10)]
    training=[];recovery=[]
    for ordinal,(suite,task) in enumerate(tasks):
        if ordinal<20:
            for initial in range(20):
                key=f'native-full40::{suite}::{task}::{initial}'
                training.append(dict(state=dict(pair_key=key,task_id=task,split='train',initialization=dict(init_state_index=initial))))
        split='seen_new_initials' if ordinal<20 else 'validation' if ordinal<28 else 'unseen_test'
        for initial in range(20 if split=='validation' else 40,30 if split=='validation' else 50):
            key=f'native-full40::{suite}::{task}::{initial}';index=ordinal*10+(initial%10)
            for direction in range(8):
                view=1+8*((index+direction)%3)+direction
                recovery.append(dict(state_key=key,task=task,suite=suite,initial=index,split=split,view_id=view,
                    angles_deg=views()[view],start=direction,method='B-seed-0',query_cap=8,execution_repeats=[0,1],max_steps=HORIZONS[suite]))
    schedules={}
    for seed in range(3):
        schedules[seed]=[]
        for episode in range(2048):
            i=(episode+seed*37)%400;s=training[i]['state'];_,suite,task,_=s['pair_key'].split('::')
            schedules[seed].append(dict(episode=episode,initial=i,state_key=s['pair_key'],suite=suite,task=task))
    models=[OperationSpec('fastwam',{s:700 if s=='libero_10' else 400 for s in HORIZONS},10,30),OperationSpec('groot',dict(HORIZONS),10,10,1)]
    return schedules,training,recovery,models


def test_complete_scope_and_native_operation_budgets():
    args=inputs();before=deepcopy(args);p=compile_mainline(*args)
    assert args==before and p['dispatch_enabled'] is False
    assert p['total_call_cap_before_verified_reuse']==4_007_512
    f,g=p['models']['fastwam'],p['models']['groot']
    assert f['budgets']['ours']['operation_calls']==912_000
    assert g['budgets']['ours']['operation_calls']==633_600
    assert sum(x['primary_call_cap'] for x in (f,g))==2_717_912
    assert g['budgets']['training']['warmup_calls']==3456
    for model in (f,g):
        assert model['operation_episodes']==39_200 and model['primary_operation_episodes']==26_400
        assert model['budgets']['training']['score_calls']==98_304
        assert len(model['phases']['baseline'])==3600
        assert sum(r['method']=='reference' for r in model['phases']['baseline'])==400
        assert len(model['phases']['ours'])==9600
    assert {r['trial_id'] for r in f['phases']['ours']}.isdisjoint(r['trial_id'] for r in g['phases']['ours'])


@pytest.mark.parametrize('change',['train_split','episode_missing','duplicate_start','wrong_angle','wrong_task_split','model_duplicate'])
def test_scientific_changes_rejected(change):
    schedules,training,recovery,models=inputs()
    if change=='train_split':training[0]['state']['split']='validation'
    if change=='episode_missing':schedules[0].pop()
    if change=='duplicate_start':recovery[1]=deepcopy(recovery[0])
    if change=='wrong_angle':recovery[0]['angles_deg']=[1.,2.]
    if change=='wrong_task_split':recovery[0]['split']='unseen_test'
    if change=='model_duplicate':models.append(models[0])
    with pytest.raises(ValueError):compile_mainline(schedules,training,recovery,models)


def test_invalid_native_contract_is_not_an_admission():
    with pytest.raises(ValueError):OperationSpec('groot',dict(HORIZONS),0,10).validate()
