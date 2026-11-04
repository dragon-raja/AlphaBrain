from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.prefix_protocol import validate_mainline,validate_evaluation_row
from AlphaBrain.research.dsol.data.score_prefix import contract


def protocol():
    cells={}
    for f,p in [('pi05',10),('groot_n15',4),('rldx',4)]:
        for seed in range(3):cells[f'{f}-p{p}-seed-{seed}']=dict(family=f,prefix=p,seed=seed,score_contract=contract(f,p))
    return dict(schema='PREFIX_MAINLINE_V1',updates=64,episodes_per_update=32,seeds=[0,1,2],maximum_model_calls=5200000,query_cap=8,query_cost=.01,cells=cells)


@pytest.mark.parametrize('change',['seed','rldx_p3','budget','updates'])
def test_approved_scope_is_exact(change):
    p=protocol();validate_mainline(p)
    if change=='seed':p['cells']['rldx-p4-seed-2']['seed']=0
    if change=='rldx_p3':p['cells']['rldx-p4-seed-2']['prefix']=3
    if change=='budget':p['maximum_model_calls']+=1
    if change=='updates':p['updates']=8
    with pytest.raises(ValueError):validate_mainline(p)


def test_final_version_and_initialization_never_inherit_stale_headers():
    c=dict(schedule='frozen',evaluation_version=64,stay=False)
    cell=dict(family='rldx',seed=0,evaluation_schedules={'frozen':dict(purpose='final',version=64,method='ppo')})
    row=dict(settle_steps=10,query_cap=8,execution_repeats=[0,1],split='unseen_test');state=dict(split='test_unseen_task',initialization=dict(settle_steps=10))
    validate_evaluation_row(c,cell,row,state)
    with pytest.raises(ValueError):validate_evaluation_row(dict(c,evaluation_version=8),cell,row,state)
    with pytest.raises(ValueError):validate_evaluation_row(c,cell,dict(row,settle_steps=30),state)
    with pytest.raises(ValueError):validate_evaluation_row(dict(c,schedule='unregistered'),cell,row,state)
    dev=deepcopy(cell);dev['evaluation_schedules']['frozen'].update(purpose='development',version=16)
    with pytest.raises(ValueError):validate_evaluation_row(dict(c,evaluation_version=16),dev,row,state)


def test_seen_new_initials_preserve_original_training_task_membership():
    c=dict(schedule='frozen',evaluation_version=64,stay=False)
    cell=dict(family='rldx',seed=0,evaluation_schedules={'frozen':dict(purpose='final',version=64,method='ppo')})
    row=dict(settle_steps=10,query_cap=8,execution_repeats=[0,1],split='seen_new_initials')
    state=dict(split='train',initialization=dict(settle_steps=10))
    validate_evaluation_row(c,cell,row,state)
    with pytest.raises(ValueError):validate_evaluation_row(c,cell,dict(row,split='unseen_test'),state)


def test_task_admission_cannot_reuse_no_task_or_other_task_proof(tmp_path):
    from AlphaBrain.research.dsol.data.prefix_protocol import validate_admission
    from AlphaBrain.research.dsol.data.artifacts import write
    from AlphaBrain.research.dsol.execution.manifest import digest,file_digest
    cell=dict(family='pi05',model_hashes={'suite':'model'},core_spec={'variant':'U0'},task_condition={'cache_sha256':'a'*64})
    path=tmp_path/'proof.json'
    proof=dict(status='PASS_PREFIX_NUMERICAL_AND_TRAJECTORY',family='pi05',source_identity='source',model_hashes=cell['model_hashes'],admitted_host='host',admitted_suite='suite',search_core_sha256=digest(cell['core_spec']))
    c=dict(source_identity='source',admission_proofs={'host/suite':str(path)},input_hashes={})
    b=dict(host='host',rows=[dict(suite='suite')])
    for task_hash in [None,'wrong-source',digest(cell['task_condition'])]:
        value=dict(proof)
        if task_hash is not None:value['task_condition_sha256']=task_hash
        write(path,value);c['input_hashes'][str(path)]=file_digest(path)
        if task_hash==digest(cell['task_condition']):validate_admission(c,cell,b)
        else:
            with pytest.raises(ValueError,match='Task feature'):validate_admission(c,cell,b)
