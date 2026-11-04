"""CPU result-contract tests; synthetic artifacts never grant GPU admission."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest
from benchmarks.LIBERO import native_recovery


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def journal(path,rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def fixture(tmp_path,all_completed=False):
    old=tmp_path/'old/block/case-0000';out=tmp_path/'new';out.mkdir()
    row=dict(trial_id='synthetic',method='stay',state_key='s',initial=0,start=2,suite='libero_10')
    source_attempts=[];results=[]
    for repeat,n in [(0,2),(1,1)]:
        ep=old/f'episode-{repeat}';ep.mkdir(parents=True)
        arrays=np.stack([np.full((32,7),j+1,np.float32) for j in range(n)])
        available=n if all_completed or repeat==1 else 1
        events=[]
        for j in range(available):
            ident=dict(row,repeat=repeat,replan=j)
            source_attempts.append(dict(identity=dict(ident,stage='execution'),input_sha256=f'input-{repeat}-{j}',noise_sha256=f'noise-{repeat}-{j}'))
            events.append(dict(kind='prediction',identity=ident,input_sha256=f'input-{repeat}-{j}',full_decoded_actions=arrays[j].tolist()))
        journal(ep/'events.jsonl',events)
        trajectory=(ep if available==n else out)/f'trajectory-{repeat}.npz'
        if available==n:trajectory=ep/'trajectory.npz'
        np.savez(trajectory,actions=np.zeros((n*10,7),np.float32),full_decoded_action_chunks=arrays)
        result=dict(row,repeat=repeat,replans=n,steps=n*10,success=bool(repeat),trajectory=str(trajectory),
                    input_sha256=[f'input-{repeat}-{j}' for j in range(n)],execution_noise_sha256=[f'noise-{repeat}-{j}' for j in range(n)])
        if available==n:save(ep/'result.json',result)
        results.append(result)
    journal(old.parent/'model-attempts.jsonl',source_attempts)
    save(old/'search.json',dict(row,queries=0))
    proto=tmp_path/'protocol.json';save(proto,dict(family='fastwam'))
    proof=dict(status='AUDITED_EXPLICIT_CONTINUATION',old_workers_exited=True,automatic_retry=False,uncertain_calls_refunded=False,
               artifacts={str(p):file_digest(p) for p in old.parent.rglob('*') if p.is_file()},records={digest(row):dict(case=str(old))})
    proof_path=tmp_path/'proof.json';save(proof_path,proof)
    c=dict(phase='baseline',protocol=str(proto),explicit_recovery=str(proof_path),input_hashes={str(proof_path):file_digest(proof_path)},
           blocks=[dict(rows=[row])])
    campaign=tmp_path/'campaign.json';save(campaign,c)
    reused=[dict(stage='score',source=str(old/'search.json'),count=0,model_calls=0)]
    if all_completed:reused.append(dict(stage='execution',source=str(old/'episode-0/result.json'),count=2,model_calls=0))
    else:reused.append(dict(stage='execution',source=str(old),identity=dict(row,repeat=0,replan=0),input_sha256='input-0-0',noise_sha256='noise-0-0',model_calls=0))
    reused.append(dict(stage='execution',source=str(old/'episode-1/result.json'),count=1,model_calls=0))
    journal(out/'reused-calls.jsonl',reused)
    actual=[] if all_completed else [dict(identity=dict(row,repeat=0,replan=1,stage='execution'),input_sha256='input-0-1',noise_sha256='noise-0-1')]
    journal(out/'model-attempts.jsonl',actual);journal(out/'calls.jsonl',[{} for _ in actual])
    save(out/'status.json',dict(status='COMPLETE',calls=dict(score=0,execution=len(actual))))
    save(out/'results.json',dict(tasks=results,searches=[dict(row,queries=0)]))
    request=dict(campaign=str(campaign),block=dict(rows=[row],limits=dict(score=0,paired_execution=4)))
    stamp(request,out)
    return request,out,c,proof,reused


def stamp(request,out):
    save(out/'complete.json',dict(request_sha256=digest(request),artifacts={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}))


@pytest.mark.parametrize('all_completed',[False,True])
def test_native_baseline_settles_only_actual_calls_and_keeps_all_results(tmp_path,all_completed):
    request,out,_,_,_=fixture(tmp_path,all_completed)
    result=native_recovery.validate_block(request,out)
    assert result['calls']==dict(score=0,execution=0 if all_completed else 1)
    tasks=json.loads((out/'results.json').read_text())['tasks']
    assert {v['success'] for v in tasks}=={False,True}


@pytest.mark.parametrize('fault',['duplicate','input','noise','count','source','charged','stage'])
def test_native_baseline_rejects_forged_or_duplicate_cache_receipts(tmp_path,fault):
    request,out,_,_,reused=fixture(tmp_path)
    if fault=='duplicate':reused.append(copy.deepcopy(reused[1]))
    elif fault=='input':reused[1]['input_sha256']='wrong'
    elif fault=='noise':reused[1]['noise_sha256']='wrong'
    elif fault=='count':reused[2]['count']=2
    elif fault=='source':reused[1]['source']=str(tmp_path/'unapproved')
    elif fault=='charged':reused[1]['model_calls']=1
    else:reused[1]['stage']='unrecognized'
    journal(out/'reused-calls.jsonl',reused);stamp(request,out)
    with pytest.raises(ValueError):native_recovery.validate_block(request,out)


@pytest.mark.parametrize('fault',['owner','refund','automatic','record','scope','family','unbound'])
def test_native_baseline_requires_explicit_bound_stopped_owner_scope(tmp_path,fault):
    request,out,c,proof,_=fixture(tmp_path)
    if fault=='owner':proof['old_workers_exited']=False
    elif fault=='refund':proof['uncertain_calls_refunded']=True
    elif fault=='automatic':proof['automatic_retry']=True
    elif fault=='record':proof['records']={}
    elif fault=='scope':c['phase']='ours'
    elif fault=='family':save(Path(c['protocol']),dict(family='groot_n15'))
    elif fault=='unbound':proof['artifacts'].pop(next(k for k in proof['artifacts'] if k.endswith('model-attempts.jsonl')))
    save(Path(c['explicit_recovery']),proof);c['input_hashes'][c['explicit_recovery']]=file_digest(c['explicit_recovery']);save(Path(request['campaign']),c)
    stamp(request,out)
    with pytest.raises(ValueError):native_recovery.validate_block(request,out)


def test_native_baseline_rejects_changed_old_prediction_bytes(tmp_path):
    request,out,_,proof,_=fixture(tmp_path)
    source=next(Path(k) for k in proof['artifacts'] if k.endswith('episode-0/events.jsonl'))
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='evidence changed'):native_recovery.validate_block(request,out)


def test_native_baseline_rejects_actual_call_overlap_even_when_totals_match(tmp_path):
    request,out,_,_,_=fixture(tmp_path)
    attempts=[json.loads(s) for s in (out/'model-attempts.jsonl').read_text().splitlines()]
    attempts[0]['identity']['replan']=0
    attempts[0].update(input_sha256='input-0-0',noise_sha256='noise-0-0')
    journal(out/'model-attempts.jsonl',attempts);stamp(request,out)
    with pytest.raises(ValueError,match='duplicates a reused prediction'):native_recovery.validate_block(request,out)
