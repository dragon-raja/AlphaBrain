"""Complete only predeclared missing RLDX grid operations; no new score queries.

Model/runtime and interpreter stay resident, every operation owns a fresh
physical environment and native temporal history. Four-operation chunks bound
long tails; trial/noise identities are independent of host, slot and chunk.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest,task_key
from AlphaBrain.research.dsol.data.operation_contract import rldx_contract
from AlphaBrain.research.dsol.data.spatial_completion import key
from .rldx_runtime import validate_contract
from .controlled_rldx_operation import move_to,rollout,input_stamp,LiberoHistory

SCHEMA='CONTROLLED_RLDX_SPATIAL_COMPLETION_V1'


def limits(rows):
    return dict(episodes=len(rows),score=0,paired_execution=90*len(rows),nominal_execution=0)


def validate_campaign(path):
    c=read(path)
    if c['schema']!=SCHEMA or not 1<=len(c['blocks'])<=288 or not c['hosts']:
        raise ValueError('Explicit bounded missing-grid campaign required')
    contract=validate_contract(c['runtime_contract'],c['runtime_contract_sha256']);sim=read(c['sim_config'])
    if sim['operation_contract']!=rldx_contract(c['runtime_contract_sha256']):raise ValueError('Native operation contract differs')
    required={c['runtime_contract']:c['runtime_contract_sha256'],**contract['input_hashes']}
    required.update({p:file_digest(p) for p in (c['sim_config'],sim['inputs'],sim['anchors'],c['reuse_manifest'])})
    for p,h in c['engineering_proofs'].items():
        if file_digest(p)!=h or read(p)['status']!='PASS_NATIVE_INSTRUMENTED_TRAJECTORY':raise ValueError('Missing verified transport/node trajectory')
        required[p]=h
    if c['input_hashes']!=required:raise ValueError('Changed or unbound completion inputs')
    plan=read(c['reuse_manifest']);expected={key(r):r for r in plan['missing']};seen=[];states=read(sim['inputs'])['records']
    reused={key(r) for r in plan['reuse']}
    for i,b in enumerate(c['blocks']):
        if b['index']!=i or b['host'] not in c['hosts'] or not 1<=len(b['rows'])<=4 or b['limits']!=limits(b['rows']):
            raise ValueError('Explicit four-operation-or-smaller block required')
        for r in b['rows']:
            if r!=expected.get(key(r)) or key(r) in reused:raise ValueError('Unplanned or already completed operation')
            state=states[r['initial']]['state']
            if (state['pair_key']!=r['state_key'] or state['split']!='validation' or state['suite']!=r['suite']
                    or state['initialization']['settle_steps']!=10):raise ValueError('Native development state identity differs')
            seen.append(key(r))
    if (len(seen)!=len(expected) or len(set(seen))!=len(seen)
            or c['limits']!=limits(plan['missing']) or c['limits']!=plan['limits']):
        raise ValueError('Missing/repeated grid coverage or changed budget')
    return c


def request_for(path,index):
    c=read(path);b=dict(c['blocks'][index]);b['addresses']=[dict(model='rldx',task=r['state_key'],initial=r['state_key'],
        view=f"fixed-{r['view_id']}",stream='execution',repeat=r['repeat'],noise=f"execution-{r['repeat']}") for r in b['rows']]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_spec(binding,key_):
    s=binding['tasks'][key_];req=s['request'];b=req['block'];validate_campaign(req['campaign'])
    if (req!=request_for(binding['campaign'],b['index']) or s['limits']!=b['limits']
            or s['addresses']!=b['addresses'] or s['host']!=req['host'] or s['stage']!='formal'
            or key_!=task_key(s['identity_experiment_id'],req['campaign_sha256'],b['addresses'])):
        raise ValueError('Completion scientific task identity differs')
    return b['limits']


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/rldx_spatial_panel.py:run')
    if not callable(guard):raise ValueError('Live per-call continuation required')
    c=validate_campaign(request['campaign']);b=request['block']
    if request!=request_for(request['campaign'],b['index']):raise ValueError('Changed grid request')
    from .rldx_runtime import RLDXRuntime
    from .search_live import simulator
    output=Path(output);output.mkdir(parents=True,exist_ok=False);guard();sim=read(c['sim_config'])
    open_sim=simulator if simulator_pool is None else simulator_pool.session
    total=b['limits']['paired_execution'];tasks=[]
    with ExitStack() as stack:
        factory=lambda:RLDXRuntime(c['runtime_contract'],c['runtime_contract_sha256'],output,total)
        if runtime_cache is None:runtime=factory()
        else:
            runtime,reused=stack.enter_context(runtime_cache.lease(c['runtime_contract_sha256'],factory,lambda v:v.bind_task(output,total)))
            write(output/'resident.json',dict(reused=reused,identity=c['runtime_contract_sha256']))
        for i,r in enumerate(b['rows']):
            folder=output/f'episode-{i:03d}';folder.mkdir();guard()
            with open_sim(c['sim_config'],sim,r['initial'],folder) as query:
                selected=move_to(query,r['angles_deg']);example=query(dict(kind='begin_execution'))['example']
                if input_stamp(LiberoHistory(selected))!=input_stamp(LiberoHistory(example)):
                    raise ValueError('Execution initial sensors changed')
                result,arrays=rollout(query,runtime,example,r,r['repeat'],'fixed-grid',guard=guard)
            result.update(view_id=r['view_id'],angles_deg=r['angles_deg'],initial=r['initial'],suite=r['suite'])
            np.savez_compressed(folder/'trajectory.npz',**arrays);write(folder/'result.json',result);tasks.append(result)
            write(output/'progress.json',dict(operations=len(tasks),calls=runtime.calls))
    write(output/'results.json',dict(status='COMPLETE_FIXED_GRID',tasks=tasks,training_updates=0))


def validate_block(request,output):
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Changed grid result request')
    for p,h in receipt['artifacts'].items():
        if file_digest(output/p)!=h:raise ValueError('Changed grid artifact')
    value=read(output/'results.json');rows=request['block']['rows'];tasks=value['tasks']
    calls=[json.loads(x) for x in (output/'calls.jsonl').read_text().splitlines()]
    attempts=[json.loads(x) for x in (output/'model-attempts.jsonl').read_text().splitlines()]
    if (value['status']!='COMPLETE_FIXED_GRID' or value['training_updates'] or len(tasks)!=len(rows)
            or {key(r) for r in tasks}!={key(r) for r in rows} or len(calls)!=len(attempts)
            or [x['index'] for x in calls]!=list(range(1,len(calls)+1))
            or any(x['identity']['phase']!='execution' for x in calls)
            or len(calls)!=sum(t['replans'] for t in tasks) or len(calls)>request['block']['limits']['paired_execution']
            or any(type(t['success']) is not bool or not 0<t['steps']<=720 or t['replans']!=(t['steps']+7)//8 for t in tasks)):
        raise ValueError('Incomplete, unknown or miscounted fixed-grid operation')
    return dict(calls=dict(score=0,execution=len(calls)),receipt_sha256=file_digest(output/'complete.json'))
