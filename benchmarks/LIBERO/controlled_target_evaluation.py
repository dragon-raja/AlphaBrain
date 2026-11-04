"""Target-model operation panels under the existing center authority.

Frozen rows and noise identities are independent of resource slots. The first
panel compares native single-action execution with the ten-action transport;
its outcomes are engineering evidence, never a baseline success-rate estimate.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import math
import re
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.data.operation_contract import validate_fastwam
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest,task_key

SCHEMA='CONTROLLED_TARGET_OPERATION_V1'


def execution_hosts(c):
    """Explicit resource binding, independent of frozen scientific rows."""
    hosts=c.get('execution_hosts',['fresh'])
    if (not isinstance(hosts,list) or not hosts or len(hosts)!=len(set(hosts))
            or any(not isinstance(h,str) or re.fullmatch(r'[a-z][a-z0-9_-]{0,63}',h) is None for h in hosts)):
        raise ValueError('Explicit unique execution hosts required')
    # Actual admission remains the exact node/SSH/identity intersection in
    # controlled_campaign.campaign_nodes; a syntactically valid name is no permit.
    return hosts


def row_limits(rows,contract):
    return dict(episodes=len(rows),score=0,paired_execution=sum(
        math.ceil(contract['horizons'][r['suite']]/10) for r in rows),nominal_execution=0)


def validate_assets(path):
    c=read(path)
    from .search_runtime import selected_checkpoint
    from AlphaBrain.research.dsol.data.operation_contract import simulator_action_contract
    sim=read(c['sim_config']);model=selected_checkpoint(sim,allow_fastwam=True,allow_groot=True)
    is_groot=model.get('implementation')=='groot_native'
    if is_groot:
        from .groot_runtime import validate_contract
        from AlphaBrain.research.dsol.data.operation_contract import validate_groot
        contract=validate_groot(sim['operation_contract'])
        if c.get('target_model')!='groot_n15' or c.get('numerical_proof') not in c['input_hashes']:
            raise ValueError('GR00T needs its explicit numerical proof and model identity')
        proof=read(c['numerical_proof'])
        if proof['status']!='PASS_NUMERICAL_ONLY' or not proof['rows'] or any(not r['exact'] for r in proof['rows']):
            raise ValueError('GR00T numerical proof did not pass')
    else:
        from .legacy.fastwam_runtime import validate_contract
        contract=validate_fastwam(sim['operation_contract'])
        if c.get('target_model','fastwam')!='fastwam':raise ValueError('Target model identity differs')
    simulator_action_contract(sim)
    runtime=validate_contract(model['runtime_contract'],model['runtime_contract_sha256'])
    verified={model['runtime_contract']:model['runtime_contract_sha256'],**runtime['input_hashes']}
    if model['path']!=runtime['checkpoint'] or model['weights_sha256']!=verified[model['path']]:
        raise ValueError('Target model weights differ')
    required={c['sim_config'],sim['inputs'],c['execution_noise']}
    noise=read(c['execution_noise']);required.add(noise['noise_file'])
    if (not required.issubset(c['input_hashes']) or
            any(c['input_hashes'].get(p)!=h for p,h in verified.items())):
        raise ValueError('Unbound target operation assets')
    for p,h in c['input_hashes'].items():
        if verified.get(p)!=h and file_digest(p)!=h:raise ValueError('Changed target operation asset: '+p)
    if (noise['action_horizon']!=10 or noise['action_dim']!=7 or noise['repeat_count']!=2
            or noise['max_replans']<70 or noise['noise_file_sha256']!=c['input_hashes'][noise['noise_file']]):
        raise ValueError('Insufficient explicit paired target execution noise')
    states=read(sim['inputs'])['records']
    return c,sim,states,contract


def validate_campaign(path):
    c,sim,states,contract=validate_assets(path)
    if c['schema']!=SCHEMA or c['purpose']!='trajectory_admission':
        raise ValueError('Explicit target operation admission required')
    noise=read(c['execution_noise']);rows=[]
    for i,b in enumerate(c['blocks']):
        if b['index']!=i or b['host'] not in execution_hosts(c) or not b['rows'] or b['limits']!=row_limits(b['rows'],contract):
            raise ValueError('Unbound block/host/budget')
        rows.extend(b['rows'])
    if not rows or len(rows)>8 or c['limits']!=row_limits(rows,contract):
        raise ValueError('Bounded complete trajectory-admission budget required')
    groups={}
    for r in rows:
        if (type(r['initial']) is not int or not 0<=r['initial']<len(states)
                or r['repeat'] not in (0,1) or r['variant'] not in ('native_single_step','chunk10')):
            raise ValueError('Invalid target admission identity')
        state=states[r['initial']]['state']
        if (state['split']!='train' or state['pair_key']!=r['state_key'] or state['suite']!=r['suite']
                or state['initialization']['settle_steps']!=contract['settle_steps'] or r['state_key'] not in noise['state_keys']):
            raise ValueError('Wrong target initialization/state identity')
        groups.setdefault((r['state_key'],r['repeat']),[]).append(r['variant'])
    if any(sorted(v)!=['chunk10','native_single_step'] for v in groups.values()):
        raise ValueError('Complete nonduplicated reference/chunk pairs required')
    return c


def request_for(path,index):
    c=read(path);b=dict(c['blocks'][index])
    b['addresses']=[dict(model=c.get('target_model','fastwam'),task=r['state_key'],initial=r['state_key'],
        view='canonical-'+r['variant'],stream='execution',repeat=r['repeat'],noise=f"paired-{r['repeat']}") for r in b['rows']]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_spec(binding,key):
    s=binding['tasks'][key];req=s['request'];b=req['block'];validate_campaign(req['campaign'])
    if (req!=request_for(binding['campaign'],b['index']) or s['limits']!=b['limits']
            or s['addresses']!=b['addresses'] or s['host']!=req['host'] or s['stage']!='formal'
            or key!=task_key(s['identity_experiment_id'],req['campaign_sha256'],b['addresses'])):
        raise ValueError('Target operation request differs')
    return b['limits']


class ExecutionVariant:
    def __init__(self,runtime,reference):self.runtime,self.reference=runtime,reference
    @property
    def calls(self):return self.runtime.calls
    def infer(self,*args,**kwargs):
        return self.runtime.infer(*args,**kwargs,instrument=not self.reference)


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_target_evaluation.py:run')
    from .search_runtime import create_policy_runtime,bind_policy_task
    from .search_live import simulator
    from .local_search_rollout import rollout_fastwam_episode,rollout_groot_episode
    from .continuous_evaluation import EvaluationCalls,append_event
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    c=validate_campaign(request['campaign']);b=request['block']
    if request!=request_for(request['campaign'],b['index']):raise ValueError('Changed operation block')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sim=read(c['sim_config']);bank=ExplicitFlowNoiseBank(Path(c['execution_noise']))
    is_groot=c.get('target_model')=='groot_n15'
    rollout=rollout_groot_episode if is_groot else rollout_fastwam_episode
    started=time.monotonic();results=[]
    with ExitStack() as stack:
        if runtime_cache is None:runtime=create_policy_runtime(sim,output,b['limits']['paired_execution'])
        else:runtime,_=stack.enter_context(runtime_cache.lease(digest(sim),
            lambda:create_policy_runtime(sim,output,b['limits']['paired_execution']),
            lambda v:bind_policy_task(v,output,b['limits']['paired_execution'])))
        calls=EvaluationCalls(runtime,output,dict(score=0,execution=b['limits']['paired_execution']),guard=guard)
        calls.stage='execution';open_sim=simulator if simulator_pool is None else simulator_pool.session
        for i,row in enumerate(b['rows']):
            guard();folder=output/f'episode-{i:03d}';folder.mkdir()
            reference=row['variant']=='native_single_step'
            # GR00T graph observation has its own independent native-action
            # proof. This gate compares only single-step vs chunk transport.
            calls.runtime=runtime if is_groot else ExecutionVariant(runtime,reference)
            with open_sim(c['sim_config'],sim,row['initial'],folder) as query:
                example=query(dict(kind='begin_execution'))['example']
                result,arrays=rollout(query,calls.infer,example,bank,row['state_key'],
                    row['repeat'],row,contract=sim['operation_contract'],suite=row['suite'],
                    record=lambda event:append_event(folder/'events.jsonl',event),reference_single_step=reference)
            np.savez_compressed(folder/'trajectory.npz',**arrays)
            result.update(row,trajectory=str(folder/'trajectory.npz'))
            write(folder/'result.json',result);results.append(result)
            write(output/'progress.json',dict(tasks=len(results),calls=calls.counts))
        write(output/'results.json',dict(tasks=results))
        write(output/'status.json',dict(status='COMPLETE',calls=calls.counts,training_updates=0,
            seconds=time.monotonic()-started))


def validate_block(request,output):
    output=Path(output);complete=read(output/'complete.json')
    if complete['request_sha256']!=digest(request):raise ValueError('Changed operation receipt')
    for p,h in complete['artifacts'].items():
        if file_digest(output/p)!=h:raise ValueError('Changed operation result')
    tasks=read(output/'results.json')['tasks'];status=read(output/'status.json')
    rows=request['block']['rows'];key=lambda r:(r['state_key'],r['repeat'],r['variant'])
    if len(tasks)!=len(rows) or {key(r) for r in rows}!={key(r) for r in tasks}:
        raise ValueError('Missing/duplicated target episodes')
    attempts=[json.loads(x) for x in (output/'model-attempts.jsonl').read_text().splitlines()]
    runtime=[json.loads(x) for x in (output/'calls.jsonl').read_text().splitlines()]
    count=sum(r['replans'] for r in tasks)
    if (len(attempts)!=count or len(runtime)!=count or count>request['block']['limits']['paired_execution']
            or status['status']!='COMPLETE' or status['calls']!=dict(score=0,execution=count)
            or any(type(r['success']) is not bool or not 0<r['steps']<=r['horizon'] for r in tasks)):
        raise ValueError('Target operation accounting differs')
    return dict(calls=dict(score=0,execution=count),receipt_sha256=file_digest(output/'complete.json'))


def compare_trajectories(output):
    """Exact paired full-trajectory gate; no success-rate claim or tolerance drift."""
    output=Path(output);tasks=read(output/'results.json')['tasks'];groups={}
    for row in tasks:groups.setdefault((row['state_key'],row['repeat']),{})[row['variant']]=row
    checks=[]
    for (state,repeat),pair in groups.items():
        if set(pair)!={'native_single_step','chunk10'}:raise ValueError('Missing paired trajectory')
        a,b=pair['native_single_step'],pair['chunk10']
        fields=('success','steps','replans','horizon','input_sha256','execution_noise_sha256',
                'final_input_sha256','physics_after_replan')
        values={k:a[k]==b[k] for k in fields}
        with np.load(a['trajectory']) as x,np.load(b['trajectory']) as y:
            for k in ('actions','full_decoded_action_chunks','replan_action_offsets'):
                values[k]=x[k].dtype==y[k].dtype and np.array_equal(x[k],y[k])
        if not a['physics_after_replan'] or len(a['physics_after_replan'])!=a['replans']:
            values['physical_evidence']=False
        checks.append(dict(state_key=state,repeat=repeat,equal=values,passed=all(values.values())))
    return dict(status='PASS' if checks and all(r['passed'] for r in checks) else 'FAIL',pairs=checks)
