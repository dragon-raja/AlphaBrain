"""Finite RLDX numerical admission under the standard live center authority.

Three variants share the original input and native noise: no observer, velocity
observer, observer plus read-only cognition tokens. No optimizer or old queue.
"""
from pathlib import Path
import json
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest,task_key
from .rldx_runtime import validate_contract

SCHEMA='CONTROLLED_RLDX_NUMERICAL_V1'


def validate_campaign(path):
    c=read(path)
    if c['schema']!=SCHEMA or len(c['blocks'])!=1:
        raise ValueError('One bounded numerical block required')
    if (not isinstance(c['seeds'],list) or len(c['seeds'])!=2 or len(set(c['seeds']))!=2
            or any(type(v) is not int or not 0<=v<2**63 for v in c['seeds'])):
        raise ValueError('Two distinct explicit native seeds required')
    b=c['blocks'][0]
    if b['index']!=0 or not 1<=len(b['rows'])<=2 or b['host'] not in c.get('hosts',['fresh']):
        raise ValueError('One or two declared-host training states required')
    if len({r['initial'] for r in b['rows']})!=len(b['rows']):raise ValueError('Duplicate state')
    expected=dict(episodes=0,score=6*len(b['rows']),paired_execution=0,nominal_execution=0)
    if b['limits']!=expected or c['limits']!=expected:raise ValueError('Numerical budget differs')
    runtime=validate_contract(c['runtime_contract'],c['runtime_contract_sha256'])
    sim=read(c['sim_config'])
    if (sim.get('continuous_camera') is not True or sim.get('continuous_execution')
            or sim.get('raw_observation')!='LIBERO_RAW_RGB256_PROPRIO_V1'):
        raise ValueError('Fixed-state query-only raw renderer required')
    required={c['runtime_contract']:c['runtime_contract_sha256'],**runtime['input_hashes']}
    required.update({p:file_digest(p) for p in (c['sim_config'],sim['inputs'],sim['anchors'])})
    if c['input_hashes']!=required:raise ValueError('Changed or unbound numerical input')
    records=read(sim['inputs'])['records']
    for row in b['rows']:
        i=row['initial']
        if type(i) is not int or not 0<=i<len(records):raise ValueError('Invalid initial')
        state=records[i]['state']
        if (state['split']!='train' or state['pair_key']!=row['state_key']
                or state['initialization']['settle_steps']!=10):
            raise ValueError('Expected declared training initialization')
    return c


def request_for(path,index):
    c=read(path)
    if index!=0:raise ValueError('Only one numerical block')
    b=dict(c['blocks'][0]);b['addresses']=[dict(model='rldx',task=r['state_key'],initial=r['state_key'],
        view='native-numerical-'+variant,stream='score',repeat=m,noise='seed-'+str(c['seeds'][m]))
        for r in b['rows'] for m in (0,1) for variant in ('native','trace','features')]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_spec(binding,key):
    s=binding['tasks'][key];req=s['request'];b=req['block'];validate_campaign(req['campaign'])
    if (req!=request_for(binding['campaign'],b['index']) or s['limits']!=b['limits']
            or s['addresses']!=b['addresses'] or s['host']!=req['host'] or s['stage']!='formal'
            or key!=task_key(s['identity_experiment_id'],req['campaign_sha256'],b['addresses'])):
        raise ValueError('RLDX numerical task identity differs')
    return b['limits']


def compare(native,trace,features):
    for key in ('actions','normalized_actions','rng_cpu','rng_cuda'):
        if not np.array_equal(native[key],trace[key]) or not np.array_equal(native[key],features[key]):
            raise ValueError('RLDX observation changed native output or RNG: '+key)
    for key in ('states','velocities','initial_noise'):
        if not np.array_equal(trace[key],features[key]):raise ValueError('Native feature read changed flow trace')
    return True


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_rldx.py:run')
    if not callable(guard):raise ValueError('Live controller guard required')
    c=validate_campaign(request['campaign'])
    if request!=request_for(request['campaign'],0):raise ValueError('Changed request')
    from .rldx_runtime import RLDXRuntime,LiberoHistory
    from .search_live import simulator
    from .local_search_rollout import input_identity
    output=Path(output);output.mkdir(parents=True,exist_ok=False);guard()
    started=time.monotonic();sim=read(c['sim_config'])
    runtime=RLDXRuntime(c['runtime_contract'],c['runtime_contract_sha256'],output,c['limits']['score'])
    rows=[]
    for case in request['block']['rows']:
        folder=output/f"initial-{case['initial']:03d}";folder.mkdir();guard()
        with simulator(c['sim_config'],sim,case['initial'],folder) as query:
            example=query(dict(kind='query_pose'))['example']
        np.savez(folder/'raw-input.npz',**{k:v for k,v in example['fastwam_raw'].items() if k!='schema'})
        write(folder/'input-identity.json',dict(input_sha256=input_identity(example),language=example['lang']))
        history=LiberoHistory(example)
        for member,seed in enumerate(c['seeds']):
            values=[]
            for variant in ('native','trace','features'):
                guard();identity=dict(state=case['state_key'],member=member,phase='score',variant=variant)
                value=runtime.infer(history,seed,identity,instrument=variant!='native',features=variant=='features')
                np.savez(folder/f'member-{member}-{variant}.npz',**{k:v for k,v in value.items() if isinstance(v,np.ndarray)})
                values.append(value)
            exact=compare(*values);value=values[-1]
            rows.append(dict(state=case['state_key'],member=member,exact=exact,scores=value['scores'],
                seconds=[v['seconds'] for v in values],peak_allocated_mib=value['peak_allocated_mib'],
                peak_reserved_mib=value['peak_reserved_mib'],
                features={k:list(value[k].shape) for k in ('backbone_features','state_features')}))
    if runtime.calls!=c['limits']['score']:raise ValueError('Numerical call accounting differs')
    write(output/'admission.json',dict(status='PASS_NUMERICAL_ONLY',model='rldx',calls=runtime.calls,rows=rows,
        manipulation_actions=0,training_updates=0,seconds=time.monotonic()-started))


def validate_block(request,output):
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Changed admission request')
    for name,stamp in receipt['artifacts'].items():
        if file_digest(output/name)!=stamp:raise ValueError('Changed result')
    value=read(output/'admission.json');calls=[json.loads(x) for x in (output/'calls.jsonl').read_text().splitlines()]
    if (value['status']!='PASS_NUMERICAL_ONLY' or value['calls']!=request['block']['limits']['score']
            or len(calls)!=value['calls'] or [x['index'] for x in calls]!=list(range(1,len(calls)+1))
            or any(not x['exact'] for x in value['rows']) or len(value['rows'])!=2*len(request['block']['rows'])
            or value['training_updates'] or value['manipulation_actions']):raise ValueError('Incomplete RLDX numerical receipt')
    return dict(calls=dict(score=value['calls'],execution=0),receipt_sha256=file_digest(output/'complete.json'))
