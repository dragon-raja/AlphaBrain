"""GR00T numerical admission using the standard ledger and live authority.

Two noise members on one or two training states; one explicitly charged graph
warmup. No manipulation, optimizer, policy search or result-based selection.
"""
from pathlib import Path
import gc
import json
import time
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key
from .groot_runtime import validate_contract

SCHEMA = 'CONTROLLED_GROOT_ADMISSION_V1'


def validate_campaign(path):
    c = read(path)
    from .controlled_target_evaluation import execution_hosts
    if c['schema'] != SCHEMA or not 1<=len(c['blocks'])<=2:
        raise ValueError('One or two bounded GR00T admission blocks required')
    all_rows=[];total=dict(episodes=0,score=0,paired_execution=0,nominal_execution=0)
    for i,b in enumerate(c['blocks']):
        if b['index']!=i or b['host'] not in execution_hosts(c) or not 1<=len(b['rows'])<=2:
            raise ValueError('Explicit admitted host and bounded states required')
        limits=dict(episodes=0,score=4*len(b['rows'])+1,paired_execution=0,nominal_execution=0)
        if b['limits']!=limits:raise ValueError('Admission calls including warmup changed')
        total['score']+=limits['score'];all_rows.extend(b['rows'])
    if (not 1<=len(all_rows)<=2 or len({r['initial'] for r in all_rows})!=len(all_rows)
            or c['limits']!=total):raise ValueError('Unique states and complete bounded budget required')
    contract = validate_contract(c['runtime_contract'], c['runtime_contract_sha256'])
    sim = read(c['sim_config'])
    if (not sim.get('continuous_camera') or sim.get('continuous_execution')
            or sim.get('raw_observation') != 'LIBERO_RAW_RGB256_PROPRIO_V1'):
        raise ValueError('Raw fixed-state query-only admission required')
    required = {c['runtime_contract']:c['runtime_contract_sha256'], **contract['input_hashes']}
    for p in (c['sim_config'], sim['inputs'], sim['anchors']):
        if p not in c['input_hashes']: raise ValueError('Unbound admission input')
    if any(c['input_hashes'].get(p) != h for p,h in required.items()):
        raise ValueError('Unbound model dependency')
    for p,h in c['input_hashes'].items():
        if p not in required and file_digest(p) != h: raise ValueError('Changed admission input')
    states = read(sim['inputs'])['records']
    for row in all_rows:
        if type(row['initial']) is not int or not 0 <= row['initial'] < len(states):
            raise ValueError('Invalid admission state index')
        state = states[row['initial']]['state']
        if (state['pair_key'] != row['state_key'] or state['split'] != 'train'
                or state['suite'] != contract['suite']
                or state['initialization']['settle_steps'] != 10):
            raise ValueError('Training-state / native initialization identity differs')
    return c


def request_for(path,index):
    c = read(path)
    if type(index) is not int or not 0<=index<len(c['blocks']):raise ValueError('Invalid admission block')
    b = dict(c['blocks'][index])
    b['addresses'] = [dict(model='groot_n15', task=r['state_key'], initial=r['state_key'],
        view='numerical_admission_initial', stream='score', repeat=m, noise=f'member-{m}')
        for r in b['rows'] for m in (0,1)]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_spec(binding,key):
    s = binding['tasks'][key]; req = s['request']; b = req['block']
    validate_campaign(req['campaign'])
    if (req != request_for(binding['campaign'],b['index']) or s['limits'] != b['limits']
            or s['addresses'] != b['addresses'] or s['host'] != req['host']
            or s['stage'] != 'formal'
            or key != task_key(s['identity_experiment_id'],req['campaign_sha256'],b['addresses'])):
        raise ValueError('GR00T admission task identity differs')
    return b['limits']


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_groot_admission.py:run')
    if not callable(guard): raise ValueError('Live admission guard required')
    c = validate_campaign(request['campaign'])
    if request != request_for(request['campaign'],request['block']['index']): raise ValueError('Changed admission request')
    from .groot_runtime import GrootRuntime
    from .search_live import simulator
    from .local_search_rollout import input_identity
    import torch
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sim=read(c['sim_config']);started=time.perf_counter();examples=[];controls={};rows=[]
    with np.load(sim['anchors'],allow_pickle=False) as z:noises=z['first_score_noise'].copy()
    for case in request['block']['rows']:
        folder=output/f"initial-{case['initial']:03d}";folder.mkdir();guard()
        with simulator(c['sim_config'],sim,case['initial'],folder) as query:
            example=query(dict(kind='query_pose'))['example']
        np.savez(folder/'raw-input.npz',**{k:v for k,v in example['fastwam_raw'].items() if k!='schema'})
        write(folder/'input-identity.json',dict(input_sha256=input_identity(example),language=example['lang']))
        examples.append((case,example))
    native_dir=output/'native';native_dir.mkdir();observed_dir=output/'observed';observed_dir.mkdir()
    native=GrootRuntime(c['runtime_contract'],c['runtime_contract_sha256'],native_dir,2*len(examples),instrument=False)
    for case,example in examples:
        for m in (0,1):
            guard();identity=dict(state=case['state_key'],member=m,phase='score',variant='native')
            value=native.infer(example,noises[case['initial'],m],identity)
            controls[(case['initial'],m)]=value
            np.savez(native_dir/f"initial-{case['initial']}-member-{m}.npz",actions=value['actions'],normalized_actions=value['normalized_actions'])
    del native;gc.collect();torch.cuda.empty_cache();guard()
    observed=GrootRuntime(c['runtime_contract'],c['runtime_contract_sha256'],observed_dir,2*len(examples)+1,instrument=True)
    case,example=examples[0]
    observed.infer(example,noises[case['initial'],0],dict(state=case['state_key'],member=0,phase='score',variant='charged_graph_warmup'),warmup=True)
    for case,example in examples:
        for m in (0,1):
            guard();identity=dict(state=case['state_key'],member=m,phase='score',variant='observed')
            value=observed.infer(example,noises[case['initial'],m],identity,capture=True)
            np.savez(observed_dir/f"initial-{case['initial']}-member-{m}.npz",**{k:v for k,v in value.items() if isinstance(v,np.ndarray)})
            control=controls[(case['initial'],m)]
            if any(not np.array_equal(control[k],value[k]) for k in ('actions','normalized_actions')):
                raise ValueError('Graph observation changed native GR00T actions')
            rows.append(dict(identity,exact=True,scores=value['scores'],seconds=value['seconds'],control_seconds=control['seconds'],peak_allocated_mib=value['peak_allocated_mib']))
    calls=[]
    for folder in (native_dir,observed_dir):
        for line in (folder/'calls.jsonl').read_text().splitlines():
            item=json.loads(line);item.update(component=folder.name,component_index=item['index'],index=len(calls)+1);calls.append(item)
    if len(calls)!=request['block']['limits']['score']:raise ValueError('GR00T admission accounting differs')
    (output/'calls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in calls))
    write(output/'admission.json',dict(status='PASS_NUMERICAL_ONLY',rows=rows,calls=len(calls),warmup_calls=1,model='groot_n15',training_updates=0,manipulation_actions=0,seconds=time.perf_counter()-started))


def validate_block(request,output):
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Changed admission receipt')
    for name,stamp in receipt['artifacts'].items():
        if file_digest(output/name)!=stamp:raise ValueError('Changed admission result')
    value=read(output/'admission.json');calls=[json.loads(x) for x in (output/'calls.jsonl').read_text().splitlines()]
    if (value['status']!='PASS_NUMERICAL_ONLY' or value['calls']!=request['block']['limits']['score']
            or len(calls)!=value['calls'] or [x['index'] for x in calls]!=list(range(1,len(calls)+1))
            or sum(x['warmup'] for x in calls)!=1 or len(value['rows'])!=2*len(request['block']['rows'])
            or any(not x['exact'] for x in value['rows']) or value['training_updates'] or value['manipulation_actions']):
        raise ValueError('Incomplete GR00T numerical admission')
    return dict(calls=dict(score=value['calls'],execution=0),receipt_sha256=file_digest(output/'complete.json'))
