"""Bounded numerical admission through the same center ledger as production.

At most two declared training states and two paired noise members. No optimizer,
manipulation action, search policy or success-rate experiment is run here.
"""
from pathlib import Path
import json
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest,task_key

SCHEMA='CONTROLLED_RUNTIME_ADMISSION_V1'


def validate_campaign(path):
    c=read(path)
    from . import native_feature_admission
    if c.get('schema')==native_feature_admission.SCHEMA:return native_feature_admission.validate_campaign(path)
    if (c['schema']!=SCHEMA or c['kind'] not in ('fastwam_trace','source_visual')
            or len(c['blocks'])!=1 or c['blocks'][0]['host']!='fresh'):
        raise ValueError('Only explicitly bounded Fresh numerical admission is supported')
    b=c['blocks'][0];cases=b['rows']
    if (not 1<=len(cases)<=2 or b['index']!=0
            or len({r['initial'] for r in cases})!=len(cases)):
        raise ValueError('One or two unique explicit states required')
    calls=len(cases)*(4 if c['kind']=='fastwam_trace' else 2)
    expected=dict(episodes=0,score=calls,paired_execution=0,nominal_execution=0)
    if c['limits']!=expected or b['limits']!=expected:
        raise ValueError('Numerical admission budget changed')
    verified={}
    if c['sim_config'] not in c['input_hashes']:raise ValueError('Unbound admission simulator')
    sim=read(c['sim_config'])
    if any(p not in c['input_hashes'] for p in (sim['inputs'],sim['anchors'])):
        raise ValueError('Unbound admission states/noise')
    if not sim.get('continuous_camera') or sim.get('continuous_execution'):
        raise ValueError('Admission must be fixed-state query-only')
    from .search_runtime import selected_checkpoint
    model=selected_checkpoint(sim,allow_native=True,allow_fastwam=True)
    if c['kind']=='fastwam_trace':
        from .legacy.fastwam_runtime import RAW_OBSERVATION,validate_contract
        if model.get('implementation')!='fastwam_native' or sim.get('raw_observation')!=RAW_OBSERVATION:
            raise ValueError('Explicit FastWAM/raw-pixel binding required')
        contract=validate_contract(model['runtime_contract'],model['runtime_contract_sha256'])
        required={model['runtime_contract']:model['runtime_contract_sha256'],**contract['input_hashes']}
        if any(c['input_hashes'].get(p)!=h for p,h in required.items()):
            raise ValueError('Unbound FastWAM admission dependency')
        # validate_contract has just hashed these exact bytes in this same
        # invocation. Do not reread tens of GB a second time. This is not a
        # persistent/mtime cache: every new validation rehashes the assets.
        verified=required
    else:
        if model.get('implementation')!='openpi_native':raise ValueError('Native source reference required')
        bridge=c['visual_bridge']
        required={str(Path(model['path'])/'model.safetensors'):model['weights_sha256'],
            str(Path(model['path'])/'assets/physical-intelligence/libero/norm_stats.json'):model['normalization_sha256'],
            bridge['config']:bridge['config_sha256'],bridge['checkpoint']:bridge['weights_sha256']}
        if any(c['input_hashes'].get(p)!=h for p,h in required.items()):
            raise ValueError('Unbound source visual identity')
    for filename,stamp in c['input_hashes'].items():
        if verified.get(filename)!=stamp and file_digest(filename)!=stamp:
            raise ValueError('Admission asset changed: '+filename)
    states=read(sim['inputs'])['records']
    for r in cases:
        if type(r['initial']) is not int or not 0<=r['initial']<len(states):raise ValueError('Bad admission state')
        state=states[r['initial']]['state']
        if state['split']!='train' or r['state_key']!=state['pair_key']:
            raise ValueError('Admission uses declared training states only')
    return c


def request_for(path,index):
    c=read(path)
    from . import native_feature_admission
    if c.get('schema')==native_feature_admission.SCHEMA:return native_feature_admission.request_for(path,index)
    if index!=0:raise ValueError('Only one admission block')
    b=dict(c['blocks'][0]);b['addresses']=[dict(model=c['kind'],task=r['state_key'],
        initial=r['state_key'],view='numerical_admission_initial',stream='score',repeat=m,noise=f'member-{m}')
        for r in b['rows'] for m in (0,1)]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_spec(binding,key):
    s=binding['tasks'][key];req=s['request'];b=req['block']
    validate_campaign(req['campaign'])
    if (req!=request_for(binding['campaign'],b['index']) or s['limits']!=b['limits'] or s['addresses']!=b['addresses']
            or s['host']!=req['host'] or s['stage']!='formal'
            or key!=task_key(s['identity_experiment_id'],req['campaign_sha256'],b['addresses'])):
        raise ValueError('Admission task identity differs')
    return b['limits']


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_admission.py:run')
    if not callable(guard):raise ValueError('Live admission guard required')
    c=validate_campaign(request['campaign'])
    from . import native_feature_admission
    if c.get('schema')==native_feature_admission.SCHEMA:
        return native_feature_admission.run(request,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
    if request!=request_for(request['campaign'],0):raise ValueError('Changed admission request')
    import torch
    from .search_runtime import create_policy_runtime
    from .search_live import simulator
    from .local_search_rollout import input_identity
    from .legacy.fastwam_runtime import source_pixels
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sim=read(c['sim_config']);started=time.perf_counter();guard()
    runtime=create_policy_runtime(sim,output,c['limits']['score'])
    load_seconds=time.perf_counter()-started
    bridge=None
    if c['kind']=='source_visual':
        from AlphaBrain.model.pi05_features import load_standalone_siglip
        b=c['visual_bridge'];bridge=load_standalone_siglip(b['checkpoint'],
            expected_sha256=b['weights_sha256'],vision_config=read(b['config']),device='cuda:0')
    with np.load(sim['anchors'],allow_pickle=False) as z:noises=z['first_score_noise'].copy()
    rows=[]
    for case in request['block']['rows']:
        folder=output/f"initial-{case['initial']:03d}";folder.mkdir()
        guard()
        with simulator(c['sim_config'],sim,case['initial'],folder) as query:
            reply=query(dict(kind='query_pose'));example=reply['example']
            identity=input_identity(example)
            # Persist exact pixels before the first call, so numerical failures
            # can be diagnosed without recreating a renderer or VLA trajectory.
            np.savez(folder/'visual-input.npz',external=np.asarray(example['image'][0]),
                     wrist=np.asarray(example['image'][1]))
            write(folder/'input-identity.json',dict(input_sha256=identity,language=example['lang']))
            for m in (0,1):
                noise=noises[case['initial'],m]
                address=dict(state=case['state_key'],member=m,input_sha256=identity,phase='score')
                guard()
                if c['kind']=='fastwam_trace':
                    control=runtime.infer(example,noise,dict(address,variant='uninstrumented'),capture=False,instrument=False)
                    guard()
                    observed=runtime.infer(example,noise,dict(address,variant='instrumented'))
                    if not np.array_equal(control['full_actions'],observed['full_actions']):
                        raise ValueError('Instrumented FastWAM actions differ from native control')
                    arrays={k:v for k,v in observed.items() if isinstance(v,np.ndarray)}
                else:
                    observed=runtime.infer(example,noise,dict(address,variant='source_policy'))
                    guard()
                    with torch.inference_mode():features=bridge(source_pixels(example,device='cuda:0'))
                    arrays={k:features[k].float().cpu().numpy().reshape(-1) for k in ('external','wrist')}
                    for name in arrays:
                        if not np.array_equal(observed[name],arrays[name]):
                            np.savez(folder/f'failed-feature-{m}.npz',**arrays,
                                source_external=observed['external'],source_wrist=observed['wrist'])
                            raise ValueError('Standalone source visual features differ: '+name)
                np.savez(folder/f'member-{m}.npz',**arrays)
                row=dict(state=case['state_key'],member=m,input_sha256=identity,
                    seconds=observed['seconds'],score=observed['score'],exact=True)
                if c['kind']=='fastwam_trace':
                    row['control_seconds']=control['seconds']
                    row.update({k:observed[k] for k in ('peak_allocated_mib','peak_reserved_mib') if k in observed})
                rows.append(row)
    if runtime.calls!=c['limits']['score']:raise ValueError('Admission call accounting differs')
    write(output/'admission.json',dict(status='PASS_NUMERICAL_ONLY',kind=c['kind'],rows=rows,
        calls=runtime.calls,seconds=time.perf_counter()-started,load_seconds=load_seconds,
        training_updates=0,manipulation_actions=0))


def validate_block(request,output):
    from . import native_feature_admission
    if read(request['campaign']).get('schema')==native_feature_admission.SCHEMA:return native_feature_admission.validate_block(request,output)
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Changed admission receipt')
    for name,stamp in receipt['artifacts'].items():
        if file_digest(output/name)!=stamp:raise ValueError('Changed admission result')
    value=read(output/'admission.json')
    calls=[json.loads(x) for x in (output/'calls.jsonl').read_text().splitlines()]
    if (value['status']!='PASS_NUMERICAL_ONLY' or value['calls']!=request['block']['limits']['score']
            or len(calls)!=value['calls'] or [r['index'] for r in calls]!=list(range(1,len(calls)+1))
            or len(value['rows'])!=len(request['block']['rows'])*2
            or any(not r['exact'] for r in value['rows']) or value['training_updates'] or value['manipulation_actions']):
        raise ValueError('Incomplete numerical admission')
    return dict(calls=dict(score=value['calls'],execution=0),receipt_sha256=file_digest(output/'complete.json'))
