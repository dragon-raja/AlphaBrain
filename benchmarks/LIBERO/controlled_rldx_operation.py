"""RLDX engineering trajectories and bounded development selection diagnostics.

No PPO, no outcome-dependent retry. Selection uses two fixed scoring members;
operation seeds depend on state/repeat/replan, never selected view or host.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest,task_key
from AlphaBrain.research.dsol.data.operation_contract import rldx_contract,validate_rldx
from .rldx_runtime import validate_contract,LiberoHistory

SCHEMA='CONTROLLED_RLDX_OPERATION_V1'


def seed_for(state,repeat,replan):
    b=digest(dict(schema='RLDX_NATIVE_OPERATION_SEED_V1',state=state,repeat=repeat,replan=replan)).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'little')%2**63


def input_stamp(history):
    obs=history.observation()
    return digest(dict(language=history.language,modalities={k:{name:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                     for name,v in obs[k].items()} for k in ('video','state')}))


def limits(rows,kind):
    if kind=='trajectory':return dict(episodes=2*len(rows),score=0,paired_execution=180*len(rows),nominal_execution=0)
    return dict(episodes=8*len(rows),score=54*len(rows),paired_execution=720*len(rows),nominal_execution=0)


def validate_campaign(path):
    c=read(path)
    if c['schema']=='CONTROLLED_RLDX_SPATIAL_COMPLETION_V1':
        from .rldx_spatial_panel import validate_campaign
        return validate_campaign(path)
    if c['schema']!=SCHEMA or c['kind'] not in ('trajectory','selection') or not 1<=len(c['blocks'])<=16:
        raise ValueError('Finite RLDX engineering/development diagnostic required')
    runtime=validate_contract(c['runtime_contract'],c['runtime_contract_sha256']);sim=read(c['sim_config'])
    validate_rldx(sim['operation_contract'])
    if sim['operation_contract']!=rldx_contract(c['runtime_contract_sha256']):raise ValueError('Different native operation contract')
    required={c['runtime_contract']:c['runtime_contract_sha256'],**runtime['input_hashes']}
    required.update({p:file_digest(p) for p in (c['sim_config'],sim['inputs'],sim['anchors'])})
    if c.get('reference_trajectory'):
        required[c['reference_trajectory']]=c['reference_trajectory_sha256']
        if file_digest(c['reference_trajectory'])!=c['reference_trajectory_sha256']:
            raise ValueError('Native full-trajectory reference changed')
    if c['input_hashes']!=required:raise ValueError('Unbound or changed diagnostic input')
    if (c.get('score_seeds')!=[20311002,20311003]):raise ValueError('Fixed independent scoring noise members required')
    states=read(sim['inputs'])['records'];seen=[]
    for i,b in enumerate(c['blocks']):
        if b['index']!=i or b['host'] not in c.get('hosts',['fresh']) or len(b['rows'])!=1 or b['limits']!=limits(b['rows'],c['kind']):
            raise ValueError('One explicit state per bounded Fresh block')
        row=b['rows'][0];idx=row['initial']
        if type(idx) is not int or not 0<=idx<len(states):raise ValueError('Invalid initial')
        state=states[idx]['state'];seen.append(row['state_key'])
        if (state['pair_key']!=row['state_key'] or state['suite']!=row['suite']
                or state['split']!=('train' if c['kind']=='trajectory' else 'validation')
                or state['initialization']['settle_steps']!=10):raise ValueError('State/split/initialization differs')
        if c['kind']=='selection':
            if (len(row['views'])!=27 or row['views'][0]!=[0.,0.] or row['start_views']!=[23,24]
                    or len(set(map(tuple,row['views'])))!=27):raise ValueError('Fixed global development grid required')
            from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
            for angles in row['views']:PanTiltMount(np.eye(4)).angles(angles)
    if len(set(seen))!=len(seen) or c['limits']!={k:sum(b['limits'][k] for b in c['blocks']) for k in c['limits']}:
        raise ValueError('Duplicate state or incomplete budget')
    if c['kind']=='trajectory' and (len(seen)!=1 or c.get('trajectory_proof') is not None):
        raise ValueError('One original engineering trajectory pair required')
    if c['kind']=='selection':
        proof=read(c['trajectory_proof'])
        if proof['status']!='PASS_NATIVE_INSTRUMENTED_TRAJECTORY' or file_digest(c['trajectory_proof'])!=c['trajectory_proof_sha256']:
            raise ValueError('Verified full trajectory required')
    return c


def request_for(path,index):
    c=read(path)
    if c['schema']=='CONTROLLED_RLDX_SPATIAL_COMPLETION_V1':
        from .rldx_spatial_panel import request_for
        return request_for(path,index)
    b=dict(c['blocks'][index]);row=b['rows'][0]
    variants=['native','trace'] if c['kind']=='trajectory' else ['stay-23','stay-24','reference','minimum-p3']
    addresses=[dict(model='rldx',task=row['state_key'],initial=row['state_key'],view=v,
                    stream='execution',repeat=m,noise=f'execution-{m}') for v in variants
               for m in ([0] if c['kind']=='trajectory' else [0,1])]
    if c['kind']=='selection':
        addresses += [dict(model='rldx',task=row['state_key'],initial=row['state_key'],view=f'grid-{i}',
                           stream='score',repeat=m,noise=f'score-{m}') for i in range(27) for m in (0,1)]
    b['addresses']=addresses
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_spec(binding,key):
    s=binding['tasks'][key];req=s['request'];b=req['block']
    if read(req['campaign'])['schema']=='CONTROLLED_RLDX_SPATIAL_COMPLETION_V1':
        from .rldx_spatial_panel import validate_spec
        return validate_spec(binding,key)
    validate_campaign(req['campaign'])
    if (req!=request_for(binding['campaign'],b['index']) or s['limits']!=b['limits']
            or s['addresses']!=b['addresses'] or s['host']!=req['host'] or s['stage']!='formal'
            or key!=task_key(s['identity_experiment_id'],req['campaign_sha256'],b['addresses'])):
        raise ValueError('RLDX operation task identity differs')
    return b['limits']


def move_to(query,angles):
    from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
    info=query(dict(kind='continuous_info'));mount=PanTiltMount(info['reference'])
    physical=info['receipt']['physical_state_id']
    for point in mount.route([0.,0.],angles):
        receipt=query(dict(kind='move_pose',camera_to_world=mount.pose(point)))['receipt']
        if receipt['physical_state_id']!=physical or receipt['settled'] is not True:raise ValueError('Camera changed physics or is unsettled')
    reply=query(dict(kind='query_pose'));receipt=reply['receipt']
    if (receipt['physical_state_id']!=physical or receipt['settled'] is not True
            or not np.allclose(receipt['camera_to_world'],mount.pose(angles),atol=1e-9,rtol=0)):
        raise ValueError('Wrong settled fixed-pose input')
    return reply['example']


def rollout(query,runtime,example,row,repeat,variant,*,guard,instrument=True):
    history=LiberoHistory(example);steps=0;success=False;normalized=[];actions=[];inputs=[];physics=[];seeds=[]
    started=time.monotonic()
    for replan in range(90):
        guard();seed=seed_for(row['state_key'],repeat,replan);seeds.append(seed);inputs.append(input_stamp(history))
        value=runtime.infer(history,seed,dict(phase='execution',state=row['state_key'],repeat=repeat,replan=replan,variant=variant),instrument=instrument)
        normalized.append(value['normalized_actions']);executed=[]
        for action in value['actions'][:8]:
            guard();reply=query(dict(kind='step',actions=action[None]))
            if reply['executed']!=1:raise ValueError('Expected exactly one native action and real history frame')
            steps+=1;executed.append(action);physics.append(reply['physics']);history.append(reply['example'])
            success=reply['success']
            if success:break
        actions.extend(executed)
        if success:break
    result=dict(state_key=row['state_key'],repeat=repeat,variant=variant,steps=steps,success=bool(success),
                replans=len(seeds),horizon=720,seconds=time.monotonic()-started)
    arrays=dict(normalized_actions=np.stack(normalized),actions=np.asarray(actions),inputs=np.asarray(inputs),
                physics=np.asarray(physics),seeds=np.asarray(seeds,dtype=np.int64))
    return result,arrays


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_rldx_operation.py:run')
    if not callable(guard):raise ValueError('Live continuation required')
    c=validate_campaign(request['campaign'])
    if c['schema']=='CONTROLLED_RLDX_SPATIAL_COMPLETION_V1':
        from .rldx_spatial_panel import run
        return run(request,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
    if request!=request_for(request['campaign'],request['block']['index']):raise ValueError('Changed operation request')
    from .rldx_runtime import RLDXRuntime
    from .search_live import simulator
    open_sim=simulator if simulator_pool is None else simulator_pool.session
    output=Path(output);output.mkdir(parents=True,exist_ok=False);guard()
    runtime=RLDXRuntime(c['runtime_contract'],c['runtime_contract_sha256'],output,
                        request['block']['limits']['score']+request['block']['limits']['paired_execution'])
    sim=read(c['sim_config']);row=request['block']['rows'][0];scores=[];tasks=[];trajectories=[]
    if c['kind']=='selection':
        for view,angles in enumerate(row['views']):
            folder=output/f'grid-{view:02d}';folder.mkdir();guard()
            with open_sim(c['sim_config'],sim,row['initial'],folder) as query:
                example=move_to(query,angles)
            history=LiberoHistory(example);values=[]
            for m,seed in enumerate(c['score_seeds']):
                guard();value=runtime.infer(history,seed,dict(phase='score',state=row['state_key'],view=view,member=m))
                np.savez_compressed(folder/f'member-{m}.npz',**{k:v for k,v in value.items() if isinstance(v,np.ndarray)})
                values.append(value['scores'])
            score=dict(view=view,angles_deg=angles,input_sha256=input_stamp(history),scores=values)
            scores.append(score);write(folder/'score.json',score)
        selected=min(range(1,27),key=lambda i:(np.mean([v['3'] for v in scores[i]['scores']]),i))
        variants=[('reference',0),('stay-23',23),('stay-24',24),('minimum-p3',selected)]
        write(output/'selection.json',dict(selected=selected,rule='minimum mean P3 among perturbed 26; reference excluded',scores=scores))
    else:variants=[('native',0),('trace',0)]
    for variant,view in variants:
        for repeat in ([0] if c['kind']=='trajectory' else [0,1]):
            folder=output/f'{variant}-repeat-{repeat}';folder.mkdir();guard()
            with open_sim(c['sim_config'],sim,row['initial'],folder) as query:
                selected_example=move_to(query,[0.,0.] if c['kind']=='trajectory' else row['views'][view])
                example=query(dict(kind='begin_execution'))['example']
                if input_stamp(LiberoHistory(example))!=input_stamp(LiberoHistory(selected_example)):
                    raise ValueError('Execution initial sensors changed after selection')
                result,arrays=rollout(query,runtime,example,row,repeat,variant,guard=guard,instrument=variant!='native')
            np.savez_compressed(folder/'trajectory.npz',**arrays);result.update(view=view)
            write(folder/'result.json',result);tasks.append(result);trajectories.append(arrays)
            write(output/'progress.json',dict(tasks=len(tasks),score_views=len(scores),calls=runtime.calls))
    status='COMPLETE_DEVELOPMENT_SELECTION'
    if c['kind']=='trajectory':
        if tasks[0]['success']!=tasks[1]['success'] or any(not np.array_equal(trajectories[0][k],trajectories[1][k]) for k in trajectories[0]):
            raise ValueError('Native vs instrumented full trajectory differs')
        if c.get('reference_trajectory'):
            with np.load(c['reference_trajectory'],allow_pickle=False) as ref:
                if set(ref.files)!=set(trajectories[0]) or any(not np.array_equal(ref[k],trajectories[0][k]) for k in ref.files):
                    raise ValueError('Resident/cross-node trajectory differs from immutable native reference')
        status='PASS_NATIVE_INSTRUMENTED_TRAJECTORY'
    write(output/'results.json',dict(status=status,tasks=tasks,scores=scores,training_updates=0))


def validate_block(request,output):
    if read(request['campaign'])['schema']=='CONTROLLED_RLDX_SPATIAL_COMPLETION_V1':
        from .rldx_spatial_panel import validate_block
        return validate_block(request,output)
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Changed operation receipt')
    for p,h in receipt['artifacts'].items():
        if file_digest(output/p)!=h:raise ValueError('Changed operation artifact')
    result=read(output/'results.json');c=read(request['campaign']);cap=request['block']['limits']
    calls=[json.loads(x) for x in (output/'calls.jsonl').read_text().splitlines()]
    attempts=[json.loads(x) for x in (output/'model-attempts.jsonl').read_text().splitlines()]
    counts={k:sum(x['identity']['phase']==k for x in calls) for k in ('score','execution')}
    if (len(attempts)!=len(calls) or any(x['index']!=i+1 for i,x in enumerate(calls))
            or counts['score']!=cap['score'] or counts['execution']>cap['paired_execution']
            or len(result['tasks'])!=cap['episodes'] or result['training_updates']
            or sum(t['replans'] for t in result['tasks'])!=counts['execution']):
        raise ValueError('Incomplete or unknown RLDX operation budget')
    expected='PASS_NATIVE_INSTRUMENTED_TRAJECTORY' if c['kind']=='trajectory' else 'COMPLETE_DEVELOPMENT_SELECTION'
    if result['status']!=expected:raise ValueError('Failed trajectory or incomplete selection')
    return dict(calls=counts,receipt_sha256=file_digest(output/'complete.json'))
