"""Immutable, ledger-owned calibration and PPO collection blocks.

No optimizer or operation-model weight update runs in a collector. A published
CPU actor version is hash-bound before preparing each complete on-policy batch.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import math

from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.data import reward_pilot, domain_adaptation
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key

SCHEMA='CONTROLLED_CONTINUOUS_COLLECTION_V1'


def validate_operation_binding(c, sim, *, domain):
    """Target-model opt-in is restricted to the new domain protocol.

    This validates identities only; a caller still needs a separately admitted
    runtime and a live center grant. Historical native manifests stay strict.
    """
    from .search_runtime import selected_checkpoint
    if c['operation_model']!=sim.get('operation_model','canonical'):
        raise ValueError('Collection operation model differs from simulator')
    model=selected_checkpoint(sim,allow_native=True,allow_fastwam=domain)
    verified={}
    if model.get('implementation')=='fastwam_native':
        from .legacy.fastwam_runtime import RAW_OBSERVATION,validate_contract
        contract=validate_contract(model['runtime_contract'],model['runtime_contract_sha256'])
        required={model['runtime_contract']:model['runtime_contract_sha256'],**contract['input_hashes']}
        if (sim.get('raw_observation')!=RAW_OBSERVATION or model['path']!=contract['checkpoint']
                or any(c['input_hashes'].get(p)!=h for p,h in required.items())):
            raise ValueError('Unbound FastWAM collection runtime/raw observations')
        verified=required
        weights=model['path']
    else:
        if c['operation_model']!='native_reference' or model.get('implementation')!='openpi_native':
            raise ValueError('Unadmitted collection operation model')
        weights=str(Path(model['path'])/'model.safetensors')
    if model['weights_sha256']!=c['model_sha256'] or c['input_hashes'].get(weights)!=c['model_sha256']:
        raise ValueError('Operation model identity is not bound')
    return verified


def validate_campaign(path):
    if read(path).get('schema') in ('CONTROLLED_PREFIX_PILOT_V1','CONTROLLED_PREFIX_MAINLINE_V1'):
        from .prefix_pilot import validate_campaign as prefix_call
        return prefix_call(path)
    c=read(path)
    if c.get('schema')=='CONFIRMED_CHANGE_PILOT_V1':
        from .confirmed_pilot import validate_campaign as validate_pilot
        return validate_pilot(path)
    from AlphaBrain.research.dsol.data import native_training
    if c['schema']==native_training.COLLECTION_SCHEMA:
        from .native_collection import validate_campaign as validate_native
        return validate_native(path)
    domain=c['schema']==domain_adaptation.COLLECTION_SCHEMA
    adaptive=c['schema']==reward_pilot.COLLECTION_SCHEMA
    protocol=(domain_adaptation.from_campaign(c) if domain else
              reward_pilot.from_campaign(c) if adaptive else None)
    batch_size=8 if domain else 32
    if (adaptive or domain) and (c['kind']!='rollout' or c['schedules']!=protocol['schedules']):
        raise ValueError('Adaptation only permits the approved formal rollout schedule')
    if (c['schema'] not in (SCHEMA,reward_pilot.COLLECTION_SCHEMA,domain_adaptation.COLLECTION_SCHEMA) or c['kind'] not in ('calibration','rollout')
            or (not domain and c['operation_model']!='native_reference') or c['query_cost']!=.01
            or c['query_cap']!=8 or c['observation_scale']!=0.16535300100378683):
        raise ValueError('Unsupported controlled training contract')
    required={c['sim_config'],c['training_inputs'],c['training_noise'],c['protocol']}
    if c['kind']=='calibration':required.add(c['calibration_requests'])
    if c['kind']=='rollout':
        required.update(c['schedules'].values())
        if not (adaptive or domain):required.add(c['calibration_receipt'])
    if not required.issubset(c['input_hashes']):raise ValueError('Unbound training input')
    sim=read(c['sim_config']);states=read(c['training_inputs'])['records']
    verified=validate_operation_binding(c,sim,domain=domain)
    for p,h in c['input_hashes'].items():
        if verified.get(p)!=h and file_digest(p)!=h:raise ValueError('Changed training asset: '+p)
    if (sim['inputs']!=c['training_inputs'] or sim['anchors']!=c['training_noise']
            or (not domain and len(states)!=400)):
        raise ValueError('Wrong training simulator/noise collection')
    if domain:
        identity=read(protocol['domain'])
        if any(identity[k]!=c[k] for k in ('sim_config','training_inputs','training_noise','operation_model','model_sha256')):
            raise ValueError('Collection domain identity differs')
    if any(r['state'].get('split')!='train' for r in states):
        raise ValueError('Held-out states cannot enter calibration or training')
    seen=set();run_rows={};total={k:0 for k in ('episodes','score','paired_execution','nominal_execution')}
    for block in c['blocks']:
        n=len(block['rows']);expected=dict(episodes=0 if c['kind']=='calibration' else n,
                                         score=n*(2 if c['kind']=='calibration' else 16),
                                         paired_execution=0,nominal_execution=0)
        if block['limits']!=expected or not n:raise ValueError('Collection reservation differs')
        for k,v in expected.items():total[k]+=v
        for row in block['rows']:
            state=states[row['initial']]['state']
            if state['pair_key']!=row['state_key'] or state['task_id']!=row['task']:
                raise ValueError('Collection state identity differs')
            key=(block.get('run'),row.get('episode',row.get('address')))
            if key in seen:raise ValueError('Duplicate collection address')
            seen.add(key)
        if c['kind']=='rollout':
            arm,seed,version=block['arm'],block['seed'],block['version']
            if (arm not in (protocol['arms'] if adaptive or domain else ('A','B')) or seed not in (0,1,2)
                    or type(version) is not int or not 0<=version<(protocol['updates'] if adaptive or domain else 256)):
                raise ValueError('Unknown arm, seed or policy version')
            if block['run']!=f'{arm}-seed-{seed}':raise ValueError('Run identity differs')
            if c['input_hashes'].get(block['policy'])!=block['policy_sha256']:
                raise ValueError('Actor must be hash-bound before batch dispatch')
            if adaptive or domain:
                provisional=c['observation_scale'] if domain else reward_pilot.FIXED_SCALE
                if block['reward_scale']!=provisional or not c['settle_reward_at_learner']:
                    raise ValueError('Raw collector reward must be settled only at the complete learner barrier')
            else:
                calibration=read(c['calibration_receipt'])
                expected_scale=c['observation_scale'] if arm=='A' else calibration['scale']
                if (calibration.get('schema')!='FIXED_MODEL_REWARD_SCALE_V1' or calibration.get('count')!=800
                    or calibration.get('model_sha256')!=c['model_sha256']
                    or not calibration['fixed_during_training'] or not math.isfinite(expected_scale)
                    or expected_scale<=0 or block['reward_scale']!=expected_scale):
                    raise ValueError('Reward calibration changed')
            schedule=read(c['schedules'][str(seed)])
            if any(row!=schedule[row['episode']] or row['episode']//batch_size!=version for row in block['rows']):
                raise ValueError('Episode outside immutable on-policy batch')
            run_rows.setdefault((block['run'],version),[]).extend(row['episode'] for row in block['rows'])
    if c['kind']=='rollout' and (not run_rows or any(sorted(v)!=list(range(version*batch_size,(version+1)*batch_size)) for (_,version),v in run_rows.items())):
        raise ValueError(f'Complete {batch_size}-episode policy barriers required')
    if c['kind']=='calibration':
        expected=read(c['calibration_requests'])
        actual=[row for b in c['blocks'] for row in b['rows']]
        if actual!=expected or len(actual)!=800 or any(row['split']!='train' for row in actual):
            raise ValueError('Calibration must cover the 800 fixed training initial queries')
    if total!=c['limits']:raise ValueError('Campaign budget coverage differs')
    recovery=c.get('collection_recovery')
    if recovery:
        if not adaptive or c['input_hashes'].get(recovery)!=file_digest(recovery):
            raise ValueError('Unbound collection recovery')
        proof=read(recovery)
        if (proof['status']!='VERIFIED_STOPPED_PARTIAL_COLLECTION' or not proof['all_workers_stopped']
                or proof['automatic_retry'] or proof['held_failed_phase_cap']!=4608
                or any(b['version']!=proof['resume_update'] for b in c['blocks'])):
            raise ValueError('Explicit stopped-worker and retained-budget proof required')
        for key,ref in proof['records'].items():
            if key not in {f"{b['run']}:{r['episode']}" for b in c['blocks'] for r in b['rows']}:
                raise ValueError('Recovered episode outside complete batch')
            if c['input_hashes'].get(ref['path'])!=ref['sha256']:
                raise ValueError('Unbound recovered episode')
    return c


def request_for(path,index):
    c=read(path);b=c['blocks'][index]
    if b['index']!=index:raise ValueError('Unordered collection block')
    return dict(campaign=str(path),campaign_sha256=file_digest(path),block=b,host=b['host'],stage='formal')


def validate_spec(binding,key):
    s=binding['tasks'][key];r=s['request'];b=r['block']
    if (r!=request_for(binding['campaign'],b['index']) or s['limits']!=b['limits']
            or s['addresses']!=b['addresses'] or s['host']!=b['host'] or s['stage']!='formal'
            or key!=task_key(s['identity_experiment_id'],r['campaign_sha256'],b['addresses'])):
        raise ValueError('Unbound training scientific request')
    return b['limits']


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_training.py:run')
    if read(request['campaign']).get('schema') in ('CONTROLLED_PREFIX_PILOT_V1','CONTROLLED_PREFIX_MAINLINE_V1'):
        from .prefix_pilot import run as prefix_run
        return prefix_run(request,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
    if read(request['campaign']).get('schema')=='CONFIRMED_CHANGE_PILOT_V1':
        from .confirmed_pilot import run_training
        return run_training(request,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
    c=validate_campaign(request['campaign']);b=request['block']
    if request!=request_for(request['campaign'],b['index']):raise ValueError('Changed collection request')
    if c.get('native_policy') is not None:
        # Different suites may use different frozen operation weights while
        # sharing one on-policy search update. Never mix their VLA instances.
        c=dict(c,sim_config=b['sim_config'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    from .search_runtime import create_policy_runtime,bind_policy_task
    sim=read(c['sim_config']);cap=b['limits']['score']
    with ExitStack() as stack:
        if runtime_cache is None:runtime=create_policy_runtime(sim,output,cap)
        else:
            runtime,_=stack.enter_context(runtime_cache.lease(digest(sim),
                lambda:create_policy_runtime(sim,output,cap),lambda value:bind_policy_task(value,output,cap)))
        if c['kind']=='calibration':
            records=_calibrate(c,b,output,runtime,guard,simulator_pool)
        else:
            records=_collect(c,b,output,runtime,simulator_pool)
        # A fully recovered block performs zero new calls, but still has an
        # explicit empty attempt ledger rather than a missing accounting file.
        if not (output/'model-attempts.jsonl').exists() and c.get('collection_recovery'):
            (output/'model-attempts.jsonl').touch(exist_ok=False)
        attempts=[json.loads(line) for line in (output/'model-attempts.jsonl').read_text().splitlines()]
        calls=len(attempts)
        if calls>cap or calls!=runtime.calls:raise ValueError('Collector accounting differs')
        write(output/'collection.json',dict(status='COMPLETE',kind=c['kind'],records=records,
              calls=calls,request_sha256=digest(request),training_updates=0,manipulation_actions=0))


def _calibrate(c,b,output,runtime,guard,pool):
    import numpy as np
    from .continuous_evaluation import EvaluationCalls
    from .local_view_runtime import RenderedViewSource
    from .search_live import simulator
    calls=EvaluationCalls(runtime,output,dict(score=b['limits']['score'],execution=0),guard=guard)
    sim=read(c['sim_config']);open_sim=simulator if pool is None else pool.session
    with np.load(c['training_noise'],allow_pickle=False) as archive:noises=archive['first_score_noise'].copy()
    records=[]
    for index,row in enumerate(b['rows']):
        folder=output/f'initial-{index:04d}';folder.mkdir()
        calls.stage='score';calls.identity=dict(address=row['address'],initial=row['initial'])
        with open_sim(c['sim_config'],sim,row['initial'],folder) as query:
            source=RenderedViewSource(query,calls,noises[row['initial'],:2],folder)
            source.move_angles(row['angles_deg']);source.query()
            records.append(dict(address=row['address'],split='train',query_kind='initial',
                model_sha256=c['model_sha256'],score_members=[float(v['score']) for v in source.last_values]))
    return records


def _collect(c,b,output,runtime,pool):
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_parallel import seeded_episode
    from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file,pack
    from .continuous_training import LiveEpisodes
    torch.set_num_threads(2)
    native=c.get('native_policy')
    if native is not None:
        from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
        policy=create_search_policy(native).eval()
        policy.training_protocol_sha256=file_digest(c['protocol'])
    else:policy=ContinuousActorCritic(4608).eval()
    payload=torch.load(b['policy'],map_location='cpu',weights_only=True)
    if payload['version']!=b['version']:raise ValueError('Stale published policy')
    if native is not None:
        from AlphaBrain.research.dsol.selectors.search_policy_factory import validate_publication
        from AlphaBrain.research.dsol.data.native_training import publication_origin
        validate_publication(policy,payload,version=b['version'],expected_origin=publication_origin(
            read(c['protocol']),file_digest(c['protocol']),b['seed']))
    else:policy.load_state_dict(payload['state_dict'],strict=True)
    backend=LiveEpisodes(dict(sim_config=c['sim_config'],trajectory_cap=b['limits']['score'],**({'native_policy':native} if native is not None else {})),
        dict(actor=dict(score_scale=c['observation_scale'])),output,runtime=runtime,
        reward_scale=b['reward_scale'],simulator_pool=pool)
    records={}
    recovered=read(c['collection_recovery'])['records'] if c.get('collection_recovery') else {}
    for row in b['rows']:
        ref=recovered.get(f"{b['run']}:{row['episode']}")
        if ref:
            from AlphaBrain.research.dsol.selectors.continuous_parallel import validate_rollout
            from AlphaBrain.research.dsol.data.online_reward import validate_raw
            from AlphaBrain.research.dsol.data.continuous_training_store import numpy_tree
            if file_digest(ref['path'])!=ref['sha256']:raise ValueError('Recovered rollout changed')
            payload=torch.load(ref['path'],map_location='cpu',weights_only=True)
            validate_rollout(policy,payload,row,b['version'])
            validate_raw(numpy_tree(payload['episode']),payload['metrics'])
            records[str(row['episode'])]=ref
            continue
        episode,metrics=seeded_episode(policy,backend,row,output/f"query-{row['episode']:05d}",
            version=b['version'],seed=b['seed'],source_kind='LIVE_RENDER_BATCH1_TWO_NOISES')
        if native is not None or c['schema'] in (reward_pilot.COLLECTION_SCHEMA,domain_adaptation.COLLECTION_SCHEMA):
            metrics['reward_status']='PROVISIONAL_COLLECTOR_ONLY_SETTLED_AT_COMPLETE_PPO_BATCH'
        p=output/f"episode-{row['episode']:05d}.pt"
        atomic_file(p,lambda stream:torch.save(pack(dict(episode=episode,metrics=metrics)),stream))
        records[str(row['episode'])]=dict(path=str(p),sha256=file_digest(p))
    return records


def validate_block(request,output):
    if read(request['campaign']).get('schema') in ('CONTROLLED_PREFIX_PILOT_V1','CONTROLLED_PREFIX_MAINLINE_V1'):
        from .prefix_pilot import validate_block as prefix_call
        return prefix_call(request,output)
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):
        raise ValueError('Collection completion identity changed')
    for name,expected in receipt['artifacts'].items():
        if file_digest(output/name)!=expected:raise ValueError('Collection artifact changed')
    value=read(output/'collection.json')
    if value['request_sha256']!=digest(request):raise ValueError('Collection request changed')
    n=sum(1 for _ in (output/'model-attempts.jsonl').open())
    rows=request['block']['rows']
    if (value['status']!='COMPLETE' or value['calls']!=n or n>request['block']['limits']['score']
            or value['training_updates']!=0 or value['manipulation_actions']!=0):
        raise ValueError('Collection accounting changed')
    if value['kind']=='calibration':
        if n!=len(rows)*2 or [v['address'] for v in value['records']]!=[r['address'] for r in rows]:
            raise ValueError('Calibration coverage differs')
    elif value['kind']=='rollout':
        if set(value['records'])!={str(r['episode']) for r in rows}:raise ValueError('Rollout coverage differs')
    else:raise ValueError('Unknown collection result')
    return dict(calls=dict(score=n,execution=0),receipt_sha256=file_digest(output/'complete.json'))
