"""Bounded three-model pilot composed with standard authority/collectors.

No queue, SSH, simulator or model is started on import. Frozen legacy schemas
remain separate; this opt-in schema binds every actor and outcome-independent row.
"""
from pathlib import Path
from contextlib import ExitStack
import json
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest

SCHEMA='CONFIRMED_CHANGE_PILOT_V1'


def protocol_limits(path):
    """The original four updates stay frozen; extensions require a new binding."""
    p=read(path)
    if p.get('schema')=='CONFIRMED_CHANGE_SMALL_PILOT_PROTOCOL_V1' and p['maximum_updates']==4:
        return p,0,4,'G-matched-v4'
    if (p.get('schema')!='CONFIRMED_CHANGE_CONTINUATION_V1' or p.get('maximum_updates')!=16
            or p.get('initial_updates')!=4 or p.get('episodes_per_update')!=32
            or p.get('automatic_retry') is not False or p.get('automatic_mainline') is not False):
        raise ValueError('Explicit bounded continuation protocol required')
    parent=p['parent_protocol']
    if file_digest(parent['path'])!=parent['sha256']:raise ValueError('Parent pilot changed')
    old=read(parent['path'])
    if old['maximum_updates']!=4:raise ValueError('Wrong continuation boundary')
    for key in ('query_cap','query_cost','maximum_new_candidate_queries','terminal_confirmation_queries',
                'terminal_signed_gain','seeds','models','unchanged'):
        if p[key]!=old[key]:raise ValueError('Continuation changes the scientific method')
    if (file_digest(old['training_schedule'])!=p['parent_schedule_sha256']
            or read(p['training_schedule'])[:128]!=read(old['training_schedule'])
            or len(read(p['training_schedule']))!=512):raise ValueError('Continuation schedule prefix differs')
    return p,4,16,'G-matched-v16'


def campaign_limits(rows,sim,kind,family):
    warm=int(family=='groot_n15')
    if kind=='training':return dict(episodes=len(rows),score=16*len(rows)+warm,paired_execution=0,nominal_execution=0)
    if family=='pi05':
        step=5;horizons=read(sim['pilot_execution_plan'])['evaluation']['horizons']
    else:step=10;horizons=sim['operation_contract']['horizons']
    return dict(episodes=2*len(rows),score=16*len(rows)+warm,
        paired_execution=sum(2*((horizons[r['suite']]+step-1)//step) for r in rows),nominal_execution=0)


def validate_campaign(path):
    c=read(path)
    if (c.get('schema')!=SCHEMA or c['kind'] not in ('training','evaluation') or c['query_cap']!=8
            or c['query_cost']!=.01 or c['family'] not in ('pi05','fastwam','groot_n15')):
        raise ValueError('Explicit bounded confirmed-change pilot required')
    for p,h in c['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Pilot artifact changed: '+p)
    protocol,initial_updates,maximum_updates,comparison=protocol_limits(c['pilot_protocol'])
    registry=read(c['model_registry']);model=registry[c['family']]
    if (c['input_hashes'].get(c['model_registry'])!=file_digest(c['model_registry'])
            or c.get('native_policy')!=model['native_policy'] or c['observation_scale']!=model['observation_scale']):
        raise ValueError('Wrong pilot representation/source family')
    if c['input_hashes'].get(c['pilot_protocol'])!=file_digest(c['pilot_protocol']):
        raise ValueError('Unbound pilot training limit')
    required={c['pilot_protocol'],c['model_registry'],c['schedule'],c['confirmation_noise']}
    if c['kind']=='evaluation':required.update([c['score_noise'],c['execution_noise']])
    if not required.issubset(c['input_hashes']):raise ValueError('Unbound pilot input/noise')
    planned=read(c['schedule']);known={digest(r):r for r in planned}
    if len(known)!=len(planned) or c['input_hashes'].get(c['schedule'])!=file_digest(c['schedule']):raise ValueError('Unbound unique pilot schedule')
    total={k:0 for k in c['limits']};seen=set();versions={}
    for index,b in enumerate(c['blocks']):
        rows=b['rows'];sim=read(b['sim_config'])
        if b['index']!=index or not 1<=len(rows)<=2 or b['host'] not in c['execution_hosts']:
            raise ValueError('Explicit short pilot blocks required')
        if c['input_hashes'].get(b['sim_config'])!=file_digest(b['sim_config']):raise ValueError('Unbound runtime route')
        route=model['training_common'] if c['kind']=='training' else model['evaluation_common']
        expected_route=route.get('sim_configs',{}).get(rows[0]['suite'],route['sim_config'])
        checkpoint=sim['models'][sim['operation_model']]
        hashes=model['training_common'].get('model_hashes')
        expected_model=hashes[rows[0]['suite']] if hashes else model['training_common']['model_sha256']
        if (b['sim_config']!=expected_route or len({r['suite'] for r in rows})!=1
                or checkpoint['weights_sha256']!=expected_model or not {sim['inputs'],sim['anchors']}.issubset(c['input_hashes'])):
            raise ValueError('Pilot runtime/model/state route differs')
        if b['limits']!=campaign_limits(rows,sim,c['kind'],c['family']):raise ValueError('Pilot block budget differs')
        states=read(sim['inputs'])['records']
        for r in rows:
            key=digest(r)
            if key in seen or key not in known or states[r['initial']]['state']['pair_key']!=r['state_key']:
                raise ValueError('Repeated/unplanned/wrong-state pilot row')
            seen.add(key)
            if c['kind']=='training':
                if states[r['initial']]['state']['split']!='train' or r['episode']//32!=b['version'] or not initial_updates<=b['version']<maximum_updates:
                    raise ValueError('Only predeclared on-policy train batches admitted')
                versions.setdefault(b['version'],[]).append(r['episode'])
            elif states[r['initial']]['state']['split']!='validation':raise ValueError('Development-only pilot evaluation')
        actor=b['policy'] if c['kind']=='training' else None
        if actor and c['input_hashes'].get(actor)!=b['policy_sha256']:raise ValueError('Unbound current actor')
        for k in total:total[k]+=b['limits'][k]
    if any(sorted(v)!=list(range(u*32,(u+1)*32)) for u,v in versions.items()):raise ValueError('Complete 32-episode barriers required')
    if c['kind']=='training' and len(versions)!=1:raise ValueError('One on-policy actor version per collection barrier')
    if c['kind']=='training' and c['schedule']!=protocol['training_schedule']:
        raise ValueError('Wrong continuation training schedule')
    if c['kind']=='evaluation' and (seen!=set(known) or set(c['actors'])!={comparison,'confirmed_change'}
            or any(a['version']!=maximum_updates for a in c['actors'].values())):
        raise ValueError('Entire prespecified two-arm development panel required')
    if total!=c['limits']:raise ValueError('Pilot total budget differs')
    for method,spec in c.get('actors',{}).items():
        if c['input_hashes'].get(spec['path'])!=file_digest(spec['path']):raise ValueError('Unbound pilot comparison actor')
    return c


def request_for(path,index):
    c=read(path);b=c['blocks'][index]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),block=b,host=b['host'],stage='formal')


def load_actor(c,spec):
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
    torch.set_num_threads(2)
    payload=torch.load(spec['path'],map_location='cpu',weights_only=True)
    if payload.get('version')!=spec['version']:raise ValueError('Pilot actor version differs')
    if spec.get('pilot'):
        if (payload.get('pilot_protocol_sha256')!=file_digest(c['pilot_protocol'])
                or payload.get('family')!=c['family'] or payload.get('arm')!='confirmed_change'):
            raise ValueError('Wrong pilot reward/model provenance')
        model=read(c['model_registry'])[c['family']];tc=model['training_common']
        hashes=tc.get('model_hashes') or dict.fromkeys(['libero_spatial','libero_object','libero_goal','libero_10'],tc['model_sha256'])
        if (payload['model_hashes']!=hashes or payload['feature_spec_sha256']!=digest(c.get('native_policy'))
                or payload['initialization_sha256']!=file_digest(Path(model['old_run'])/'checkpoints/update-000.pt')
                or payload['training_episodes']!=spec['version']*32):raise ValueError('Wrong pilot actor origin')
    elif c.get('native_policy') is not None and payload.get('origin') is None:raise ValueError('Legacy native provenance missing')
    elif payload.get('origin') is not None:
        from AlphaBrain.research.dsol.data.native_training import publication_origin
        if payload['origin']!=publication_origin(read(spec['source_protocol']),file_digest(spec['source_protocol']),0):
            raise ValueError('Wrong legacy native actor provenance')
    policy=create_search_policy(c['native_policy']) if c.get('native_policy') is not None else ContinuousActorCritic(4608)
    policy.load_state_dict(payload['state_dict'],strict=True)
    policy.training_protocol_sha256=file_digest(c['pilot_protocol'])
    return policy.eval()


def run_training(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from .search_runtime import create_policy_runtime,bind_policy_task
    from .continuous_training import LiveEpisodes
    from AlphaBrain.research.dsol.selectors.continuous_parallel import seeded_episode
    from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file,pack
    from .paired_change_evidence import attach
    import torch
    c=validate_campaign(request['campaign']);b=request['block'];output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if request!=request_for(request['campaign'],b['index']):raise ValueError('Changed pilot collection request')
    policy=load_actor(c,dict(path=b['policy'],version=b['version'],pilot=True))
    sim=read(b['sim_config']);cap=b['limits']['score']
    with ExitStack() as stack:
        runtime=(create_policy_runtime(sim,output,cap) if runtime_cache is None else
            stack.enter_context(runtime_cache.lease(digest(sim),lambda:create_policy_runtime(sim,output,cap),lambda v:bind_policy_task(v,output,cap)))[0])
        backend=LiveEpisodes(dict(sim_config=b['sim_config'],trajectory_cap=cap,confirmation_noise=c['confirmation_noise'],
            **({'native_policy':c['native_policy']} if c.get('native_policy') is not None else {})),
            dict(actor=dict(score_scale=c['observation_scale'])),output,runtime=runtime,reward_scale=b['reward_scale'],simulator_pool=simulator_pool)
        records={}
        for row in b['rows']:
            guard();folder=output/f"query-{row['episode']:05d}"
            episode,metrics=seeded_episode(policy,backend,row,folder,version=b['version'],seed=0,source_kind='LIVE_RENDER_BATCH1_TWO_NOISES')
            episode=attach(episode,folder);metrics['pilot_protocol_sha256']=file_digest(c['pilot_protocol'])
            path=output/f"episode-{row['episode']:05d}.pt";atomic_file(path,lambda stream:torch.save(pack(dict(episode=episode,metrics=metrics)),stream))
            records[str(row['episode'])]=dict(path=str(path),sha256=file_digest(path))
        write(output/'collection.json',dict(status='COMPLETE',kind='rollout',records=records,calls=runtime.calls,
            request_sha256=digest(request),training_updates=0,manipulation_actions=0))


def run_evaluation(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from .search_runtime import create_policy_runtime,bind_policy_task
    from .search_live import simulator
    from .continuous_evaluation import EvaluationCalls,search,append_event,move_to_search_selection
    from .local_view_runtime import RenderedViewSource
    from .local_search_rollout import rollout_local_episode,rollout_fastwam_episode,rollout_groot_episode,input_identity
    from .search_domain import exact_input
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    c=validate_campaign(request['campaign']);b=request['block'];sim=read(b['sim_config']);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if request!=request_for(request['campaign'],b['index']):raise ValueError('Changed pilot evaluation request')
    cap=b['limits']['score']+b['limits']['paired_execution'];open_sim=simulator if simulator_pool is None else simulator_pool.session
    with np.load(c['score_noise'],allow_pickle=False) as z:noises=z['noise'].copy()
    with np.load(c['confirmation_noise'],allow_pickle=False) as z:held=z['noise'].copy()
    bank=ExplicitFlowNoiseBank(Path(c['execution_noise']));actors={m:load_actor(c,c['actors'][m]) for m in {r['method'] for r in b['rows']}}
    tasks=[];searches=[]
    with ExitStack() as stack:
        runtime=(create_policy_runtime(sim,output,cap) if runtime_cache is None else
            stack.enter_context(runtime_cache.lease(digest(sim),lambda:create_policy_runtime(sim,output,cap),lambda v:bind_policy_task(v,output,cap)))[0])
        calls=EvaluationCalls(runtime,output,dict(score=b['limits']['score'],execution=b['limits']['paired_execution']),guard=guard)
        for i,row in enumerate(b['rows']):
            folder=output/f'case-{i:04d}';folder.mkdir();calls.stage='score';values=noises[row['initial'],:2]
            with open_sim(b['sim_config'],sim,row['initial'],folder) as query:
                info,selected=search(dict(row,method='continuous_ppo'),query,calls,values,folder,actors[row['method']],c['observation_scale'],
                    native_policy=c.get('native_policy'),native_model_sha256=sim['models'][sim['operation_model']]['weights_sha256'],
                    confirmation_noises=held[row['initial']] if row['method']=='confirmed_change' else None)
                info.update(row,selected_input_sha256=input_identity(selected));write(folder/'search.json',info);searches.append(info)
            for repeat in (0,1):
                ep=folder/f'episode-{repeat}';ep.mkdir();calls.stage='execution'
                with open_sim(b['sim_config'],sim,row['initial'],ep) as query:
                    source=RenderedViewSource(query,calls,values,ep);move_to_search_selection(source,info)
                    example=query(dict(kind='begin_execution'))['example'];exact_input(selected,example)
                    rollout={'pi05':rollout_local_episode,'fastwam':rollout_fastwam_episode,'groot_n15':rollout_groot_episode}[c['family']]
                    extra=dict(plan=read(sim['pilot_execution_plan'])) if c['family']=='pi05' else dict(contract=sim['operation_contract'])
                    result,arrays=rollout(query,calls.infer,example,bank,row['state_key'],repeat,row,suite=row['suite'],
                        record=lambda event:append_event(ep/'events.jsonl',event),**extra)
                asset=ep/'trajectory.npz';np.savez_compressed(asset,**arrays);result.update(row,repeat=repeat,queries=info['queries'],selected_angles=info['selected_angles'],trajectory=str(asset));write(ep/'result.json',result);tasks.append(result)
        write(output/'results.json',dict(tasks=tasks,searches=searches));write(output/'status.json',dict(status='COMPLETE',calls=calls.counts,optimizer_updates=0))


def validate_evaluation_block(request,output):
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Pilot request identity changed')
    for name,h in receipt['artifacts'].items():
        if file_digest(output/name)!=h:raise ValueError('Pilot output changed')
    value=read(output/'results.json');status=read(output/'status.json');rows=request['block']['rows']
    expected={(r['trial_id'],s) for r in rows for s in (0,1)}
    if len(value['tasks'])!=len(expected) or {(r['trial_id'],r['repeat']) for r in value['tasks']}!=expected:raise ValueError('Pilot operation coverage differs')
    if (len(value['searches'])!=len(rows) or {r['trial_id'] for r in value['searches']}!={r['trial_id'] for r in rows}
            or any(not 1<=r['queries']<=8 for r in value['searches'])):raise ValueError('Pilot search coverage/budget differs')
    attempts=[json.loads(v) for v in (output/'model-attempts.jsonl').read_text().splitlines()]
    counts={s:sum(v['identity']['stage']==s for v in attempts) for s in ('score','execution')}
    warm=sum(v['identity'].get('purpose')=='native_graph_warmup' for v in attempts)
    if counts!=status['calls'] or counts['score']!=2*sum(r['queries'] for r in value['searches'])+warm or counts['execution']!=sum(t['replans'] for t in value['tasks']):raise ValueError('Pilot ledger differs')
    if (sum(1 for _ in (output/'calls.jsonl').open())!=sum(counts.values())
            or counts['score']>request['block']['limits']['score'] or counts['execution']>request['block']['limits']['paired_execution']):
        raise ValueError('Pilot runtime call count/budget differs')
    return dict(calls=counts,receipt_sha256=file_digest(output/'complete.json'))


def learner_step(config_path,*,records_path=None):
    """The normal CPU trainer and ordered on-policy barrier, pilot provenance."""
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Pilot learner is explicitly CPU-only')
    c=read(config_path)
    for p,h in c['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Pilot learner input changed: '+p)
    protocol,initial_updates,maximum_updates,_=protocol_limits(c['pilot_protocol'])
    if (c['schema']!='CONFIRMED_CHANGE_PILOT_LEARNER_V1'
            or not {c[k] for k in ('pilot_protocol','schedule','initial_checkpoint','initial_actor')}.issubset(c['input_hashes'])):
        raise ValueError('Bound pilot learner required')
    if c['schedule']!=protocol['training_schedule']:raise ValueError('Wrong learner schedule')
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig,training_binding
    from AlphaBrain.research.dsol.selectors.continuous_parallel import ingest
    from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file,fork_complete_stage
    from AlphaBrain.research.dsol.data import confirmed_reward
    torch.set_num_threads(2)
    policy=create_search_policy(c['native_policy']) if c.get('native_policy') is not None else ContinuousActorCritic(4608)
    policy.training_protocol_sha256=file_digest(c['pilot_protocol'])
    if c.get('native_policy') is not None:policy.operation_model_hashes=c['model_hashes']
    old=torch.load(c['initial_checkpoint'],map_location='cpu',weights_only=True)
    initial=torch.load(c['initial_actor'],map_location='cpu',weights_only=True)
    if old['update']!=0 or old['cursor']!=0 or initial['version']!=0 or any(not torch.equal(v,old['policy'][k]) for k,v in initial['state_dict'].items()):
        raise ValueError('Exact original untrained v0 policy/optimizer required')
    resume=records_path is not None;root=Path(c['output'])
    if root.exists()!=resume:raise ValueError('Explicit clean pilot init/update boundary required')
    config=ContinuousPPOConfig(updates=maximum_updates,seed=0)
    binding=dict(config_sha256=file_digest(config_path),source_kind='LIVE_RENDER_BATCH1_TWO_NOISES')
    if initial_updates and not resume:
        origin=protocol['resume'][c['family']]
        if c.get('continuation_origin')!=origin or c['input_hashes'].get(origin['files'])!=file_digest(origin['files']):
            raise ValueError('Unbound continuation origin')
        checkpoint=read(Path(origin['root'])/'latest.json')
        if checkpoint['sha256']!=origin['checkpoint_sha256'] or checkpoint['update']!=initial_updates:
            raise ValueError('Wrong continuation checkpoint')
        fork_complete_stage(origin['root'],root,
            training_binding(policy,config,read(c['schedule']),binding,confirmed_reward.config()),
            previous_schedule=read(read(protocol['parent_protocol']['path'])['training_schedule']),
            schedule=read(c['schedule']),source_files=read(origin['files']),expected_update=initial_updates)
        resume=True
    def no_live(*args,**kwargs):raise RuntimeError('CPU pilot learner cannot render or infer')
    with ContinuousPPO(policy,config,read(c['schedule']),no_live,root,
            binding=binding,
            reward_config=confirmed_reward.config(),resume=resume,warm_start=None if resume else old) as trainer:
        if records_path is not None:
            batch=read(records_path)
            if (batch['version']!=trainer.update or batch['policy_sha256']!=file_digest(root/'publications'/f'policy-{trainer.update:04d}.pt')
                    or batch['family']!=c['family'] or not initial_updates<=trainer.update<maximum_updates):raise ValueError('Stale/unbounded pilot update')
            ingest(trainer,{int(k):v for k,v in batch['records'].items()});trainer.optimize()
        folder=root/'publications';folder.mkdir(exist_ok=True);target=folder/f'policy-{trainer.update:04d}.pt'
        if target.exists():raise ValueError('Pilot published version already exists; no replay')
        payload=dict(state_dict=policy.state_dict(),version=trainer.update,pilot_protocol_sha256=file_digest(c['pilot_protocol']),
            family=c['family'],arm='confirmed_change',training_episodes=trainer.cursor,training_updates=trainer.update,
            model_hashes=c['model_hashes'],feature_spec_sha256=digest(c.get('native_policy')),
            initialization_sha256=file_digest(c['initial_checkpoint']),accepted_for_mainline=False)
        if initial_updates:
            origin=protocol['resume'][c['family']]
            payload['continuation_origin']=origin
            if records_path is None:
                parent_actor=origin['actor']
                if file_digest(parent_actor)!=origin['actor_sha256']:raise ValueError('Parent actor changed')
                previous=torch.load(parent_actor,map_location='cpu',weights_only=True)
                if (previous['pilot_protocol_sha256']!=protocol['parent_protocol']['sha256']
                        or previous['version']!=initial_updates
                        or previous['family']!=c['family']
                        or any(not torch.equal(v,previous['state_dict'][k]) for k,v in policy.state_dict().items())):
                    raise ValueError('Continuation actor weights differ from the frozen parent')
        atomic_file(target,lambda stream:torch.save(payload,stream))
        return dict(run='confirmed-'+c['family']+'-seed-0',version=trainer.update,
            episodes=trainer.cursor,status='COMPLETE' if trainer.update==maximum_updates else 'PAUSED_AT_UPDATE_BOUNDARY',
            policy=str(target),policy_sha256=file_digest(target),operation_model_updates=0,optimizer_device='cpu')
