"""On-policy prefix campaigns, composed with standard collectors and authority.

No solver/operation model is trained. Prefix, actor origin, schedules and budget
are immutable. Unknown calls never retry. The two arms do not share rollouts.
"""
from pathlib import Path
from contextlib import ExitStack
import os
import json
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.data.score_prefix import validate as validate_score
from AlphaBrain.research.dsol.execution.manifest import file_digest, digest

SCHEMA='CONTROLLED_PREFIX_PILOT_V1'
PROTOCOL='PREFIX_PPO_COMPARISON_V1'
MAINLINE='CONTROLLED_PREFIX_MAINLINE_V1'
MAINLINE_PROTOCOL='PREFIX_MAINLINE_V1'
SCHEMAS=(SCHEMA,MAINLINE)
CORE_PROTOCOL='UNIFIED_CORE_PILOT_V1'


def cell_for(c):
    p=read(c['protocol'])
    mainline=p['schema']==MAINLINE_PROTOCOL
    if p['schema']==CORE_PROTOCOL:
        from AlphaBrain.research.dsol.data.core_pilot import validate_protocol
        validate_protocol(p)
    elif mainline:
        from AlphaBrain.research.dsol.data.prefix_protocol import validate_mainline
        validate_mainline(p)
    elif (p['schema']!=PROTOCOL or p['updates']!=8 or p['episodes_per_update']!=32 or p['seeds']!=[0]
            or p['maximum_model_calls']!=66000 or p['query_cap']!=8 or p['query_cost']!=.01):
        raise ValueError('Explicit approved prefix protocol required')
    if c['input_hashes'].get(c['protocol'])!=file_digest(c['protocol']):raise ValueError('Unbound prefix protocol')
    authorization=read(p['authorization'])
    if (c['input_hashes'].get(p['authorization'])!=file_digest(p['authorization'])
            or authorization['total_model_call_limit']!=p['maximum_model_calls'] or authorization['status']!='EXPLICIT_USER_APPROVAL'):
        raise ValueError('New bounded budget is not approved')
    cell=p['cells'][c['cell']];validate_score(cell['score_contract'])
    if cell.get('task_condition') is not None:
        binding=cell['task_condition']
        if (c['input_hashes'].get(binding['cache'])!=binding['cache_sha256']
                or file_digest(binding['cache'])!=binding['cache_sha256']):
            raise ValueError('Unbound or changed task encoder cache')
    if p['schema']!=CORE_PROTOCOL and cell.get('core_spec') is not None:
        raise ValueError('A core change requires its separately authorized protocol')
    return p,cell


def limits(rows, sim, kind, family, *, stay=False,collector_smoke=False):
    warm=int(family=='groot_n15')
    if kind=='admission':
        n=(90 if family=='rldx' else (read(sim['pilot_execution_plan'])['evaluation']['horizons'][rows[0]['suite']]+4)//5 if family=='pi05'
            else (sim['operation_contract']['horizons'][rows[0]['suite']]+9)//10)
        return dict(episodes=0,score=2*n+3+warm+(16 if family=='rldx' or collector_smoke else 0),paired_execution=0,nominal_execution=0)
    if kind=='training':return dict(episodes=len(rows),score=16*len(rows)+warm,paired_execution=0,nominal_execution=0)
    if family=='rldx':cost=90*2*len(rows)
    else:
        horizons=(read(sim['pilot_execution_plan'])['evaluation']['horizons'] if family=='pi05' else sim['operation_contract']['horizons'])
        step=5 if family=='pi05' else 10
        cost=sum(2*((horizons[r['suite']]+step-1)//step) for r in rows)
    return dict(episodes=2*len(rows),score=0 if stay else 16*len(rows)+warm,paired_execution=cost,nominal_execution=0)


def validate_campaign(path):
    c=read(path);p,cell=cell_for(c)
    mainline=p['schema'] in (MAINLINE_PROTOCOL,CORE_PROTOCOL)
    if c['schema']!=(MAINLINE if mainline else SCHEMA) or c['kind'] not in ('training','evaluation','admission'):raise ValueError('Explicit prefix stage required')
    for f,h in c['input_hashes'].items():
        if file_digest(f)!=h:raise ValueError('Changed prefix input: '+f)
    if c.get('native_policy')!=cell['native_policy']:raise ValueError('Native representation differs')
    if c['kind']!='admission' and not mainline:
        proof_path=c['admission_proof']
        if c['input_hashes'].get(proof_path)!=file_digest(proof_path):raise ValueError('Unbound actual-entry admission')
        proof=read(proof_path)
        if (proof['status']!='PASS_PREFIX_NUMERICAL_AND_TRAJECTORY' or proof['family']!=cell['family']
                or proof['source_identity']!=c['source_identity']
                or proof['model_hashes']!=cell['model_hashes']):raise ValueError('Wrong or stale actual-entry admission')
    scheduled=read(c['schedule']);seen=[];total=dict(episodes=0,score=0,paired_execution=0,nominal_execution=0)
    versions=set()
    for i,b in enumerate(c['blocks']):
        if mainline and c['kind']!='admission':
            from AlphaBrain.research.dsol.data.prefix_protocol import validate_admission
            validate_admission(c,cell,b)
        sim=read(b['sim_config']);model=sim['models'][sim['operation_model']]
        if (i!=b['index'] or b['host'] not in c['hosts'] or not 1<=len(b['rows'])<=2
                or sim.get('score_contract')!=cell['score_contract'] or sim.get('native_features')!=cell.get('feature_producer')
                or model['weights_sha256']!=cell['model_hashes'][b['rows'][0]['suite']]):
            raise ValueError('Changed host, score axes, features or native model')
        required={b['sim_config'],sim['inputs'],sim['anchors'],c['schedule'],c['protocol'],p['authorization']}
        if not required.issubset(c['input_hashes']):raise ValueError('Unbound simulator/population')
        stage='training' if c['kind'] in ('training','admission') else 'evaluation'
        if b['sim_config']!=cell[stage+'_routes'][b['rows'][0]['suite']]:raise ValueError('Wrong native route')
        states=read(sim['inputs'])['records']
        if b['limits']!=limits(b['rows'],sim,c['kind'],cell['family'],stay=c.get('stay',False),collector_smoke=mainline):
            raise ValueError('Changed stage budget')
        for r in b['rows']:
            state=states[r['initial']]['state']
            if r not in scheduled or state['pair_key']!=r['state_key'] or r['suite']!=state['suite']:
                raise ValueError('Unplanned, repeated or wrong-state input')
            if c['kind']=='training':
                if (state['split']!='train' or r['episode']//32!=b['version'] or not 0<=b['version']<p['updates']
                        or (mainline and b['seed']!=cell['seed'])
                        or c['schedule']!=cell['training_schedule']):raise ValueError('Wrong on-policy training boundary')
                versions.add(b['version'])
            elif c['kind']=='evaluation':
                if p['schema']==CORE_PROTOCOL:
                    from AlphaBrain.research.dsol.data.core_pilot import validate_evaluation
                    validate_evaluation(c,cell,r,state)
                elif mainline:
                    from AlphaBrain.research.dsol.data.prefix_protocol import validate_evaluation_row
                    validate_evaluation_row(c,cell,r,state)
                elif state['split']!='validation':raise ValueError('Development inputs only')
            seen.append(digest(r))
        if c['kind']=='training' and c['input_hashes'].get(b['policy'])!=b['policy_sha256']:raise ValueError('Unbound current actor')
        for k in total:total[k]+=b['limits'][k]
    if len(seen)!=len(set(seen)) or total!=c['limits']:raise ValueError('Duplicated rows or budget divergence')
    if c['kind']=='training' and (len(versions)!=1 or set(seen)!={digest(r) for r in scheduled if r['episode']//32==next(iter(versions))}):
        raise ValueError('One complete 32-episode version barrier required')
    if c['kind']=='evaluation' and (set(seen)!={digest(r) for r in scheduled} or (not mainline and len(seen)!=32)):
        raise ValueError('Full fixed evaluation panel required')
    if p['schema']==CORE_PROTOCOL and (len(scheduled)!=(32*p['updates'] if c['kind']=='training' else 32 if c['kind']=='evaluation' else len(scheduled))):
        raise ValueError('Core screening schedule length differs')
    if c['kind']=='evaluation' and not c.get('stay') and c['input_hashes'].get(c['actor']['path'])!=c['actor']['sha256']:
        raise ValueError('Unbound final actor')
    return c


def request_for(path,index):
    c=read(path);return dict(campaign=str(path),campaign_sha256=file_digest(path),block=c['blocks'][index],host=c['blocks'][index]['host'],stage='formal')


def policy_for(c, cell, payload=None, *, version=0):
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
    if cell.get('core_spec') is not None:
        from AlphaBrain.research.dsol.selectors.unified_search_policy import UnifiedActorCritic
        factory=UnifiedActorCritic;extra={}
        if cell.get('mechanism') is not None:
            from AlphaBrain.research.dsol.selectors.mechanism_policy import MechanismActorCritic
            factory=MechanismActorCritic;extra['mechanism']=cell['mechanism']
        policy=factory(cell['core_spec'],visual_dim=4608,
            task_spec=cell.get('task_condition',{}).get('spec'),**extra)
    else:
        policy=create_search_policy(cell['native_policy']) if cell['native_policy'] is not None else ContinuousActorCritic(4608)
    policy.training_protocol_sha256=file_digest(c['protocol'])
    if cell.get('task_condition') is not None:policy.task_condition=cell['task_condition']
    if cell['native_policy'] is not None:policy.operation_model_hashes=cell['model_hashes']
    if payload is not None:
        expected=dict(protocol_sha256=file_digest(c['protocol']),cell=c['cell'],score_contract_sha256=digest(cell['score_contract']),
            native_policy_sha256=digest(cell['native_policy']),model_hashes=cell['model_hashes'],initialization_sha256=file_digest(cell['initial_checkpoint']))
        if payload.get('prefix_origin')!=expected or payload.get('version')!=version:raise ValueError('Wrong prefix/model/feature/initialization actor origin')
        policy.load_state_dict(payload['state_dict'],strict=True)
    return policy.eval()


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/prefix_pilot.py:run')
    c=validate_campaign(request['campaign']);_,cell=cell_for(c);b=request['block']
    if request!=request_for(request['campaign'],b['index']):raise ValueError('Changed prefix request')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if c['kind']=='admission':return run_admission(c,cell,b,output,guard,simulator_pool)
    from .search_runtime import create_policy_runtime,bind_policy_task
    cap=b['limits']['score']+b['limits']['paired_execution'];sim=read(b['sim_config'])
    with ExitStack() as stack:
        runtime=(create_policy_runtime(sim,output,cap) if runtime_cache is None else
            stack.enter_context(runtime_cache.lease(digest(sim),lambda:create_policy_runtime(sim,output,cap),lambda r:bind_policy_task(r,output,cap)))[0])
        if c['kind']=='training':
            import torch
            from .continuous_training import LiveEpisodes
            from AlphaBrain.research.dsol.selectors.continuous_parallel import seeded_episode
            from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file,pack
            actor=policy_for(c,cell,torch.load(b['policy'],map_location='cpu',weights_only=True),version=b['version'])
            backend=LiveEpisodes(dict(sim_config=b['sim_config'],trajectory_cap=cap,**({'native_policy':cell['native_policy']} if cell['native_policy'] else {}),
                **({'task_condition':cell['task_condition']} if cell.get('task_condition') else {})),
                dict(actor=dict(score_scale=cell['observation_scale'])),output,runtime=runtime,reward_scale=.01214765438717935,simulator_pool=simulator_pool)
            records={}
            for r in b['rows']:
                guard();episode,metrics=seeded_episode(actor,backend,r,output/f"query-{r['episode']:05d}",version=b['version'],seed=cell.get('seed',0),source_kind='LIVE_RENDER_BATCH1_TWO_NOISES')
                metrics.update(prefix_cell=c['cell'],score_contract_sha256=digest(cell['score_contract']))
                path=output/f"episode-{r['episode']:05d}.pt";atomic_file(path,lambda f:torch.save(pack(dict(episode=episode,metrics=metrics)),f))
                records[str(r['episode'])]=dict(path=str(path),sha256=file_digest(path))
            write(output/'collection.json',dict(status='COMPLETE',kind='rollout',records=records,calls=runtime.calls,request_sha256=digest(request),training_updates=0,manipulation_actions=0))
        else:run_evaluation(c,cell,b,output,runtime,guard,simulator_pool)


def run_evaluation(c,cell,b,output,runtime,guard,pool):
    import torch
    from .search_live import simulator
    from .continuous_evaluation import EvaluationCalls,search,append_event,move_to_search_selection
    from .local_view_runtime import RenderedViewSource
    from .local_search_rollout import rollout_local_episode,rollout_groot_episode,input_identity
    from .rldx_search_runtime import rollout_rldx_episode
    from .search_domain import exact_input
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    open_sim=simulator if pool is None else pool.session;sim=read(b['sim_config'])
    with np.load(c['score_noise'],allow_pickle=False) as z:noises=z['noise'].copy()
    bank=ExplicitFlowNoiseBank(Path(c['execution_noise']));stay=c.get('stay',False)
    actor=None if stay else policy_for(c,cell,torch.load(c['actor']['path'],map_location='cpu',weights_only=True),version=c.get('evaluation_version',8))
    calls=EvaluationCalls(runtime,output,dict(score=b['limits']['score'],execution=b['limits']['paired_execution']),guard=guard)
    tasks=[];searches=[]
    for i,row in enumerate(b['rows']):
        folder=output/f'case-{i:03d}';folder.mkdir();values=noises[row['initial'],:2];calls.stage='score'
        with open_sim(b['sim_config'],sim,row['initial'],folder) as query:
            if stay:
                source=RenderedViewSource(query,calls,values,folder);source.move_angles(row['angles_deg']);selected=query(dict(kind='query_pose'))['example']
                info=dict(selected_angles=row['angles_deg'],queries=0,reason='stay',status='COMPLETE')
            else:
                info,selected=search(dict(row,method='continuous_ppo'),query,calls,values,folder,actor,cell['observation_scale'],
                    native_policy=cell['native_policy'],native_model_sha256=cell['model_hashes'][row['suite']])
            info.update(row,selected_input_sha256=input_identity(selected));write(folder/'search.json',info);searches.append(info)
        for repeat in (0,1):
            ep=folder/f'episode-{repeat}';ep.mkdir();calls.stage='execution'
            with open_sim(b['sim_config'],sim,row['initial'],ep) as query:
                source=RenderedViewSource(query,calls,values,ep);move_to_search_selection(source,info)
                example=query(dict(kind='begin_execution'))['example'];exact_input(selected,example)
                record=lambda event:append_event(ep/'events.jsonl',event)
                if cell['family']=='rldx':result,arrays=rollout_rldx_episode(query,calls.infer,example,bank,row['state_key'],repeat,row,runtime=runtime,record=record)
                else:
                    fn=rollout_local_episode if cell['family']=='pi05' else rollout_groot_episode
                    extra=dict(plan=read(sim['pilot_execution_plan'])) if cell['family']=='pi05' else dict(contract=sim['operation_contract'])
                    result,arrays=fn(query,calls.infer,example,bank,row['state_key'],repeat,row,suite=row['suite'],record=record,**extra)
            path=ep/'trajectory.npz';np.savez_compressed(path,**arrays);result.update(row,repeat=repeat,queries=info['queries'],selected_angles=info['selected_angles'],trajectory=str(path));write(ep/'result.json',result);tasks.append(result)
    write(output/'results.json',dict(tasks=tasks,searches=searches));write(output/'status.json',dict(status='COMPLETE',calls=calls.counts,optimizer_updates=0))


def validate_block(request,output):
    output=Path(output);receipt=read(output/'complete.json');c=read(request['campaign'])
    if receipt['request_sha256']!=digest(request):raise ValueError('Changed prefix completion identity')
    for p,h in receipt['artifacts'].items():
        if file_digest(output/p)!=h:raise ValueError('Changed prefix artifact')
    attempts=[json.loads(v) for v in (output/'model-attempts.jsonl').read_text().splitlines()]
    count=sum(1 for _ in (output/'calls.jsonl').open())
    if len(attempts)!=count:raise ValueError('Unknown or miscounted prefix forward')
    cap=request['block']['limits']
    if count>cap['score']+cap['paired_execution']:raise ValueError('Prefix call reservation exceeded')
    if c['kind']=='training':
        value=read(output/'collection.json');expected={str(r['episode']) for r in request['block']['rows']}
        if (value['status']!='COMPLETE' or value['request_sha256']!=digest(request) or set(value['records'])!=expected
                or value['calls']!=count or value['training_updates'] or value['manipulation_actions']):raise ValueError('Incomplete live training block')
        return dict(calls=dict(score=count,execution=0),receipt_sha256=file_digest(output/'complete.json'))
    if c['kind']=='admission':
        if read(output/'admission.json')['status']!='PASS_PREFIX_NUMERICAL_AND_TRAJECTORY':raise ValueError('Numerical/trajectory admission failed')
        return dict(calls=dict(score=count,execution=0),receipt_sha256=file_digest(output/'complete.json'))
    value=read(output/'results.json');status=read(output/'status.json');expected={(r['trial_id'],n) for r in request['block']['rows'] for n in (0,1)}
    counts={k:sum(r['identity']['stage']==k for r in attempts) for k in ('score','execution')}
    if (status['status']!='COMPLETE' or counts!=status['calls'] or len(value['tasks'])!=len(expected)
            or {(r['trial_id'],r['repeat']) for r in value['tasks']}!=expected
            or counts['execution']!=sum(r['replans'] for r in value['tasks'])):raise ValueError('Incomplete paired operation block')
    if counts['score']>cap['score'] or counts['execution']>cap['paired_execution']:raise ValueError('Stage call limit exceeded')
    return dict(calls=counts,receipt_sha256=file_digest(output/'complete.json'))


def learner_step(config_path,*,records_path=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Prefix learner must be CPU-only')
    c=read(config_path);protocol,cell=cell_for(c)
    for p,h in c['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Changed learner input')
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
    from AlphaBrain.research.dsol.selectors.continuous_parallel import ingest
    from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file
    from AlphaBrain.research.dsol.data.reward_pilot import reward_config
    if cell.get('evaluation_only'):raise ValueError('Deployment-only condition cannot train')
    reward=reward_config('G')
    if cell.get('mechanism') is not None:
        from AlphaBrain.research.dsol.data.core_mechanism import reward_for
        reward=reward_for(cell['mechanism'])
    seed=cell.get('seed',0)
    torch.set_num_threads(2);torch.manual_seed(seed);policy=policy_for(c,cell)
    initial=torch.load(cell['initial_checkpoint'],map_location='cpu',weights_only=True)
    if 'policy' in initial:
        if initial['update']!=0 or initial['cursor']!=0:raise ValueError('Exact matched v0 initialization required')
        policy.load_state_dict(initial['policy'],strict=True)
    else:
        if initial['optimizer_updates']!=0:raise ValueError('Untrained target initialization required')
        policy.load_state_dict(initial['state_dict'],strict=True)
    root=Path(c['output']);resume=records_path is not None or c.get('stage_resume',False)
    if c.get('stage_resume'):
        origin=read(root/'stage-origin.json')
        expected=cell.get('continuation',{})
        if (protocol.get('design_stage')!='long_continuation' or origin['source']!=expected.get('source_training')
                or origin['update']!=16 or origin['checkpoint']['sha256']!=expected.get('checkpoint_sha256')):
            raise ValueError('Continuation must use the audited complete stage fork')
    if root.exists()!=resume:raise ValueError('Clean init/update boundary required; no replay')
    def no_live(*args):raise RuntimeError('CPU learner cannot render or infer')
    with ContinuousPPO(policy,ContinuousPPOConfig(updates=protocol['updates'],seed=seed),read(cell['training_schedule']),no_live,root,
            binding=dict(config_sha256=file_digest(config_path),source_kind='LIVE_RENDER_BATCH1_TWO_NOISES',
                         **({'retained_checkpoint_updates':protocol.get('retained_checkpoint_updates',[4,8,16])} if protocol['schema']==CORE_PROTOCOL else {})),resume=resume,
            reward_config=reward,warm_start=initial if 'policy' in initial and not resume else None) as trainer:
        if records_path is not None:
            batch=read(records_path)
            if (batch['cell']!=c['cell'] or batch['version']!=trainer.update
                    or batch['policy_sha256']!=file_digest(root/'publications'/f'policy-{trainer.update:04d}.pt')):raise ValueError('Stale or wrong-prefix barrier')
            for ref in batch['records'].values():
                if file_digest(ref['path'])!=ref['sha256']:raise ValueError('Changed prefix rollout record')
                payload=torch.load(ref['path'],map_location='cpu',weights_only=True)
                if payload['metrics']['prefix_cell']!=c['cell'] or payload['metrics']['score_contract_sha256']!=digest(cell['score_contract']):raise ValueError('Mixed prefix rollouts')
            ingest(trainer,{int(k):v for k,v in batch['records'].items()});trainer.optimize()
        folder=root/'publications';folder.mkdir(exist_ok=True);target=folder/f'policy-{trainer.update:04d}.pt'
        if target.exists():raise ValueError('Published actor cannot be overwritten')
        origin=dict(protocol_sha256=file_digest(c['protocol']),cell=c['cell'],score_contract_sha256=digest(cell['score_contract']),
            native_policy_sha256=digest(cell['native_policy']),model_hashes=cell['model_hashes'],initialization_sha256=file_digest(cell['initial_checkpoint']))
        atomic_file(target,lambda f:torch.save(dict(state_dict=policy.state_dict(),version=trainer.update,prefix_origin=origin),f))
        return dict(run=c['cell'],version=trainer.update,episodes=trainer.cursor,status='COMPLETE' if trainer.update==protocol['updates'] else 'PAUSED_AT_UPDATE_BOUNDARY',policy=str(target),policy_sha256=file_digest(target),operation_model_updates=0,optimizer_device='cpu')


def run_admission(c,cell,b,output,guard,pool):
    from .prefix_admission import run
    return run(c,cell,b,output,guard,pool)
