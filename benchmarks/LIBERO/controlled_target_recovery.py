"""Hash-bound target-model searches followed by paired native manipulation.

Development only: actors are fixed endpoints, outcomes cannot change the panel.
Uses the admitted target operation loop and existing continuous/local search.
"""
from contextlib import ExitStack
from pathlib import Path
import json,time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest,task_key
from .controlled_target_evaluation import validate_assets,execution_hosts

SCHEMA='CONTROLLED_TARGET_RECOVERY_V1'


def limits(rows,contract):
    return dict(episodes=len(rows)*2,score=sum(16 for r in rows if r['method'] not in ('stay','reference')),
        paired_execution=sum(2*(contract['horizons'][r['suite']]//10) for r in rows),nominal_execution=0)


def row_key(r):return (r['state_key'],r['start'],r['method'])


def validate_campaign(path):
    if read(path).get('schema') in ('CONTROLLED_PREFIX_PILOT_V1','CONTROLLED_PREFIX_MAINLINE_V1'):
        from .prefix_pilot import validate_campaign as prefix_call
        return prefix_call(path)
    if read(path).get('schema')=='CONFIRMED_CHANGE_PILOT_V1':
        from .confirmed_pilot import validate_campaign as validate_pilot
        return validate_pilot(path)
    from . import native_recovery
    if read(path).get('schema')==native_recovery.SCHEMA:return native_recovery.validate_campaign(path)
    c,sim,states,contract=validate_assets(path)
    from .target_recovery_resume import validate_proof
    proof=validate_proof(c)
    if (c['schema']!=SCHEMA or c['purpose']!='fewshot_development' or c['score_scale']!=.16535300100378683
            or c['query_cap']!=8 or c['execution_repeats']!=[0,1]):
        raise ValueError('Explicit target fewshot evaluation contract required')
    required={c['schedule'],c['score_noise'],c['trajectory_proof']}
    if read(c['trajectory_proof'])['status']!='PASS':raise ValueError('Trajectory gate has not passed')
    scheduled=read(c['schedule']);planned={row_key(r):r for r in scheduled}
    if len(planned)!=len(scheduled):raise ValueError('Duplicate frozen development rows')
    actual=[]
    for i,b in enumerate(c['blocks']):
        if b['index']!=i or b['host'] not in execution_hosts(c) or not b['rows'] or b['limits']!=limits(b['rows'],contract):
            raise ValueError('Target evaluation block differs')
        for r in b['rows']:
            if r!=planned.get(row_key(r)):raise ValueError('Outcome-selected or unplanned evaluation row')
            if proof and digest(r) not in proof['records']:raise ValueError('Row absent from explicit recovery audit')
            state=states[r['initial']]['state']
            if (state['pair_key']!=r['state_key'] or state['suite']!=r['suite'] or state['split']!='validation'
                    or state['initialization']['settle_steps']!=30 or r['query_cap']!=8):
                raise ValueError('Development task/state/initialization mismatch')
            if r['method'].startswith('actor:'):
                spec=c['actors'][r['method']];required.add(spec['path'])
                if spec['version'] not in (0,1,4,8):raise ValueError('Unplanned learner endpoint')
            elif r['method'] not in ('stay','reference','local','random_from_local'):
                raise ValueError('Unknown target selection method')
            actual.append(r)
    if (not required.issubset(c['input_hashes']) or len({row_key(r) for r in actual})!=len(actual)
            or c['limits']!=limits(actual,contract)):
        raise ValueError('Repeated or unbound evaluation budget')
    noise=read(c['execution_noise'])
    if any(r['state_key'] not in noise['state_keys'] for r in actual):raise ValueError('Missing execution noise state')
    return c


def request_for(path,index):
    c=read(path)
    if c.get('schema') in ('CONTROLLED_PREFIX_PILOT_V1','CONTROLLED_PREFIX_MAINLINE_V1'):
        from .prefix_pilot import request_for as prefix_request
        return prefix_request(path,index)
    if c.get('schema')=='CONFIRMED_CHANGE_PILOT_V1':
        from .confirmed_pilot import request_for as pilot_request
        return pilot_request(path,index)
    from . import native_recovery
    if c.get('schema')==native_recovery.SCHEMA:return native_recovery.request_for(path,index)
    b=dict(c['blocks'][index]);b['addresses']=[dict(model='fastwam',task=r['state_key'],
        initial=r['state_key'],view=f"{r['start']}:{r['method']}",stream='execution',repeat=s,noise=f'paired-{s}')
        for r in b['rows'] for s in (0,1)]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_spec(binding,key):
    s=binding['tasks'][key];r=s['request'];b=r['block']
    if (r!=request_for(binding['campaign'],b['index']) or s['limits']!=b['limits'] or s['addresses']!=b['addresses']
            or s['host']!=r['host'] or key!=task_key(s['identity_experiment_id'],r['campaign_sha256'],b['addresses'])):
        raise ValueError('Target recovery binding changed')
    return b['limits']


def load_actors(c,rows):
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    torch.set_num_threads(2);actors={}
    native=c.get('native_policy')
    for method in sorted({r['method'] for r in rows if r['method'].startswith('actor:') or (native is not None and r['method'] in c['actors'])}):
        spec=c['actors'][method];p=torch.load(spec['path'],map_location='cpu',weights_only=True)
        if p['version']!=spec['version']:raise ValueError('Wrong fixed learner endpoint')
        if native is not None:
            from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy,validate_publication
            actor=create_search_policy(native).eval();actor.training_protocol_sha256=file_digest(c['protocol'])
            from AlphaBrain.research.dsol.data.native_training import publication_origin
            validate_publication(actor,p,version=spec['version'],expected_origin=publication_origin(
                read(c['protocol']),file_digest(c['protocol']),spec['seed']))
        else:
            actor=ContinuousActorCritic(4608).eval();actor.load_state_dict(p['state_dict'],strict=True)
        actor.requires_grad_(False);actors[method]=actor
    return actors


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_target_recovery.py:run')
    if read(request['campaign']).get('schema') in ('CONTROLLED_PREFIX_PILOT_V1','CONTROLLED_PREFIX_MAINLINE_V1'):
        from .prefix_pilot import run as prefix_run
        return prefix_run(request,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
    if read(request['campaign']).get('schema')=='CONFIRMED_CHANGE_PILOT_V1':
        from .confirmed_pilot import run_evaluation
        return run_evaluation(request,output,guard=guard,runtime_cache=runtime_cache,simulator_pool=simulator_pool)
    c=validate_campaign(request['campaign']);b=request['block']
    if request!=request_for(request['campaign'],b['index']):raise ValueError('Changed target recovery request')
    native=c.get('native_policy')
    if native is not None:c=dict(c,sim_config=b['sim_config'])
    from .search_runtime import create_policy_runtime,bind_policy_task
    from .search_live import simulator
    from .local_view_runtime import RenderedViewSource
    from .continuous_evaluation import EvaluationCalls,search,append_event,move_to_search_selection
    from .controlled_recovery import random_visited
    from .local_search_rollout import rollout_fastwam_episode,rollout_groot_episode,input_identity
    from .search_domain import exact_input
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    from .target_recovery_resume import validate_proof,RecoveredCalls,completed_episode
    recovery=validate_proof(c)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);sim=read(c['sim_config']);started=time.monotonic()
    bank=ExplicitFlowNoiseBank(Path(c['execution_noise']))
    with np.load(c['score_noise']) as z:noises=z['noise'].copy()
    actors=load_actors(c,b['rows']);tasks=[];searches=[]
    if native is not None:
        from .model_adapters import model_adapter
        rollout=model_adapter(read(c['protocol'])['family']).rollout
    else:rollout=rollout_fastwam_episode
    open_sim=simulator if simulator_pool is None else simulator_pool.session
    with ExitStack() as stack:
        cap=b['limits']['score']+b['limits']['paired_execution']
        if runtime_cache is None:runtime=create_policy_runtime(sim,output,cap)
        else:runtime,_=stack.enter_context(runtime_cache.lease(digest(sim),lambda:create_policy_runtime(sim,output,cap),
            lambda v:bind_policy_task(v,output,cap)))
        live_calls=EvaluationCalls(runtime,output,dict(score=b['limits']['score'],execution=b['limits']['paired_execution']),guard=guard)
        for i,row in enumerate(b['rows']):
            guard();folder=output/f'case-{i:04d}';folder.mkdir();values=noises[row['initial'],:2]
            ref=recovery.get('records',{}).get(digest(row));old=Path(ref['case']) if ref else None
            calls=RecoveredCalls(live_calls,old,output,guard) if ref else live_calls
            calls.stage='score'
            with open_sim(c['sim_config'],sim,row['initial'],folder) as query:
                if ref and (old/'search.json').exists():
                    info=read(old/'search.json')
                    if any(info.get(k)!=v for k,v in row.items()):raise ValueError('Recovered search identity differs')
                    source=RenderedViewSource(query,calls,values,folder)
                    move_to_search_selection(source,info)
                    response=query(dict(kind='query_pose'));source._receipt(response['receipt']);selected=response['example']
                    if input_identity(selected)!=info['selected_input_sha256']:raise ValueError('Recovered selected input differs')
                    append_event(output/'reused-calls.jsonl',dict(stage='score',count=2*info['queries'],source=str(old/'search.json'),model_calls=0))
                elif row['method'] in ('stay','reference'):
                    source=RenderedViewSource(query,calls,values,folder)
                    angles=row['angles_deg'] if row['method']=='stay' else [0.,0.]
                    # The reference is reached from the same perturbed start.
                    source.move_angles(row['angles_deg']);source.move_angles(angles)
                    response=query(dict(kind='query_pose'));source._receipt(response['receipt'])
                    selected=response['example'];info=dict(selected_angles=angles,queries=0,reason=row['method'],events=[])
                else:
                    r=dict(row,method='continuous_ppo' if row['method'].startswith('actor:') or row['method'] in actors else 'local_near_best_accel')
                    info,selected=search(r,query,calls,values,folder,actors.get(row['method']),c['score_scale'],**({'native_policy':native,'native_model_sha256':sim['models'][sim['operation_model']]['weights_sha256']} if native is not None else {}))
                    if row['method']=='random_from_local':
                        angles,choice,count=random_visited(info,row['random_seed'])
                        # Diagnostic random-visited control: no additional
                        # score is used to choose its already visited input.
                        # This is not a new deployment selection algorithm.
                        source=RenderedViewSource(query,calls,values,folder,
                            current_angles=info['current_angles']);source.move_angles(angles)
                        response=query(dict(kind='query_pose'));source._receipt(response['receipt'])
                        selected=response['example'];info.update(selected_angles=angles,random_candidate_index=choice,random_candidate_count=count)
                info.update(row,selected_input_sha256=input_identity(selected));write(folder/'search.json',info);searches.append(info)
                if native is not None and getattr(runtime,'native_needs_warmup',False):
                    calls.infer(selected,values[0],dict(purpose='native_graph_warmup'),capture=False,warmup=True)
            if native is not None and row['method'] not in ('stay','reference'):
                from .native_view_source import NativeRenderedViewSource
                previous_identity=dict(live_calls.identity);live_calls.identity.update(purpose='native_diagnostic')
                diagnostic=folder/'diagnostic';diagnostic.mkdir()
                with open_sim(c['sim_config'],sim,row['initial'],diagnostic) as query:
                    source=NativeRenderedViewSource(query,calls,values,diagnostic,expected_model_sha256=sim['models'][sim['operation_model']]['weights_sha256'],expected_producer_sha256=native['producer_sha256'])
                    source.move_angles([0.,0.]);source.query()
                    reference_score=[float(v['score']) for v in source.last_values]
                    move_to_search_selection(source,info);source.query()
                    write(diagnostic/'scores.json',dict(reference=reference_score,selected=[float(v['score']) for v in source.last_values],used_by_policy=False))
                live_calls.identity=previous_identity
            for repeat in (0,1):
                ep=folder/f'episode-{repeat}';ep.mkdir();calls.stage='execution'
                prior=completed_episode(old,repeat,row) if ref else None
                if prior is not None:
                    if prior['input_sha256'][0]!=input_identity(selected):raise ValueError('Recovered episode selected input differs')
                    write(ep/'result.json',prior);tasks.append(prior)
                    append_event(output/'reused-calls.jsonl',dict(stage='execution',count=prior['replans'],source=str(old/f'episode-{repeat}/result.json'),model_calls=0))
                    continue
                with open_sim(c['sim_config'],sim,row['initial'],ep) as query:
                    source=RenderedViewSource(query,calls,values,ep);move_to_search_selection(source,info)
                    example=query(dict(kind='begin_execution'))['example'];exact_input(selected,example)
                    result,arrays=rollout(query,calls.infer,example,bank,row['state_key'],repeat,row,
                        contract=sim['operation_contract'],suite=row['suite'],record=lambda e:append_event(ep/'events.jsonl',e))
                if ref:calls.verify_trajectory(repeat,result,arrays)
                asset=ep/'trajectory.npz';np.savez_compressed(asset,**arrays)
                result.update(row,repeat=repeat,selected_angles=info['selected_angles'],queries=info['queries'],trajectory=str(asset))
                write(ep/'result.json',result);tasks.append(result)
            write(output/'progress.json',dict(searches=len(searches),tasks=len(tasks),calls=live_calls.counts))
        write(output/'results.json',dict(tasks=tasks,searches=searches))
        # Empty actual-call ledgers are explicit when every prediction was recovered.
        for name in ('model-attempts.jsonl','calls.jsonl'):(output/name).touch(exist_ok=True)
        write(output/'status.json',dict(status='COMPLETE',calls=live_calls.counts,seconds=time.monotonic()-started,optimizer_updates=0))


def validate_block(request,output):
    if read(request['campaign']).get('schema') in ('CONTROLLED_PREFIX_PILOT_V1','CONTROLLED_PREFIX_MAINLINE_V1'):
        from .prefix_pilot import validate_block as prefix_call
        return prefix_call(request,output)
    if read(request['campaign']).get('schema')=='CONFIRMED_CHANGE_PILOT_V1':
        from .confirmed_pilot import validate_evaluation_block
        return validate_evaluation_block(request,output)
    from . import native_recovery
    if read(request['campaign']).get('schema')==native_recovery.SCHEMA:return native_recovery.validate_block(request,output)
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Target recovery request differs')
    for p,h in receipt['artifacts'].items():
        if file_digest(output/p)!=h:raise ValueError('Changed target recovery artifact')
    value=read(output/'results.json');status=read(output/'status.json');rows=request['block']['rows']
    expected={(row_key(r),s) for r in rows for s in (0,1)}
    if (len(value['tasks'])!=len(expected) or {(row_key(r),r['repeat']) for r in value['tasks']}!=expected
            or len(value['searches'])!=len(rows) or {row_key(r) for r in value['searches']}!={row_key(r) for r in rows}):
        raise ValueError('Incomplete paired target recovery')
    attempts=[json.loads(v) for v in (output/'model-attempts.jsonl').read_text().splitlines()]
    counts={stage:sum(r['identity']['stage']==stage for r in attempts) for stage in ('score','execution')}
    runtime=sum(1 for _ in (output/'calls.jsonl').open())
    reused=dict(score=0,execution=0)
    if (output/'reused-calls.jsonl').exists():
        from .target_recovery_resume import validate_proof
        if not validate_proof(read(request['campaign'])):raise ValueError('Reused calls lack explicit audit')
        for line in (output/'reused-calls.jsonl').read_text().splitlines():
            event=json.loads(line)
            if event['model_calls']!=0:raise ValueError('Cached result charged as a new call')
            reused[event['stage']]+=event.get('count',1)
    if (counts!=status['calls'] or status['status']!='COMPLETE' or runtime!=sum(counts.values())
            or counts['execution']+reused['execution']!=sum(r['replans'] for r in value['tasks'])
            or counts['score']+reused['score']!=2*sum(r['queries'] for r in value['searches'])
            or counts['score']>request['block']['limits']['score']
            or counts['execution']>request['block']['limits']['paired_execution']):
        raise ValueError('Target recovery call ledger differs')
    return dict(calls=counts,receipt_sha256=file_digest(output/'complete.json'))
