"""Ledger-only native response-panel adapter; imports never launch a model."""
from pathlib import Path
from contextlib import ExitStack
import json
import sys
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.data.native_view_panel import blocks
from AlphaBrain.research.dsol.data import native_baseline
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key


def campaign_blocks(campaign):
    if campaign['schema']==native_baseline.SCHEMA:
        return native_baseline.blocks(campaign['states'])
    if campaign['schema']=='NATIVE_VIEW_RESPONSE_V1':
        return blocks(campaign['states'])
    raise ValueError('Unregistered native panel schema')


def request_for(campaign_path, index):
    campaign=read(campaign_path)
    block=campaign_blocks(campaign)[index]
    assignment=campaign['assignments'][str(index)]
    return dict(campaign=str(campaign_path),campaign_sha256=file_digest(campaign_path),
        model='native_reference',host=assignment['host'],stage=assignment['stage'],block=block)


def validate_spec(binding,key):
    spec=binding['tasks'][key];req=spec['request'];block=req['block']
    if req!=request_for(binding['campaign'],block['index']):
        raise ValueError('Native panel scientific request changed')
    if (spec['limits']!=block['limits'] or spec['addresses']!=block['addresses']
        or spec['host']!=req['host'] or spec['stage']!=req['stage']
        or key!=task_key(spec['identity_experiment_id'],req['campaign_sha256'],block['addresses'])):
        raise ValueError('Native panel identity or budget changed')
    return block['limits']


def validate_campaign(path):
    c=read(path)
    if c['model']!='native_reference':
        raise ValueError('Native panel scope required')
    campaign_blocks(c)
    expected=(21600,1468800) if c['schema']==native_baseline.SCHEMA else (768,43072)
    if (c['authorization']['new_episodes'],c['authorization']['new_denoising_calls'])!=expected or c['authorization']['training']:
        raise ValueError('Approved scope changed')
    for p,h in c['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Bound panel input changed: '+p)
    if c['schema']==native_baseline.SCHEMA:
        paired=read(c['paired_config'])
        if set(paired['cohorts'])!={'A'}:
            raise ValueError('Full baseline uses one explicit state collection')
        sim_path=paired['cohorts']['A']['sim_config'];sim=read(sim_path)
        required=[c['paired_config'],c['score_noise'],paired['design'],sim_path,sim['inputs'],
                  paired['cohorts']['A']['execution_noise']]
        if any(p not in c['input_hashes'] for p in required):
            raise ValueError('Unbound full baseline asset')
        records=read(sim['inputs'])['records']
        if len(records)!=400 or c['state_keys']!=[s['state_key'] for s in c['states']]:
            raise ValueError('Full baseline state/noise index mismatch')
        for row in c['states']:
            state=records[row['initial']]['state']
            if (state['pair_key']!=row['state_key'] or state['task_id']!=row['task']
                    or state['suite']!=row['suite'] or state['initialization']['settle_steps']!=10):
                raise ValueError('Full baseline simulator identity mismatch')
    return c


def run(request, output, *, guard, runtime_cache=None, simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/native_view_panel.py:run')
    if not callable(guard):raise ValueError('Live authority guard required')
    c=validate_campaign(request['campaign'])
    if request!=request_for(request['campaign'],request['block']['index']):raise ValueError('Changed request')
    from .continuous_evaluation import EvaluationCalls, append_event
    from .search_runtime import create_policy_runtime, bind_policy_task
    from .search_live import simulator, check_simulator_dependencies
    from .local_view_runtime import RenderedViewSource
    from .local_search_rollout import rollout_local_episode, input_identity
    from .search_domain import exact_input
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank, tensor_sha256
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    tasks=[];scores=[];calls=None;began=time.perf_counter();model_seconds=None
    leases=ExitStack();runtime_reused=False
    open_simulator=simulator if simulator_pool is None else simulator_pool.session
    try:
        cfg=read(c['paired_config']);plan=read(cfg['design'])
        sims={k:read(v['sim_config']) for k,v in cfg['cohorts'].items()}
        for sim in sims.values():check_simulator_dependencies(sim)
        bank={k:ExplicitFlowNoiseBank(Path(v['execution_noise']),verify_file=True) for k,v in cfg['cohorts'].items()}
        cap=request['block']['limits'];limits=dict(score=cap['score'],execution=cap['paired_execution'])
        guard();loading=time.perf_counter()
        if runtime_cache is None:
            runtime=create_policy_runtime(sims['A'],output,sum(limits.values()))
        else:
            runtime,runtime_reused=leases.enter_context(runtime_cache.lease(digest(sims['A']),
                lambda:create_policy_runtime(sims['A'],output,sum(limits.values())),
                lambda value:bind_policy_task(value,output,sum(limits.values()))))
        model_seconds=time.perf_counter()-loading
        calls=EvaluationCalls(runtime,output,limits,guard=guard)
        with np.load(c['score_noise'],allow_pickle=False) as z: noises=z['noise'].copy()
        for row in request['block']['rows']:
            index=c['state_keys'].index(row['state_key']);values=noises[index]
            spec=cfg['cohorts'][row['cohort']];sim=sims[row['cohort']]
            case=output/f"view-{row['view_id']:02d}";case.mkdir()
            identity={k:row[k] for k in ('cohort','initial','state_key','task','view_id')}
            calls.identity=identity;calls.stage='score'
            with calls.timing.context(open_simulator(spec['sim_config'],sim,row['initial'],case),
                                      'score_simulator') as query:
                query=calls.timing.wrap('score_simulator_rpc',query)
                source=RenderedViewSource(query,calls,values[:2],case)
                baseline=query(dict(kind='query_pose'))['example']
                reference=c['reuse_inputs'].get(f"{row['state_key']}::0")
                if reference is None and c['schema']!=native_baseline.SCHEMA:
                    raise ValueError('Missing frozen nominal input')
                if reference is not None and input_identity(baseline)!=reference['input_sha256']:
                    raise ValueError('Native nominal first-frame identity changed')
                write(case/'nominal-input.json',dict(input_sha256=input_identity(baseline),
                    prior_reference_checked=reference is not None))
                source.move_angles(row['angles_deg'])
                response=query(dict(kind='query_pose'));example=response['example']
                receipt=source._receipt(response['receipt'])
                if not np.allclose(receipt.camera_to_world,source.mount.pose(row['angles_deg']),rtol=0,atol=1e-9):
                    raise ValueError('Applied panel pose mismatch')
                saved=dict(external_image=example['image'][0],wrist_image=example['image'][1],
                    robot_state=example['state'],camera_intrinsics=example['camera_intrinsics'],
                    camera_to_world_opencv=example['camera_to_world_opencv'])
                # Existing views prove fresh input identity without additional model calls.
                control=c['reuse_inputs'].get(f"{row['state_key']}::{row['view_id']}")
                if control and input_identity(example)!=control['input_sha256']:
                    raise ValueError('Reused panel input differs from existing execution')
                for member in row['score_members']:
                    value=calls.infer(example,values[member],dict(member=member),capture=True)
                    score=dict(identity,member=member,score=float(value['score']),angles_deg=row['angles_deg'],
                        input_sha256=input_identity(example),noise_sha256=tensor_sha256(values[member]))
                    scores.append(score)
                    # Preserve each returned score before another model call;
                    # missing results after a crash still remain charged.
                    write(case/f'score-member-{member}.json',score)
                    for k in ('trace','actions','full_trace','full_noise'):
                        if k in value:saved[f'{k}_{member}']=value[k]
                np.savez_compressed(case/'query.npz',**saved)
                write(case/'scores.json',scores[-len(row['score_members']):])
            for repeat in row['execution_repeats']:
                episode=case/f'episode-{repeat}';episode.mkdir();calls.stage='execution'
                with calls.timing.context(open_simulator(spec['sim_config'],sim,row['initial'],episode),
                                          'execution_simulator') as query:
                    query=calls.timing.wrap('execution_simulator_rpc',query)
                    source=RenderedViewSource(query,calls,values[:2],episode)
                    source.move_angles(row['angles_deg']);response=query(dict(kind='begin_execution'))
                    exact_input(example,response['example'])
                    result,arrays=rollout_local_episode(query,calls.infer,response['example'],bank[row['cohort']],
                        row['state_key'],repeat,identity,plan=plan,suite=row['suite'],
                        record=lambda event:append_event(episode/'events.jsonl',event))
                asset=episode/'trajectory.npz';np.savez_compressed(asset,**arrays)
                result.update(identity,angles_deg=row['angles_deg'],trajectory=str(asset),trajectory_sha256=file_digest(asset))
                write(episode/'result.json',result);tasks.append(result)
                write(output/'progress.json',dict(status='RUNNING',tasks=len(tasks),scores=len(scores),calls=calls.counts))
        if len(tasks)!=cap['episodes'] or len(scores)!=cap['score']:raise ValueError('Incomplete panel block')
        write(output/'results.json',dict(tasks=tasks,scores=scores,block_index=request['block']['index']))
        write(output/'status.json',dict(status='COMPLETE',calls=calls.counts,tasks=len(tasks),scores=len(scores),training_updates=0))
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),calls=calls.counts if calls else {},completed_tasks=len(tasks),automatic_retry=False))
        raise
    finally:
        leases.__exit__(*sys.exc_info())
        write(output/'timing.json',dict(elapsed_seconds=time.perf_counter()-began,
            model_load_seconds=model_seconds,runtime_reused=runtime_reused,phases=calls.timing.rows if calls else {}))


def validate_block(req, output):
    from AlphaBrain.research.dsol.data.native_view_panel import addresses
    output=Path(output);complete=read(output/'complete.json');status=read(output/'status.json');result=read(output/'results.json')
    if complete['status']!='COMPLETE' or complete['request_sha256']!=digest(req):raise ValueError('Completion identity mismatch')
    for name,h in complete['artifacts'].items():
        if file_digest(output/name)!=h:raise ValueError('Panel artifact changed: '+name)
    rows=req['block']['rows'];cap=req['block']['limits']
    expected_exec={(r['state_key'],r['view_id'],s) for r in rows for s in r['execution_repeats']}
    expected_score={(r['state_key'],r['view_id'],s) for r in rows for s in r['score_members']}
    actual_exec=[(r['state_key'],r['view_id'],r['repeat']) for r in result['tasks']]
    actual_score=[(r['state_key'],r['view_id'],r['member']) for r in result['scores']]
    if (len(actual_exec)!=len(expected_exec) or set(actual_exec)!=expected_exec
        or len(actual_score)!=len(expected_score) or set(actual_score)!=expected_score):
        raise ValueError('Missing/duplicate scientific outcome')
    attempts=[json.loads(s) for s in (output/'model-attempts.jsonl').read_text().splitlines()]
    counts={k:sum(a['identity']['stage']==k for a in attempts) for k in ('score','execution')}
    if (status['status']!='COMPLETE' or status['training_updates']!=0 or status['calls']!=counts
        or counts['score']!=cap['score'] or counts['execution']>cap['paired_execution']
        or counts['execution']!=sum(t['replans'] for t in result['tasks'])):
        raise ValueError('Scientific call accounting mismatch')
    for t in result['tasks']:
        if type(t['success']) is not bool or t['steps']>t['horizon'] or len(t['execution_noise_sha256'])!=t['replans']:
            raise ValueError('Invalid execution outcome')
    return dict(calls=counts,receipt_sha256=file_digest(output/'complete.json'))
