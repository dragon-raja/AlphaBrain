"""Frozen-actor continuous searches and paired real manipulation episodes.

One persistent policy runtime; fresh simulator for each search/execution.
There is no optimizer, cached outcome lookup, retry, or extra control task.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write as write_json
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha
from AlphaBrain.research.dsol.data.local_search_plan import validate_schedules
from AlphaBrain.research.dsol.analysis.continuous_evaluation import summarize
from .local_search_rollout import input_identity, rollout_local_episode


def validate_inputs(path):
    cfg = read(path)
    if cfg.get('schema') != 'continuous_evaluation_launch_v1':
        raise ValueError('Explicit continuous evaluation manifest required')
    if cfg['limits'] != dict(score=544, execution=10560):
        raise ValueError('Frozen evaluation budgets changed')
    for filename, digest in cfg['input_hashes'].items():
        if sha(filename) != digest:
            raise ValueError('Bound evaluation input changed: '+filename)
    required = [cfg[k] for k in ('design','schedule','training_schedule','actor','actor_controls')]
    plan = read(cfg['design'])
    manifests = {}
    for cohort, spec in cfg['cohorts'].items():
        sim = read(spec['sim_config'])
        if (sim.get('continuous_execution') != 'PAIRED_EVALUATION_ONLY'
                or sim.get('continuous_camera') is not True
                or any(k in sim for k in ('native','outcome_matrix','release'))):
            raise ValueError('Continuous live backend only; no catalog/cache inputs')
        from .search_runtime import selected_checkpoint
        checkpoint = selected_checkpoint(sim, allow_native=True)
        if checkpoint['weights_sha256'] != plan['operation_policy']['checkpoint_sha256']:
            raise ValueError('Design and simulator operation-model identities differ')
        required += [spec['sim_config'], spec['score_noise'], spec['execution_noise'], sim['inputs'],
                     str(Path(checkpoint['path'])/'model.safetensors')]
        if checkpoint.get('implementation')=='openpi_native':
            norm=str(Path(checkpoint['path'])/'assets/physical-intelligence/libero/norm_stats.json')
            if cfg['input_hashes'].get(norm)!=checkpoint.get('normalization_sha256'):
                raise ValueError('Unbound native normalization')
        manifests[cohort] = read(sim['inputs'])['records']
    if any(p not in cfg['input_hashes'] for p in required):
        raise ValueError('Unbound evaluation input')
    budget = validate_schedules(plan, read(cfg['training_schedule']), read(cfg['schedule']), manifests)
    if plan['evaluation']['methods'] != ['stay','local_near_best_accel','continuous_ppo']:
        raise ValueError('Frozen methods changed')
    rows = read(cfg['schedule'])
    for i in range(0, len(rows), 3):
        if [r['method'] for r in rows[i:i+3]] != plan['evaluation']['methods']:
            raise ValueError('Evaluate complete three-method blocks in frozen order')
    return cfg, plan, budget


def append_event(path, value):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, default=encode, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


class EvaluationCalls:
    def __init__(self, runtime, output, limits, *, guard=None):
        self.runtime, self.output, self.limits = runtime, Path(output), limits
        self.counts = dict.fromkeys(limits, 0)
        from AlphaBrain.research.dsol.execution.phase_timing import PhaseTiming
        self.timing=PhaseTiming()
        self.guard = guard
        self.stage = None
        self.identity = {}

    def infer(self, example, noise, identity, capture=True, *, warmup=False,native_request=None,feature_capture=True):
        from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256
        if self.guard is not None:
            with self.timing.measure('authority_wait'):
                self.guard()
        stage = self.stage
        if stage not in self.counts or self.counts[stage] >= self.limits[stage]:
            raise ValueError('Stage call budget exhausted')
        self.counts[stage] += 1
        address = dict(identity, **self.identity, stage=stage)
        with self.timing.measure('durable_call_ledger'):
            append_event(self.output/'model-attempts.jsonl', dict(index=sum(self.counts.values()),
            stage_index=self.counts[stage], unix_time=time.time(), identity=address,
            input_sha256=input_identity(example), noise_sha256=tensor_sha256(noise)))
        with self.timing.measure(stage+'_inference'):
            extra={'warmup':True} if warmup else {}
            if native_request is not None:extra['native_request']=native_request
            if not feature_capture:extra['feature_capture']=False
            value = self.runtime.infer(example, noise, address, capture=capture,**extra)
        if self.runtime.calls != sum(self.counts.values()):
            raise ValueError('Runtime and durable ledger diverged')
        return value


def actor_action(actor, observation, *, task_features=None):
    import torch
    from AlphaBrain.research.dsol.acquisition.continuous_search_gym import continuous_observation
    if hasattr(actor,'feature_contract_sha256'):
        from AlphaBrain.research.dsol.acquisition.native_search import padded_native_observation
        obs=padded_native_observation(observation)
    else:obs=continuous_observation(observation)
    if task_features is not None:obs=dict(obs,task_features=task_features)
    tensor = {k:torch.as_tensor(v[None]) for k,v in obs.items()}
    with torch.inference_mode():
        value = actor.distribution(tensor).sample(deterministic=True)
    return dict(stop=bool(value['stop'][0]), delta_deg=value['delta_deg'][0].numpy())


def search(row, request, calls, noises, directory, actor, scale,*,native_policy=None,native_model_sha256=None,confirmation_noises=None):
    from .local_view_runtime import RenderedViewSource
    from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewSearch
    from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
    if row['method']=='random_acquisition':
        from AlphaBrain.research.dsol.acquisition.random_acquisition import UniformIncrementAcquisition
        if actor is not None:raise ValueError('Independent random acquisition cannot use an actor')
        acquisition=UniformIncrementAcquisition(row['acquisition_seed'])
    if row['method']=='local_near_best_accel':
        from AlphaBrain.research.dsol.acquisition.local_view_gym import local_baseline_action
    shifted='mount_yaw_offset_deg' in row
    if shifted:
        full=row.get('coordinate_family')=='heldout_installation_v1'
        allowed=(-15.,0.,15.) if full else (-5.,0.,5.)
        methods=('continuous_ppo','local_near_best_accel','stay','fixed_home') if full else ('continuous_ppo',)
        if (row['method'] not in methods or row['mount_yaw_offset_deg'] not in allowed
                or row['search_yaw_limit_deg']!=(15. if full else 25.)
                or row.get('coordinate_family') not in (None,'heldout_installation_v1')):
            raise ValueError('Unsupported explicit mounting experiment')
    native_policy=native_policy or getattr(actor,'native_spec',None)
    if native_policy is not None:
        from .native_view_source import NativeRenderedViewSource
        from AlphaBrain.research.dsol.acquisition.native_search import NativeLocalSearch,NativeContinuousSearch
        from AlphaBrain.research.dsol.data.native_features import read_layout
        from AlphaBrain.research.dsol.execution.manifest import digest
        if native_model_sha256 is None:raise ValueError('Independent target model identity required for native evaluation')
        source=NativeRenderedViewSource(request,calls,noises,directory,expected_model_sha256=native_model_sha256,expected_producer_sha256=native_policy['producer_sha256'])
        cls=NativeLocalSearch if row['method']=='local_near_best_accel' else NativeContinuousSearch
    else:
        source = RenderedViewSource(request, calls, noises, directory)
        cls = LocalViewSearch if row['method']=='local_near_best_accel' else ContinuousSearch
    mount,initial=source.mount,row['angles_deg']
    if shifted:
        from dataclasses import replace
        from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
        offset=row['mount_yaw_offset_deg']
        mount=PanTiltMount(source.mount.pose([offset,0.]),replace(source.mount.config,yaw_limit_deg=row['search_yaw_limit_deg']))
        initial=np.asarray(initial)-[offset,0.]
    if shifted and full:
        # Every visit, including the initial view and a later return to it,
        # uses identical matrix composition. All paired methods share this.
        for point in mount.route(np.asarray(source.current)-[offset,0.],initial):
            source.move(mount.pose(point))
    else:
        source.move_angles(row['angles_deg'])
    first = source.query()
    if row['method']=='fixed_home':
        if not shifted or not full:raise ValueError('Explicit full mechanism home control required')
        for point in mount.route(initial,[0.,0.]):source.move(mount.pose(point))
        selected=source.query()
        return dict(status='COMPLETE',reason='fixed_home',queries=source.queries,visited=2,
            selected_angles=[offset,0.],current_angles=[offset,0.],mechanical_selected_angles=[0.,0.],
            mount_yaw_offset_deg=offset,search_yaw_limit_deg=row['search_yaw_limit_deg'],
            coordinate_family=row['coordinate_family'],event_angle_frame='shifted_mount_mechanical',
            initial_accel=first.value.accel_mean,best_accel=min(first.value.accel_mean,selected.value.accel_mean),
            selected_accel=selected.value.accel_mean,selection_rule='mechanical_home_ignores_scores',
            events=[dict(kind='initial',angles=initial.tolist()),dict(kind='fixed_home',angles=[0.,0.])]),source.last_example
    if row['method'] != 'stay' and row['query_cap'] != 8:
        from dataclasses import replace
        from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
        spec=getattr(actor,'mechanism',{})
        if spec.get('query_cap')!=row['query_cap'] or row['query_cap'] not in (4,6):
            raise ValueError('Reduced query cap requires the bound deployment mechanism')
        mount=PanTiltMount(mount.reference,replace(mount.config,query_budget=row['query_cap']))
    original, pending = source.query, [first]
    source.query = lambda: pending.pop() if pending else original()
    core = cls(mount, source, initial_angles=initial,
        physical_state_id=source.physical, **(dict(streams=read_layout(native_policy['layout']),producer_sha256=native_policy['producer_sha256'],feature_layout=native_policy['layout']) if native_policy is not None else dict(wrist=source.last_values[0]['wrist'])),
        score_scale=scale, seed=row['proposal_seed'])
    obs = core.start()
    task_features=None
    if getattr(actor,'task_condition',None) is not None:
        from AlphaBrain.research.dsol.data.task_condition import task_vector
        task_features=task_vector(actor.task_condition,source.last_example['lang'])
    while core.status == 'ACTIVE':
        if row['method'] == 'stay': action = dict(stop=True, delta_deg=np.zeros(2))
        elif row['method'] == 'continuous_ppo':
            action = actor_action(actor,obs) if task_features is None else actor_action(actor,obs,task_features=task_features)
        elif row['method'] == 'random_acquisition': action = acquisition.action(obs['delta_low'],obs['delta_high'])
        else: action = local_baseline_action(core)
        obs, _, _, _, _ = core.step(action)
        if confirmation_noises is not None:
            from .terminal_confirmation import finish_and_confirm
            finish_and_confirm(core,confirmation_noises)
    info = core.info()
    if info['queries'] != source.queries or info['queries'] > row['query_cap']:
        raise ValueError('Query accounting differs from frozen cap')
    info.update(initial_accel=core.records[0][1].accel_mean,
                best_accel=core.records[core.best][1].accel_mean, events=core.events)
    if confirmation_noises is not None:info['terminal_confirmation']=core.terminal_confirmation
    if row['method']=='random_acquisition':
        info.update(acquisition_rule='uniform_legal_increment',acquisition_seed=row['acquisition_seed'],
            selection_rule='minimum_visited',budget_comparison='same_cap_not_same_realized_queries')
    if shifted:
        # Downstream execution/validation retains the original physical camera
        # coordinate system. Preserve both frames rather than relabel history.
        info['mechanical_selected_angles']=info['selected_angles']
        info['mount_yaw_offset_deg']=offset
        info['search_yaw_limit_deg']=row['search_yaw_limit_deg']
        if full:info['coordinate_family']=row['coordinate_family']
        info['selected_angles']=(np.asarray(info['selected_angles'])+[offset,0.]).tolist()
        info['current_angles']=(np.asarray(info['current_angles'])+[offset,0.]).tolist()
        info['event_angle_frame']='shifted_mount_mechanical'
    return info, source.last_example


def move_to_search_selection(source, info):
    """Replay the search frame exactly; avoid a second float Euler composition.

    Ordinary campaigns keep their original path. The mounting diagnostic must
    use the same requested pose matrix for scoring and manipulation; algebraic
    equality after changing frames does not guarantee bitwise render identity.
    """
    if 'mount_yaw_offset_deg' not in info:
        source.move_angles(info['selected_angles'])
        return
    from dataclasses import replace
    from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
    offset = info['mount_yaw_offset_deg']
    full=info.get('coordinate_family')=='heldout_installation_v1'
    if (offset not in ((-15.,0.,15.) if full else (-5.,0.,5.))
            or info['search_yaw_limit_deg'] != (15. if full else 25.)
            or info.get('coordinate_family') not in (None,'heldout_installation_v1')):
        raise ValueError('Unsupported selection mounting frame')
    target = np.asarray(info['mechanical_selected_angles'], dtype=float)
    physical = target + [offset, 0.]
    if not np.array_equal(physical, info['selected_angles']):
        raise ValueError('Mechanical and physical selection disagree')
    mount = PanTiltMount(source.mount.pose([offset, 0.]),
        replace(source.mount.config, yaw_limit_deg=info['search_yaw_limit_deg']))
    for point in mount.route(source.current - [offset, 0.], target):
        source.move(mount.pose(point))
    source.current = physical.copy()


def verify_actor(actor, controls):
    import torch
    for control in controls:
        with torch.inference_mode():
            action = actor.distribution(control['observation']).sample(deterministic=True)
        for key in ('stop','delta_deg','log_prob'):
            if not torch.equal(action[key], control['action'][key]):
                raise ValueError('Final actor CPU/runtime deployment control mismatch: '+key)


def audit_block(tasks, searches):
    if len(tasks) != 12 or len(searches) != 3:
        raise ValueError('Complete three-method/four-seed first block required')
    summary = summarize(tasks, searches)
    if summary['complete_pairs'] != 4:
        raise ValueError('Missing paired seeds in engineering gate')
    for repeat in range(4):
        rows = [r for r in tasks if r['repeat']==repeat]
        n = min(r['replans'] for r in rows)
        if any(r['execution_noise_sha256'][:n] != rows[0]['execution_noise_sha256'][:n] for r in rows):
            raise ValueError('Methods used different execution noise addresses')
    return dict(status='PASS_ENGINEERING_ONLY', tasks=12, paired_seeds=4,
                included_in_192=True, selected_on_task_success=False)


def run(config_path, output, *, execute=False, block_index=None, guard=None):
    from AlphaBrain.research.dsol.execution import dispatch_policy
    if execute: dispatch_policy.reject_production('benchmarks/LIBERO/continuous_evaluation.py:run')
    if not execute or read(config_path).get('authorization') != 'EXPLICIT_PAIRED_EVALUATION_ONLY':
        raise ValueError('Explicit paired evaluation authorization required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    calls = None; tasks = []; searches = []; begin = time.time()
    try:
        cfg, plan, _ = validate_inputs(config_path)
        rows=read(cfg['schedule']);limits=cfg['limits'];expected_tasks=192;expected_searches=48
        if block_index is not None:
            from AlphaBrain.research.dsol.data.continuous_evaluation_blocks import blocks
            block=blocks(plan,rows)[block_index]
            if block['kind']!='paired':raise ValueError('Paired block required')
            rows=block['rows'];limits=block['limits'];expected_tasks=block['tasks'];expected_searches=block['searches']
        if guard is not None:guard()
        import torch
        from .search_runtime import create_policy_runtime
        from .search_live import simulator, check_simulator_dependencies
        from .search_domain import exact_input
        from .local_view_runtime import RenderedViewSource
        from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
        from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
        torch.set_num_threads(2)
        payload = torch.load(cfg['actor'], map_location='cpu', weights_only=True)
        if payload['update']!=16 or payload['episodes']!=512:
            raise ValueError('Final-only completed actor required')
        actor = ContinuousActorCritic(4608).eval()
        actor.load_state_dict(payload['state_dict'], strict=True); actor.requires_grad_(False)
        controls = torch.load(cfg['actor_controls'], map_location='cpu', weights_only=True)
        verify_actor(actor, controls)
        write_json(output/'actor-runtime-admission.json',dict(status='PASS',controls=len(controls)))
        sims = {k:read(v['sim_config']) for k,v in cfg['cohorts'].items()}
        for sim in sims.values(): check_simulator_dependencies(sim)
        banks = {k:ExplicitFlowNoiseBank(Path(v['execution_noise']),verify_file=True) for k,v in cfg['cohorts'].items()}
        model_started=time.perf_counter()
        runtime = create_policy_runtime(sims['A'], output, sum(limits.values()))
        model_seconds=time.perf_counter()-model_started
        calls = EvaluationCalls(runtime, output, limits, guard=guard)
        initial_reference = None
        for index, row in enumerate(rows):
            spec = cfg['cohorts'][row['cohort']]; sim = sims[row['cohort']]
            directory = output/f'session-{index:03d}'; directory.mkdir()
            identity = {k:row[k] for k in ('cohort','initial','state_key','task','start','method')}
            calls.identity = identity; calls.stage = 'score'
            with np.load(spec['score_noise'],allow_pickle=False) as z:
                noises = z[spec['score_key']][row['initial'],:2].copy()
            started = time.time()
            with calls.timing.context(simulator(spec['sim_config'],sim,row['initial'],directory),'simulator') as request:
                info, selected = search(row,request,calls,noises,directory,actor,plan['actor']['score_scale'])
            with np.load(directory/'query-01.npz',allow_pickle=False) as z:
                first = {k:z[k].copy() for k in z.files}
            if index%3 == 0: initial_reference = first
            elif any(not np.array_equal(first[k], initial_reference[k]) for k in first):
                raise ValueError('Three methods differ at identical initial scoring query')
            info.update(identity, search_seconds=time.time()-started)
            write_json(directory/'search.json',info); searches.append(info)
            suite = next(t['suite'] for t in plan['evaluation']['tasks'] if t['task']==row['task'])
            for repeat in row['execution_repeats']:
                episode_dir = directory/f'episode-{repeat}'; episode_dir.mkdir()
                calls.stage = 'execution'; started = time.time()
                with calls.timing.context(simulator(spec['sim_config'],sim,row['initial'],episode_dir),'simulator') as request:
                    source = RenderedViewSource(request,calls,noises,episode_dir)
                    source.move_angles(info['selected_angles'])
                    response = request(dict(kind='begin_execution'))
                    exact_input(selected,response['example'])
                    result, arrays = rollout_local_episode(request,calls.infer,response['example'],
                        banks[row['cohort']],row['state_key'],repeat,identity,plan=plan,suite=suite,
                        record=lambda event: append_event(episode_dir/'events.jsonl',event))
                result.update(identity, queries=info['queries'], selected_angles=info['selected_angles'],
                              elapsed_seconds=time.time()-started)
                asset = episode_dir/'trajectory.npz'; np.savez_compressed(asset,**arrays)
                result.update(trajectory=str(asset),trajectory_sha256=sha(asset))
                write_json(episode_dir/'result.json',result); tasks.append(result)
                write_json(output/'progress.json',dict(status='RUNNING',tasks=len(tasks),
                    searches=len(searches),calls=calls.counts,elapsed_seconds=time.time()-begin))
            if index%3 == 2:
                if index == 2: write_json(output/'first-block-gate.json',audit_block(tasks,searches))
                write_json(output/f'paired-{index//3:02d}.json',summarize(tasks,searches))
        if len(tasks)!=expected_tasks or len(searches)!=expected_searches or sha(cfg['actor'])!=cfg['input_hashes'][cfg['actor']]:
            raise ValueError('Final coverage or frozen actor identity mismatch')
        write_json(output/'timing.json',dict(model_prepare_seconds=model_seconds,phases=calls.timing.rows))
        write_json(output/'results.json',dict(tasks=tasks,searches=searches,block_index=block_index))
        write_json(output/'status.json',dict(status='COMPLETE',tasks=expected_tasks,searches=expected_searches,
            calls=calls.counts,config_sha256=sha(config_path),elapsed_seconds=time.time()-begin,
            training_updates=0,summary=summarize(tasks,searches)))
    except BaseException as error:
        write_json(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),
            calls=calls.counts if calls else {},completed_tasks=len(tasks),automatic_retry=False))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:
        if not a.output:p.error('--output required')
        run(a.config,a.output,execute=True)
    else: print(json.dumps(validate_inputs(a.config)[2],indent=2))
