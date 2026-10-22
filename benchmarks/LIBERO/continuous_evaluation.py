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
        checkpoint = selected_checkpoint(sim)
        if checkpoint['weights_sha256'] != plan['operation_policy']['checkpoint_sha256']:
            raise ValueError('Design and simulator operation-model identities differ')
        required += [spec['sim_config'], spec['score_noise'], spec['execution_noise'], sim['inputs'],
                     str(Path(checkpoint['path'])/'model.safetensors')]
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
    def __init__(self, runtime, output, limits):
        self.runtime, self.output, self.limits = runtime, Path(output), limits
        self.counts = dict.fromkeys(limits, 0)
        self.stage = None
        self.identity = {}

    def infer(self, example, noise, identity, capture=True):
        from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256
        stage = self.stage
        if stage not in self.counts or self.counts[stage] >= self.limits[stage]:
            raise ValueError('Stage call budget exhausted')
        self.counts[stage] += 1
        address = dict(identity, **self.identity, stage=stage)
        append_event(self.output/'model-attempts.jsonl', dict(index=sum(self.counts.values()),
            stage_index=self.counts[stage], unix_time=time.time(), identity=address,
            input_sha256=input_identity(example), noise_sha256=tensor_sha256(noise)))
        value = self.runtime.infer(example, noise, address, capture=capture)
        if self.runtime.calls != sum(self.counts.values()):
            raise ValueError('Runtime and durable ledger diverged')
        return value


def actor_action(actor, observation):
    import torch
    from AlphaBrain.research.dsol.acquisition.continuous_search_gym import continuous_observation
    obs = continuous_observation(observation)
    tensor = {k:torch.as_tensor(v[None]) for k,v in obs.items()}
    with torch.inference_mode():
        value = actor.distribution(tensor).sample(deterministic=True)
    return dict(stop=bool(value['stop'][0]), delta_deg=value['delta_deg'][0].numpy())


def search(row, request, calls, noises, directory, actor, scale):
    from .local_view_runtime import RenderedViewSource
    from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewSearch
    from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
    from AlphaBrain.research.dsol.acquisition.local_view_gym import local_baseline_action
    source = RenderedViewSource(request, calls, noises, directory)
    source.move_angles(row['angles_deg']); first = source.query()
    original, pending = source.query, [first]
    source.query = lambda: pending.pop() if pending else original()
    cls = LocalViewSearch if row['method']=='local_near_best_accel' else ContinuousSearch
    core = cls(source.mount, source, initial_angles=row['angles_deg'],
        physical_state_id=source.physical, wrist=source.last_values[0]['wrist'],
        score_scale=scale, seed=row['proposal_seed'])
    obs = core.start()
    while core.status == 'ACTIVE':
        if row['method'] == 'stay': action = dict(stop=True, delta_deg=np.zeros(2))
        elif row['method'] == 'continuous_ppo': action = actor_action(actor, obs)
        else: action = local_baseline_action(core)
        obs, _, _, _, _ = core.step(action)
    info = core.info()
    if info['queries'] != source.queries or info['queries'] > row['query_cap']:
        raise ValueError('Query accounting differs from frozen cap')
    info.update(initial_accel=core.records[0][1].accel_mean,
                best_accel=core.records[core.best][1].accel_mean, events=core.events)
    return info, source.last_example


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


def run(config_path, output, *, execute=False):
    if not execute or read(config_path).get('authorization') != 'EXPLICIT_PAIRED_EVALUATION_ONLY':
        raise ValueError('Explicit paired evaluation authorization required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    calls = None; tasks = []; searches = []; begin = time.time()
    try:
        cfg, plan, _ = validate_inputs(config_path)
        import torch
        from .search_runtime import PolicyRuntime
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
        runtime = PolicyRuntime(sims['A'], output, sum(cfg['limits'].values()))
        calls = EvaluationCalls(runtime, output, cfg['limits'])
        initial_reference = None
        for index, row in enumerate(read(cfg['schedule'])):
            spec = cfg['cohorts'][row['cohort']]; sim = sims[row['cohort']]
            directory = output/f'session-{index:03d}'; directory.mkdir()
            identity = {k:row[k] for k in ('cohort','initial','state_key','task','start','method')}
            calls.identity = identity; calls.stage = 'score'
            with np.load(spec['score_noise'],allow_pickle=False) as z:
                noises = z[spec['score_key']][row['initial'],:2].copy()
            started = time.time()
            with simulator(spec['sim_config'],sim,row['initial'],directory) as request:
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
                with simulator(spec['sim_config'],sim,row['initial'],episode_dir) as request:
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
        if len(tasks)!=192 or len(searches)!=48 or sha(cfg['actor'])!=cfg['input_hashes'][cfg['actor']]:
            raise ValueError('Final coverage or frozen actor identity mismatch')
        write_json(output/'status.json',dict(status='COMPLETE',tasks=192,searches=48,
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
