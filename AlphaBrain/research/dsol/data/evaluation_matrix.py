"""Frozen paired evaluation addresses and balanced two-host work assignment."""
from collections import defaultdict
import hashlib
import json
from numbers import Integral
from pathlib import Path
import shutil

import numpy as np

from .artifacts import read, sha, write

NUMERIC_FLAGS = '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true'
NOISE_VERSION = 'robocasa_stage2_paired_flow_v1'


def flow_noise(initial_id, stream, repeat, replan=0):
    """Common random numbers across views/models, not tied to call history."""
    if (stream not in ('score', 'execution') or not initial_id
            or any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in (repeat, replan))):
        raise ValueError('Invalid noise address')
    address = [NOISE_VERSION, initial_id, stream, int(repeat), int(replan)]
    seed = int.from_bytes(hashlib.sha256(json.dumps(address, separators=(',', ':')).encode()).digest()[:8], 'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((50, 32), dtype=np.float32)


def assign_workers(jobs, arms=('canonical', 'broad'), hosts=('fresh', 'gnu'), gpus=8):
    if not jobs or len({j['initial_id'] for j in jobs}) != len(jobs):
        raise ValueError('Unique, nonempty initial population required')
    groups = defaultdict(list)
    for job in jobs:
        groups[(job['split'], job['task'])].append(job)
    host_jobs = {host: [] for host in hosts}
    for _, group in sorted(groups.items()):
        for index, job in enumerate(sorted(group, key=lambda j: (j['scene_seed'], j['initial_id']))):
            host_jobs[hosts[index % len(hosts)]].append(job['initial_id'])
    workers = []
    for host, initials in host_jobs.items():
        for gpu in range(gpus):
            for arm in arms:
                workers.append(dict(worker_id=f'{host}-gpu{gpu}-{arm}', host=host, gpu=gpu,
                                    arm=arm, initials=initials[gpu::gpus]))
    return workers


def validate_plan(plan):
    if (plan.get('format') != 'robocasa_stage2_matrix_v1' or plan['execution_repeats'] != 4
            or plan['score_repeats'] != 8 or plan['flow_steps'] != 10 or plan['replan_steps'] != 25
            or plan['noise_version'] != NOISE_VERSION or plan['numeric_flags'] != NUMERIC_FLAGS):
        raise ValueError('Unrecognized frozen evaluation conventions')
    for field in ('reference_plan', 'reference_release'):
        if sha(plan[field]) != plan[field + '_sha256']:
            raise ValueError('Reference release identity changed')
    reference = read(plan['reference_plan'])
    views = [v['view_id'] for v in reference['catalog']['views']]
    initials = {j['initial_id'] for j in reference['jobs']}
    if views != plan['view_ids'] or len(views) != 97 or len(initials) != 400:
        raise ValueError('Formal population must remain 400 initials by 97 views')
    if set(plan['models']) != {'canonical', 'broad'}:
        raise ValueError('Expected matched two-arm comparison')
    if (plan['renderer'] != 'native_4x_msaa' or not isinstance(plan['policy_only_render'], bool)
            or not set(plan['pilot_view_ids']).issubset(views)):
        raise ValueError('Unrecognized renderer or pilot view population')
    jobs = {j['initial_id']: j for j in reference['jobs']}
    identities = set()
    for worker in plan['workers']:
        if worker['worker_id'] in identities or worker['arm'] not in plan['models']:
            raise ValueError('Duplicate or invalid worker')
        identities.add(worker['worker_id'])
        if (not set(worker['pilot_initials']).issubset(worker['initials'])
                or any(jobs[i]['split'] != 'pretrain' for i in worker['pilot_initials'])):
            raise ValueError('Engineering pilot must use assigned seen-scene initials')
    for arm in plan['models']:
        assigned = [i for w in plan['workers'] if w['arm'] == arm for i in w['initials']]
        if len(assigned) != len(initials) or set(assigned) != initials:
            raise ValueError('Worker assignment drops or duplicates initials')
    for initial in initials:
        assigned = [(w['host'], w['gpu']) for w in plan['workers'] if initial in w['initials']]
        if len(assigned) != 2 or len(set(assigned)) != 1:
            raise ValueError('Models must share initial-to-host/GPU assignment')
    return reference


def prepare_plan(program_file, dualhost_file, output, source_root, *, policy_only_render=False):
    """Freeze scientific population, source and model identities before workers."""
    program, dual = read(program_file), read(dualhost_file)
    scope = program['evaluation_scope']
    if scope['rollout_release_gate'] != 'USER_APPROVED_NATIVE_RENDER_VARIABILITY_8_SCORE_4_EXECUTION':
        raise ValueError('User-approved native renderer and 8+4 budget required')
    output, source_root = Path(output), Path(source_root)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    for name in ('AlphaBrain', 'benchmarks', 'scripts', 'configs'):
        shutil.copytree(source_root/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    write(output/'source-manifest.json', manifest)
    reference = read(scope['current_reference_plan'])
    workers = assign_workers(reference['jobs'])
    jobs = {j['initial_id']: j for j in reference['jobs']}
    pilot_tasks = ('CloseBlenderLid', 'CloseFridge', 'PickPlaceSinkToCounter', 'TurnOnMicrowave')
    for worker in workers:
        task = pilot_tasks[worker['gpu'] % len(pilot_tasks)]
        worker['pilot_initials'] = [next(i for i in worker['initials']
            if jobs[i]['task'] == task and jobs[i]['split'] == 'pretrain')]
    models = {}
    for arm, field, run_name in (('canonical', 'canonical_output', 'canonical_matched'),
                                 ('broad', 'broad_output', 'broad_exterior_matched')):
        checkpoint = Path(dual[field])/'checkpoints/pi05_pretrain_human300_all_lora_fast25'/run_name/'8250'
        model_files = {}
        for file in sorted(checkpoint.rglob('*')):
            if not file.is_file() or file.relative_to(checkpoint).parts[0] == 'train_state':
                continue
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(block)
            stat = file.stat()
            model_files[str(file.relative_to(checkpoint))] = dict(sha256=digest.hexdigest(), bytes=stat.st_size,
                                                                 mtime_ns=stat.st_mtime_ns)
        models[arm] = dict(checkpoint=str(checkpoint), config='pi05_pretrain_human300_all_lora_fast25',
                           norm_sha256=sha(checkpoint/'assets/norm_stats.json'), files=model_files)
    plan = dict(format='robocasa_stage2_matrix_v1', release_gate=scope['rollout_release_gate'],
        reference_plan=scope['current_reference_plan'], reference_release=scope['current_reference_release'],
        reference_plan_sha256=sha(scope['current_reference_plan']),
        reference_release_sha256=sha(scope['current_reference_release']),
        source=str(source), source_manifest=str(output/'source-manifest.json'),
        source_manifest_sha256=sha(output/'source-manifest.json'),
        models=models, workers=workers, execution_repeats=4, score_repeats=8,
        flow_steps=10, replan_steps=25, noise_version=NOISE_VERSION, numeric_flags=NUMERIC_FLAGS,
        renderer=scope['renderer'], task_horizons=scope['native_task_horizons'],
        policy_only_render=policy_only_render,
        view_ids=[v['view_id'] for v in reference['catalog']['views']], pilot_view_ids=['canonical', 'train_000'],
        interpretation='Native renderer residual variability retained; initial-specific empirical oracle averages noise before maximizing views',
        target_days=scope['user_target_dualhost_days'])
    validate_plan(plan)
    write(output/'plan.json', plan)
    return plan
