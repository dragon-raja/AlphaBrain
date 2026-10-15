"""One-shot, fail-closed transition between already authorized XR-1 panels."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import time

from .artifacts import read, sha, write


def process_running(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def _seed(initial, stream, repeat, replan=0):
    address = ['xr1_paired_v1', initial, stream, repeat, replan]
    return int.from_bytes(hashlib.sha256(json.dumps(address, separators=(',', ':')).encode()).digest()[:8], 'little') % (2**63)


def audit_complete(root, phase):
    """Return pending until workers exit; never gate on scientific success rate."""
    root = Path(root)
    plan = read(root / 'plan.json')
    launch = read(root / 'launch.json')
    expected_views = {'canonical': 1, 'perturbed': 4, 'remaining': 92}[phase]
    if (plan['phase'] != phase or len(plan['initial_ids']) != 40
            or len(set(plan['initial_ids'])) != 40 or len(plan['views']) != expected_views
            or plan['execution_repeats'] != 4 or plan['score_repeats'] != 8
            or plan['rollouts'] != 160 * expected_views or len(launch['workers']) != 8
            or launch['status'] != 'LAUNCHED'
            or sorted(w['shard'] for w in launch['workers']) != list(range(8))):
        raise ValueError('Panel scope or launch identity changed')
    complete = 0
    for worker in launch['workers']:
        status = read(Path(worker['output']) / 'status.json')
        if status['status'] == 'FAILED':
            raise RuntimeError(f"Worker failed: {worker['output']}: {status.get('error')}")
        if status['status'] != 'COMPLETE':
            if not process_running(worker['pid']):
                raise RuntimeError(f"Worker exited without completion: {worker['output']}")
        else:
            complete += 1
    if complete != 8 or any(process_running(w['pid']) for w in launch['workers']):
        return None
    manifest = read(root / 'source-manifest.json')
    if sha(root / 'source-manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for name, digest in manifest.items():
        if sha(Path(plan['source']) / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    for path, digest in ((Path(plan['panel']) / 'plan.json', plan['panel_sha256']),
                         (Path(plan['checkpoint']) / 'IMPORT_RECEIPT.json', plan['checkpoint_receipt_sha256'])):
        if sha(path) != digest:
            raise ValueError(f'Input identity changed: {path}')
    jobs = {j['initial_id']: j for j in plan['jobs']}
    records = []
    expected_files = set()
    for worker in launch['workers']:
        location = Path(worker['output'])
        identity = read(location / 'identity.json')
        for key in ('panel_sha256', 'checkpoint_receipt_sha256', 'views', 'noise_version', 'execution_repeats'):
            if identity[key] != plan[key]:
                raise ValueError(f'Worker identity mismatch: {key}')
        if identity['initials'] != plan['initial_ids'][worker['shard']::8]:
            raise ValueError('Worker initial assignment changed')
        for key, source_name in (('evaluator_sha256', 'xr1_evaluation.py'), ('adapter_sha256', 'xr1_policy.py')):
            if identity[key] != manifest['benchmarks/Robocasa365/' + source_name]:
                raise ValueError('Worker source identity changed')
        if read(location / 'status.json')['completed_rollouts'] != len(identity['initials']) * expected_views * 4:
            raise ValueError('Worker completion count mismatch')
        for initial in identity['initials']:
            for view in plan['views']:
                cell = location / 'cells' / initial / view
                scores = read(cell / 'scores.json')
                if len(scores['rows']) != 8 or sha(cell / 'scores.npz') != scores['arrays_sha256']:
                    raise ValueError('Missing/corrupt scoring trace')
                for repeat, row in enumerate(scores['rows']):
                    if row['seed'] != _seed(initial, 'score', repeat) or not math.isfinite(row['accel_p3']):
                        raise ValueError('Invalid scoring seed/value')
                for repeat in range(4):
                    path = cell / f'execution_{repeat}.json'
                    row = read(path)
                    expected_files.add(path)
                    if (row['repeat'] != repeat or row['complete_task_attempt'] is not True
                            or type(row['success']) is not bool or not 0 < row['steps'] <= row['horizon']
                            or not row['plans'] or sha(cell / f'trace_{repeat}.npz') != row['trace_sha256']):
                        raise ValueError(f'Invalid rollout: {path}')
                    if not (row['success'] or row['terminated'] or row['truncated'] or row['steps'] == row['horizon']):
                        raise ValueError(f'Incomplete unsuccessful rollout: {path}')
                    for index, action in enumerate(row['plans']):
                        if action['seed'] != _seed(initial, 'execution', repeat, index) or action['step'] != index * 16:
                            raise ValueError('Execution noise address/replan mismatch')
                    records.append(dict(initial=initial, view=view, repeat=repeat, success=row['success'],
                                        task=jobs[initial]['task'], split=jobs[initial]['split']))
    actual = set((root / 'workers').glob('*/cells/*/*/execution_*.json'))
    if actual != expected_files or len(records) != plan['rollouts']:
        raise ValueError('Missing, duplicate, or unexpected rollout cells')
    return dict(status='PASS', phase=phase, rollouts=len(records),
                successes=sum(r['success'] for r in records), records=records,
                plan_sha256=sha(root / 'plan.json'), checked_unix=time.time())


def supervise(canonical, output, record, poll_seconds=30, deadline_hours=12):
    """Launch exactly one perturbed panel from canonical's immutable source."""
    canonical, output, record = map(Path, (canonical, output, record))
    record.mkdir(parents=True, exist_ok=True)
    with (record / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = record / 'status.json'
        if status_path.exists():
            raise ValueError('Existing continuation record; manual audit required before recovery')
        state = dict(status='WAITING_CANONICAL', pid=os.getpid(), canonical=str(canonical),
                     output=str(output), started_unix=time.time(),
                     scope='160 canonical -> 640 perturbed; no training or selector expansion')
        write(status_path, state)
        try:
            if output.exists():
                raise FileExistsError('Destination exists; refusing duplicate launch')
            plan = read(canonical / 'plan.json')
            pinned = sha(canonical / 'plan.json')
            deadline = time.monotonic() + deadline_hours * 3600
            for phase, root in (('canonical', canonical), ('perturbed', output)):
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Continuation deadline exceeded; no automatic retry')
                    if sha(canonical / 'plan.json') != pinned:
                        raise ValueError('Canonical plan changed during wait')
                    audit = audit_complete(root, phase)
                    state['checked_unix'] = time.time()
                    write(status_path, state)
                    if audit is not None:
                        write(record / f'{phase}_audit.json', audit)
                        break
                    time.sleep(poll_seconds)
                if phase == 'canonical':
                    cmd = [plan['sim_python'], str(Path(plan['source']) / 'scripts/dsol_paper1/operations/launch_robocasa_xr1_panel.py')]
                    for key in ('panel', 'checkpoint', 'upstream', 'model_python', 'sim_python', 'source', 'pilot'):
                        cmd += ['--' + key.replace('_', '-'), plan[key]]
                    cmd += ['--output', str(output), '--phase', 'perturbed', '--execute']
                    state.update(status='LAUNCHING_PERTURBED', command=cmd)
                    write(status_path, state)
                    with (record / 'launch.log').open('xb') as log:
                        subprocess.run(cmd, cwd=plan['source'], env={**os.environ, 'PYTHONPATH': plan['source'],
                                       'OPENBLAS_NUM_THREADS': '1'}, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=300)
                    state.update(status='WAITING_PERTURBED', launched_unix=time.time())
                    write(status_path, state)
                    # Allow newly launched workers to create their status files.
                    time.sleep(poll_seconds)
            state.update(status='COMPLETE', completed_unix=time.time())
            write(status_path, state)
        except BaseException as exc:
            state.update(status='FAILED', error=repr(exc), stopped_unix=time.time())
            write(status_path, state)
            raise


def supervise_full(config):
    """Extend the existing five-view run with the disjoint remaining92, then analyze."""
    cfg = read(config)
    record = Path(cfg['record'])
    record.mkdir(parents=True, exist_ok=True)
    with (record / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = record / 'status.json'
        if status_path.exists() or Path(cfg['output']).exists():
            raise ValueError('Existing full-view continuation/output; refusing duplicate launch')
        state = dict(status='WAITING_FIVE_VIEW', pid=os.getpid(), started_unix=time.time(),
                     new_rollouts=14720, reused_rollouts=800, total_rollouts=15520,
                     config_sha256=sha(config), authorization='User explicitly requested same40 x all97 Accel/closed-loop validation')
        write(status_path, state)
        try:
            frozen = record / 'source'
            for directory in ('AlphaBrain', 'benchmarks', 'scripts', 'configs'):
                shutil.copytree(Path(cfg['source']) / directory, frozen / directory,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            manifest = {str(p.relative_to(frozen)): sha(p) for p in frozen.rglob('*') if p.is_file()}
            write(record / 'source-manifest.json', manifest)
            deadline = time.monotonic() + 96 * 3600
            while True:
                previous = read(Path(cfg['predecessor']) / 'status.json')
                if previous['status'] == 'FAILED':
                    raise RuntimeError('Five-view predecessor failed; no scope reduction or automatic retry')
                if previous['status'] == 'COMPLETE':
                    break
                if not process_running(previous['pid']) or time.monotonic() > deadline:
                    raise RuntimeError('Five-view controller exited or full-view deadline exceeded')
                state['checked_unix'] = time.time(); write(status_path, state)
                time.sleep(30)
            for phase in ('canonical', 'perturbed'):
                audit = audit_complete(cfg[phase], phase)
                if audit is None:
                    raise RuntimeError('Predecessor reports completion before workers exited')
                write(record / (phase + '_reuse_audit.json'), audit)
            plan = read(Path(cfg['canonical']) / 'plan.json')
            # Preserve rollout and model inference functions; only view partition changes.
            import ast
            for relative, function in (('benchmarks/Robocasa365/xr1_evaluation.py', 'rollout'),):
                trees = [ast.parse((root / relative).read_text()) for root in (Path(plan['source']), frozen)]
                nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function) for tree in trees]
                if ast.dump(nodes[0]) != ast.dump(nodes[1]):
                    raise ValueError('Rollout implementation changed; new runtime admission required')
            for relative in ('benchmarks/Robocasa365/xr1_policy.py', 'benchmarks/Robocasa365/xr1_bridge.py'):
                if sha(Path(plan['source']) / relative) != sha(frozen / relative):
                    raise ValueError('Policy/bridge changed; new runtime admission required')
            cmd = [plan['sim_python'], str(frozen / 'scripts/dsol_paper1/operations/launch_robocasa_xr1_panel.py')]
            for key in ('panel', 'checkpoint', 'upstream', 'model_python', 'sim_python', 'pilot'):
                cmd += ['--' + key.replace('_', '-'), plan[key]]
            cmd += ['--source', str(frozen), '--output', cfg['output'], '--phase', 'remaining', '--execute']
            state.update(status='LAUNCHING_REMAINING92', command=cmd); write(status_path, state)
            environment = {**os.environ, 'PYTHONPATH': str(frozen), 'OPENBLAS_NUM_THREADS': '1'}
            with (record / 'launch.log').open('xb') as log:
                subprocess.run(cmd, cwd=frozen, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            state.update(status='RUNNING_REMAINING92', launched_unix=time.time()); write(status_path, state)
            time.sleep(30)
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('Full-view deadline exceeded; workers/results retained')
                audit = audit_complete(cfg['output'], 'remaining')
                if audit is not None:
                    write(record / 'remaining_audit.json', audit)
                    break
                state['checked_unix'] = time.time(); write(status_path, state)
                time.sleep(30)
            state.update(status='ANALYZING'); write(status_path, state)
            command = [plan['sim_python'], '-m', 'AlphaBrain.research.dsol.analysis.xr1_fullview',
                       '--panel', plan['panel'], '--roots', cfg['canonical'], cfg['perturbed'], cfg['output'],
                       '--output', str(record / 'analysis')]
            if cfg.get('pi05_subspaces'):
                command += ['--pi05-subspaces', cfg['pi05_subspaces']]
            with (record / 'analysis.log').open('xb') as log:
                subprocess.run(command, cwd=frozen, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            state.update(status='COMPLETE', completed_unix=time.time()); write(status_path, state)
        except BaseException as exc:
            state.update(status='FAILED', error=repr(exc), stopped_unix=time.time()); write(status_path, state)
            raise
