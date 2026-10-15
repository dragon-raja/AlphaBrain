"""Isolated GPU capacity measurement, outside scientific trial accounting.

Drain a standard queue's physical GPU first. Replay frozen policy inputs and
real five-window probes at each concurrency. Resume production in finally.
This never promotes a concurrency setting automatically.
"""
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.execution.resources import resource_status, set_drain, choose_capacity
from AlphaBrain.research.dsol.execution.queue import load
from .probe_queue import ProbeBackend, process_identity, group_members


def gpu_sample(gpu):
    values = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    used, free, utilization = map(int, values.strip().split(','))
    return dict(unix=time.time(), used_mib=used, free_mib=free, utilization=utilization)


def assert_idle(gpu):
    pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    for pid in pids.splitlines():
        path = Path('/proc') / pid.strip() / 'cmdline'
        try:
            command = path.read_bytes().replace(b'\0', b' ').decode()
        except FileNotFoundError:
            continue
        if '/gpu_compute_keepalive/gpu_compute_keepalive.py' not in command:
            raise RuntimeError('GPU is not exclusively available for capacity test')


def stop_test_processes(processes):
    """Only process groups created by this benchmark, never production groups."""
    for proc, identity in processes:
        if group_members(identity['group']):
            current = process_identity(proc.pid)
            if current is not None and any(current[k] != identity[k] for k in ('start', 'boot', 'group')):
                raise RuntimeError('Test process identity changed; refusing signal')
            os.killpg(identity['group'], signal.SIGTERM)
    deadline = time.monotonic() + 5
    while any(group_members(i['group']) for _, i in processes) and time.monotonic() < deadline:
        time.sleep(.1)
    for proc, identity in processes:
        if group_members(identity['group']):
            os.killpg(identity['group'], signal.SIGKILL)
        proc.wait(timeout=10)


def run_stage(cfg, plan, root, kind, workers):
    stage = root / f'{kind}-{workers}'
    stage.mkdir(exist_ok=False)
    runtime = plan['runtime']; gpu = cfg['gpu']; processes = []; logs = []; samples = []
    env = {**os.environ, **runtime['environment'], 'CUDA_VISIBLE_DEVICES': str(gpu)}
    started = time.monotonic(); problem = None; validation = []
    assert_idle(gpu)
    try:
        for index in range(workers):
            out = stage / f'worker-{index}'
            if kind == 'anchor':
                replay = deepcopy(read(cfg['anchor_config']))
                replay['output'] = str(out.with_suffix('.json'))
                replay_config = out.with_suffix('.config.json'); write(replay_config, replay)
                command = [runtime['python'], '-c',
                    'import sys; from benchmarks.Robocasa365.policy_replay import replay_score_anchor; replay_score_anchor(sys.argv[1])',
                    str(replay_config)]
                cwd = cfg['source']; env = {**env, 'PYTHONPATH': cwd}
            else:
                command = [runtime['python'], '-m', 'benchmarks.Robocasa365.short_view_probe',
                    '--config', runtime['config'], '--model', runtime['model'], '--output', str(out),
                    '--initials', cfg['initial'], '--views', *cfg['views'], '--execute']
                cwd = runtime['source']; env = {**env, 'PYTHONPATH': cwd}
            log = (stage / f'worker-{index}.log').open('xb'); logs.append(log)
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(process.pid)
            if identity is None:
                process.wait(timeout=10)
                raise RuntimeError('Test worker exited before identity capture')
            processes.append((process, identity))
            write(stage / f'worker-{index}.process.json', dict(command=command, **identity))
        while any(p.poll() is None for p, _ in processes):
            sample = gpu_sample(gpu); samples.append(sample)
            write(stage / 'progress.json', dict(status='RUNNING', workers=workers, kind=kind,
                  seconds=time.monotonic()-started, latest=sample))
            if sample['free_mib'] < cfg['abort_free_mib']:
                raise RuntimeError('Capacity test reached memory safety floor')
            if any(p.poll() not in (None, 0) for p, _ in processes):
                raise RuntimeError('Capacity test worker failed; see preserved log')
            if time.monotonic()-started > cfg['stage_timeout_seconds']:
                raise TimeoutError('Capacity test deadline')
            time.sleep(2)
        if any(p.returncode != 0 for p, _ in processes):
            raise RuntimeError('Capacity test worker failed')
        work_seconds = time.monotonic()-started
        if kind == 'anchor':
            for index in range(workers):
                receipt = read(stage / f'worker-{index}.json')
                if receipt['status'] != 'PASS':
                    raise ValueError('Fixed-input policy replay failed')
                validation.append(receipt)
        else:
            backend = ProbeBackend(plan, stage)
            for index in range(workers):
                validation.append(backend.validate(dict(output=str(stage/f'worker-{index}'),
                    payload=dict(initial=cfg['initial'], views=cfg['views']))))
    except BaseException as exc:
        problem = repr(exc); work_seconds = time.monotonic()-started
    finally:
        stop_test_processes(processes)
        for log in logs:
            log.close()
    result = dict(status='PASS' if problem is None else 'HOLD', error=problem, kind=kind,
        workers=workers, seconds=work_seconds, quality_pass=problem is None,
        throughput=(workers * (len(cfg['views'])*8 if kind=='probe' else 8))/work_seconds if problem is None else 0,
        headroom_mib=min((s['free_mib'] for s in samples), default=0),
        peak_used_mib=max((s['used_mib'] for s in samples), default=0),
        required_headroom_mib=cfg['required_headroom_mib'], validation=validation, telemetry=samples)
    write(stage/'result.json', result)
    return result


def run(config_file):
    cfg = read(config_file); root = Path(cfg['output']); control = Path(cfg['queue'])
    root.mkdir(parents=True, exist_ok=False)
    plan, _ = load(control)
    if plan['runtime']['model'] != 'official':
        raise ValueError('This replay adapter currently certifies official pi05 only')
    if cfg['workers'] != [1, 2, 3] or not 0 < cfg['abort_free_mib'] < cfg['required_headroom_mib']:
        raise ValueError('Expected explicit 1/2/3 capacity ladder and memory reserve')
    result = dict(status='WAITING_FOR_DRAIN', config_sha256=sha(config_file),
                  scope='Engineering only; not scientific trials; no automatic promotion', stages=[],
                  manifest_sha256=sha(control/'manifest.json'),
                  worker_config_sha256=sha(plan['runtime']['config']),
                  worker_source_sha256=sha(Path(plan['runtime']['source'])/'benchmarks/Robocasa365/short_view_probe.py'),
                  anchor_config_sha256=sha(cfg['anchor_config']))
    write(root/'status.json', result)
    with (control/f'capacity-{cfg["gpu"]}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            deadline = time.monotonic()+cfg['drain_timeout_seconds']
            while True:
                device = next(r for r in resource_status(control)['devices']
                              if r['host']==cfg['host'] and r['gpu']==str(cfg['gpu']))
                if not device['drained']:
                    raise ValueError('Explicit queue drain required before benchmark')
                if device['quiescent']:
                    break
                if time.monotonic()>deadline:
                    raise TimeoutError('Drain deadline; production work was not interrupted')
                time.sleep(5)
            result['hardware'] = subprocess.check_output(['nvidia-smi','-i',str(cfg['gpu']),
                '--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
            for count in cfg['workers']:
                for kind in ('anchor', 'probe'):
                    result.update(status='RUNNING', current=dict(workers=count, kind=kind))
                    write(root/'status.json',result)
                    row = run_stage(cfg, plan, root, kind, count)
                    result['stages'].append(row);write(root/'status.json',result)
                    if row['status']!='PASS':
                        break
                if result['stages'][-1]['status']!='PASS':
                    break
            result.update(status='COMPLETE', recommended_probe_workers=choose_capacity(
                [s for s in result['stages'] if s['kind']=='probe']),
                production_promotion=False,
                limitation='Short probes only; endpoint rollout and training require separate capacity admission')
        except BaseException as exc:
            result.update(status='HOLD', error=repr(exc))
        finally:
            set_drain(control,cfg['host'],str(cfg['gpu']),reason='Capacity benchmark finished or stopped',enabled=False)
            result['production_resumed']=True
            write(root/'status.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    def interrupt(_signal,_frame):
        raise KeyboardInterrupt('Capacity test interrupted; clean owned workers and resume production')
    for signum in (signal.SIGTERM,signal.SIGHUP):signal.signal(signum,interrupt)
    status=run(args.config)['status'];print(status,flush=True)
    raise SystemExit(0 if status=='COMPLETE' else 1)
