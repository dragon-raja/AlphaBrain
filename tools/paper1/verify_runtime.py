"""Explicit GPU migration gate, reusing the frozen experiment's 48 gate keys.

Default prepares a plan only. --execute runs diagnostic episodes outside the
scientific ledger. No training, keeper signals, or implicit full-matrix launch.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def signature(row):
    return (row['aa0']['trace_sha256'], row['success'], row['completion_steps'],
            row['aa0']['counts']['calls'], row['aa0']['counts']['steps'])


def environment(experiment, repo, *, sim, gpu=None):
    env = dict(os.environ)
    runtime = experiment / 'runtime'
    shared = repo / ('scripts/vla_shared' if (repo / 'scripts/vla_shared').is_dir() else 'scripts/cabi_vla')
    dependencies = runtime / ('sim/lib/python3.8/site-packages' if sim else 'policy/lib/python3.12/site-packages')
    env.update(PYTHONPATH=os.pathsep.join(map(str, [repo, shared, dependencies,
               runtime / 'openpi-src', runtime / 'openpi-client-src'])),
               OMP_NUM_THREADS='1' if sim else '2', OPENBLAS_NUM_THREADS='1' if sim else '2',
               MKL_NUM_THREADS='1' if sim else '2', PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', ALPHABRAIN_DISABLE_AUTO_DOWNLOAD='1',
               PRETRAINED_MODELS_DIR='/share/longjunyu/alphabrain/pretrained_models', IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg')
    if gpu is None:
        env.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def prepare(args):
    assert not args.output.exists(), 'Choose a new diagnostic output; never overwrite a prior run'
    release = read(args.experiment / 'release.json')
    assert len(release['states']) == 32 and len(release['gate_indices']) == 48
    assert (release['candidate_count'], release['noise_repeats'], release['offsamples'],
            release['replan_steps'], release['denoising_steps'], release['settle_steps']) == (97, 32, 0, 5, 10, 10)
    args.output.mkdir(parents=True)
    (args.output / 'logs').mkdir()
    baseline = args.experiment / 'repo'
    for name, expected in release['source_hashes'].items():
        assert sha(name) == expected, 'Frozen source changed: ' + name
    for name, expected in release['initial_source_hashes'].items():
        assert sha(name) == expected, 'Initial source changed: ' + name
    for model in release['models'].values():
        assert sha(Path(model['path']) / 'model.safetensors') == model['weights_sha256']
    bank = read(release['noise_bank']['path'])
    assert sha(release['noise_bank']['path']) == release['noise_bank']['manifest_sha256']
    assert sha(bank['noise_file']) == release['noise_bank']['file_sha256']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip()
    current = args.output / 'source/current'
    current.mkdir(parents=True)
    payload = subprocess.check_output(['git', 'archive', commit, 'AlphaBrain', 'scripts', 'configs'], cwd=args.repo)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        archive.extractall(current, filter='data')
    probe = args.output / 'source/runtime_probe.py'
    shutil.copy2(Path(__file__).with_name('runtime_probe.py'), probe)
    versions = {'frozen': dict(repo=str(baseline), entry=str(baseline / 'scripts/dsol_paper1/standard_initial_aa0_v1.py'),
                               server=str(baseline / 'scripts/cabi_vla/serve_alphabrain_pi05_websocket.py')),
                'current': dict(repo=str(current), entry=str(current / 'scripts/dsol_paper1/operations/controllers/standard_initial_aa0_v1.py'),
                                server=str(current / 'scripts/vla_shared/serve_alphabrain_pi05_websocket.py'))}
    for v in versions.values():
        v.update(entry_sha256=sha(v['entry']), server_sha256=sha(v['server']))
    historical = {}
    for model in ['canonical', 'broad']:
        historical[model] = {}
        for index in release['gate_indices']:
            path = args.experiment / 'hosts/fresh/gate-a' / model / f'{index//3104:02d}' / f'{index:06d}.json'
            row = read(path)
            historical[model][str(index)] = dict(path=str(path), sha256=sha(path), signature=signature(row))
    plan = dict(scientific_ledger=False, experiment=str(args.experiment), release_sha256=sha(args.experiment / 'release.json'),
                current_commit=commit, source_archive_sha256=hashlib.sha256(payload).hexdigest(),
                versions=versions, probe=str(probe), probe_sha256=sha(probe),
                gate_indices=release['gate_indices'], historical=historical, gpu_ids=args.gpus,
                base_port=args.base_port, episodes=192, workers=8, stop_on_mismatch=True,
                criteria='exact full trace, per-request inputs/actions/noise/denoising times; also match historical gate',
                controls={k: release[k] for k in ('candidate_count', 'noise_repeats', 'offsamples', 'replan_steps', 'denoising_steps', 'settle_steps')})
    write(args.output / 'plan.json', plan)
    return plan, release


def compare(output, plan, model, index):
    rows = [read(output / 'episodes' / model / v / str(index) / 'episode.json') for v in ('frozen', 'current')]
    calls = [read(output / 'episodes' / model / v / str(index) / 'calls.json') for v in ('frozen', 'current')]
    expected = tuple(plan['historical'][model][str(index)]['signature'])
    return dict(model=model, index=index, exact_new_old=signature(rows[0]) == signature(rows[1]),
                exact_requests_noise_actions_times=calls[0] == calls[1],
                frozen_matches_historical=signature(rows[0]) == expected,
                current_matches_historical=signature(rows[1]) == expected,
                frozen_signature=signature(rows[0]), current_signature=signature(rows[1]))


def require_comparison(row):
    assert all(row[k] for k in ('exact_new_old', 'exact_requests_noise_actions_times',
                                'frozen_matches_historical', 'current_matches_historical')), row


def run(args, plan, release):
    children = []
    halted = threading.Event()
    started = time.monotonic()
    status = lambda **value: write(args.output / 'status.json', dict(elapsed_seconds=time.monotonic()-started, **value))
    def launch(command, log, env, cwd):
        assert not halted.is_set(), 'Validation stopped'
        with log.open('x') as stream:
            p = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p)
        if halted.is_set() and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
        return p

    def abort():
        halted.set()
        for p in list(children):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)

    def interrupt(*_):
        raise InterruptedError('User interrupted this validation')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        status(phase='STARTING_POLICY_SERVERS')
        servers = {}
        for slot, (model, version) in enumerate((m, v) for m in ['canonical', 'broad'] for v in ['frozen', 'current']):
            gpu, port = args.gpus[slot], args.base_port + slot
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', port)) != 0, 'Port is already in use'
            v = plan['versions'][version]
            out = args.output / 'servers' / (model + '-' + version)
            command = [sys.executable, plan['probe'], 'serve', '--entry', v['server'], '--repo', v['repo'],
                       '--output', out, '--checkpoint', release['models'][model]['path'], '--port', port]
            p = launch(command, args.output / 'logs' / (model+'-'+version+'-server.log'),
                       environment(args.experiment, Path(v['repo']), sim=False, gpu=gpu), v['repo'])
            servers[model, version] = dict(process=p, port=port, gpu=gpu)
        deadline = time.monotonic() + 600
        while True:
            ready = []
            for s in servers.values():
                assert s['process'].poll() is None, 'Policy server failed; inspect logs'
                with socket.socket() as sock:
                    ready.append(sock.connect_ex(('127.0.0.1', s['port'])) == 0)
            if all(ready): break
            assert time.monotonic() < deadline, 'Policy startup timeout'
            time.sleep(2)
        for model, version in servers:
            identity = read(args.output / 'servers' / (model+'-'+version) / 'server-identity.json')
            assert identity['num_inference_steps'] == 10
            source = Path(plan['versions'][version]['repo']).resolve()
            modules = {n: r for n, r in identity['modules'].items() if n.startswith('AlphaBrain')}
            assert len(modules) > 10
            assert all(Path(r['path']).is_relative_to(source) for r in modules.values()), 'Mixed model source roots'

        def episode(model, version, index):
            s, v = servers[model, version], plan['versions'][version]
            out = args.output / 'episodes' / model / version / str(index)
            command = [args.experiment / 'runtime/python38/bin/python3.8', plan['probe'], 'episode',
                       '--entry', v['entry'], '--repo', v['repo'], '--output', out,
                       '--release', args.experiment / 'release.json', '--index', index,
                       '--model', model, '--port', s['port'], '--gpu', s['gpu']]
            if version == 'current':
                command.append('--require-local-modules')
            p = launch(command, args.output / 'logs' / f'{model}-{version}-{index}.log',
                       environment(args.experiment, Path(v['repo']), sim=True), v['repo'])
            assert p.wait(timeout=900) == 0, f'Episode failed: {model}/{version}/{index}'
            return model, version, index

        first = plan['gate_indices'][0]
        status(phase='RUNNING_PILOT', completed=0, total=192)
        pilot_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                for future in as_completed([pool.submit(episode, 'canonical', version, first) for version in ['frozen', 'current']]):
                    future.result()
            except BaseException:
                abort()
                raise
        pilot = compare(args.output, plan, 'canonical', first)
        write(args.output / 'pilot.json', dict(comparison=pilot, seconds=time.monotonic()-pilot_start))
        require_comparison(pilot)
        status(phase='PILOT_PASS_RUNNING_GATE', completed=2, total=192, pilot_seconds=time.monotonic()-pilot_start)
        jobs = [(m, v, i) for i in plan['gate_indices'] for m in ['canonical', 'broad'] for v in ['frozen', 'current']
                if not (m == 'canonical' and i == first)]
        done, completed_keys, comparisons = 2, {('canonical', v, first) for v in ['frozen', 'current']}, [pilot]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(episode, *job) for job in jobs]
            try:
                for future in as_completed(futures):
                    model, version, index = future.result()
                    completed_keys.add((model, version, index)); done += 1
                    if all((model, v, index) in completed_keys for v in ('frozen', 'current')):
                        row = compare(args.output, plan, model, index)
                        comparisons.append(row)
                        write(args.output / 'comparisons.json', comparisons)
                        require_comparison(row)
                    status(phase='RUNNING_GATE', completed=done, total=192, compared_pairs=len(comparisons))
            except BaseException:
                for future in futures:
                    future.cancel()
                abort()
                raise
        assert len(comparisons) == 96
        assert sha(args.experiment / 'release.json') == plan['release_sha256']
        write(args.output / 'completion.json', dict(status='PASS_RUNTIME_MIGRATION_GATE', scientific_ledger=False,
              episodes=192, paired_cases=96, exact_comparisons=96, current_commit=plan['current_commit'],
              elapsed_seconds=time.monotonic()-started, plan_sha256=sha(args.output / 'plan.json')))
        status(phase='PASS', completed=192, total=192)
    except BaseException as exc:
        status(phase='FAILED_REQUIRES_REVIEW', error=str(exc))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            if child.poll() is None:
                try: child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs=4, default=[0, 1, 2, 3])
    p.add_argument('--base-port', type=int, default=32180)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    assert len(set(args.gpus)) == 4
    # Existing keepers remain untouched. Refuse to share selected GPUs with other jobs.
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    selected = {line.split(',')[1].strip() for line in devices.splitlines() if int(line.split(',')[0]) in args.gpus}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    for line in processes.splitlines():
        pid, gpu = [x.strip() for x in line.split(',')]
        if gpu in selected:
            command = Path('/proc', pid, 'cmdline').read_bytes()
            assert b'gpu_compute_keepalive.py' in command, 'Selected GPU has an existing non-keeper workload'
    plan, release = prepare(args)
    print(json.dumps(dict(output=str(args.output), episodes=192, execute=args.execute)), flush=True)
    if args.execute: run(args, plan, release)


if __name__ == '__main__':
    main()
