"""Run an explicitly bounded eight-task initialization attribution probe.

Uses the original simulator, initial states, prompts, cameras, horizons and
7D noise coordinates. Diagnostic policy changes are recorded separately.
Never promotes these eight outcomes to a benchmark success-rate estimate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from baseline_panel import frozen_core, read, sha, write


def episode_main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(sys.argv[2:])
    core = frozen_core(read(args.root/'plan.json'))
    core.ROOT = args.root
    args.model, args.host = 'baseline', 'fresh'
    core.episode(args, read(args.root/'release.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--alpha-checkpoint', type=Path, required=True)
    p.add_argument('--native-checkpoint', type=Path, required=True)
    p.add_argument('--variant', choices=['native', 'native7', 'alpha'], required=True)
    p.add_argument('--normalization', choices=['quantile', 'meanstd'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--native-python', type=Path)
    p.add_argument('--native-overlay', type=Path)
    p.add_argument('--native-deps', type=Path)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        p.error('Explicit --execute required')
    if args.root.exists():
        raise FileExistsError(args.root)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError('Port is already owned')
    parent_release = read(args.parent/'release.json')
    indices = [state*97*32 for state in range(0, 32, 4)]
    source = Path(__file__).with_name('initialization_diagnostic.py')
    plan = dict(parent=str(args.parent), parent_release_sha256=sha(args.parent/'release.json'),
                variant=args.variant, normalization=args.normalization, indices=indices,
                scope='Eight-task engineering attribution; one first initial, canonical view, noise0 each',
                policy_source=str(source), policy_source_sha256=sha(source),
                controller_sha256=sha(Path(__file__)),
                internal_action_dim=32 if args.variant == 'native' else 7,
                noise_extension='For native32 only: domain-separated SHA256 seed for PCG64 extra25; preserve frozen first7')
    write(args.root/'plan.json', plan)
    checkpoint = args.native_checkpoint if args.variant.startswith('native') else args.alpha_checkpoint
    release = dict(parent_release)
    release['models'] = {'baseline': dict(path=str(checkpoint), weights_sha256=sha(checkpoint/'model.safetensors'))}
    release['initialization_diagnostic'] = plan
    write(args.root/'release.json', release)
    core = frozen_core(plan)
    sim_env = core.sim_env()
    sim_python = core.python(True)
    env = core.env(gpu=args.gpu)
    repo = Path(__file__).resolve().parents[2]
    env['JAX_PLATFORMS'] = 'cpu'
    env['OPENPI_DATA_HOME'] = str(args.cache)
    env['PYTHONPATH'] = str(repo)+':'+env['PYTHONPATH']
    policy_python = core.python()
    if args.variant.startswith('native'):
        if not args.native_python or not args.native_overlay:
            raise ValueError('Native diagnostic requires an isolated validated native runtime')
        policy_python = args.native_python
        env['PYTHONPATH'] = ':'.join(map(str, [args.native_overlay, repo,
                args.parent/'runtime/policy/lib/python3.12/site-packages',
                args.parent/'runtime/openpi-src', args.parent/'runtime/openpi-client-src', args.native_deps]))
    command = [str(policy_python), str(source), '--parent', str(args.parent),
               '--alpha-checkpoint', str(args.alpha_checkpoint), '--native-checkpoint', str(args.native_checkpoint),
               '--variant', args.variant, '--normalization', args.normalization, '--port', str(args.port)]
    write(args.root/'runtime.json', dict(command=command, environment={k:env[k] for k in
          ['PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'OPENPI_DATA_HOME']}))
    started = time.time()
    server_log = (args.root/'server.log').open('x')
    server = subprocess.Popen(command, env=env, stdout=server_log, stderr=subprocess.STDOUT)
    write(args.root/'status.json', dict(status='LOADING', started=started, pid=server.pid))
    rows = []
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError('Policy failed; see server.log')
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', args.port)) == 0:
                    break
            if time.time()-started > 480:
                raise TimeoutError('Policy startup timeout')
            time.sleep(1)
        for index in indices:
            output = args.root/'episodes'/f'{index:06d}.json'
            command = [str(sim_python), str(Path(__file__)), 'episode',
                       '--root', str(args.root), '--index', str(index), '--output', str(output),
                       '--port', str(args.port), '--gpu', str(args.gpu)]
            with (args.root/'episodes.log').open('a') as log:
                subprocess.run(command, env=sim_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
            row = read(output)
            if row['aa0']['release_sha256'] != sha(args.root/'release.json'):
                raise ValueError('Episode protocol identity changed')
            rows.append(dict(index=index, pair_key=row['pair_key'], success=row['success'],
                             completion_steps=row['completion_steps'], output=str(output), sha256=sha(output)))
            write(args.root/'status.json', dict(status='RUNNING', completed=len(rows), count=8, rows=rows, started=started))
            print(json.dumps(rows[-1]), flush=True)
        write(args.root/'status.json', dict(status='COMPLETE', completed=8, count=8, rows=rows, started=started, finished=time.time()))
    except BaseException as exc:
        write(args.root/'status.json', dict(status='ERROR', error=repr(exc), completed=len(rows), rows=rows, started=started))
        raise
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        server_log.close()


if __name__ == '__main__':
    episode_main() if len(sys.argv) > 1 and sys.argv[1] == 'episode' else main()
