#!/usr/bin/env python3
"""Bounded isolated two-policy / three-state root-cause probe, not formal eval."""
from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[4]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import scripts.dsol_paper1.operations.controllers.run_full64_view_landscape_v2 as control
from scripts.dsol_paper1.protocols.build_matched_view_landscape_protocol_v1 import write_new_json, source_identity
from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count

ROOT = control.ROOT / 'repeatability-root-cause-v1'


def main():
    release = control.preflight()
    ROOT.mkdir(exist_ok=True)
    control.require(not (ROOT/'probe-entry.json').exists(), 'No implicit probe rerun')
    protocol = control.read_json(Path(release['benchmark_protocol']['path']))
    diagnostics = control.read_json(control.ROOT/'early-speed-gate-v1/equivalence-diagnostics-v1.json')
    details = diagnostics['comparisons']['copies3-workers48']['details']
    targets = []
    for task in ('goal_wine_rack', 'libero10_mug_microwave'):
        candidates = [r for r in details if r['task_id'] == task and r['first_different_action_call'] is not None]
        targets.append(min(candidates, key=lambda r: (r['first_different_action_call'], r['pair_key'], r['candidate'], r['repeat'])))
    targets.append(next(r for r in details if r['task_id'] == 'goal_cream_cheese_bowl' and r['candidate'] == 'canonical' and r['repeat'] == 0))
    specs = [protocol_spec_at(protocol, i) for i in range(protocol_spec_count(protocol))]
    selected = []
    for target in targets:
        spec = next(s for s in specs if (s['pair_key'], s['selected_candidate_id'], s['policy_repeat_id']) ==
                    (target['pair_key'], target['candidate'], target['repeat']))
        path = ROOT / (target['task_id']+'-spec.json')
        write_new_json(path, spec)
        selected.append((target['task_id'], path))
    write_new_json(ROOT/'probe-entry.json', {'status': 'FROZEN_ENGINEERING_DIAGNOSTIC',
        'created_at_utc': control.stamp(), 'sources': [source_identity(path) for _, path in selected],
        'source_selection': 'Two earliest observed divergent requests by task, plus a fixed nondivergent task. Debugging only, not a research effect estimate.',
        'trace_script': source_identity(control.REPO/'scripts/dsol_paper1/diagnostics/trace_view_repeatability_v1.py'),
        'driver': source_identity(Path(__file__)), 'model_changed': False, 'formal_samples_added': 0})
    control.resources_available(29600, 2)
    servers, streams, stopped = [], [], []
    child = None

    def interrupt(signum, frame):
        raise InterruptedError('Diagnostic interrupted')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for gpu in (0, 1):
            name = f'gpu-keepalive-{gpu}'
            if subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True).returncode == 0:
                subprocess.run(['tmux', 'kill-session', '-t', name], check=True)
                stopped.append(gpu)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                PRETRAINED_MODELS_DIR='/share/longjunyu/alphabrain/pretrained_models', ALPHABRAIN_DISABLE_AUTO_DOWNLOAD='1',
                PYTHONPATH=str(control.REPO)+':/projects/openpi/src:/projects/openpi/packages/openpi-client/src')
            stream = (ROOT/f'policy-{gpu}.log').open('a'); streams.append(stream)
            servers.append(subprocess.Popen(['/alphabrain/.venv/bin/python',
                str(control.REPO/'scripts/cabi_vla/serve_alphabrain_pi05_websocket.py'),
                '--checkpoint', release['old_release_content']['canonical_checkpoint']['path'],
                '--port', str(29600+gpu), '--device', 'cuda:0', '--cpu-threads', '2'],
                env=env, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.time()+180
        while True:
            control.require(all(p.poll() is None for p in servers), 'Diagnostic policy failed to start')
            ready = []
            for port in (29600, 29601):
                with socket.socket() as sock:
                    ready.append(sock.connect_ex(('127.0.0.1', port)) == 0)
            if all(ready):
                break
            control.require(time.time() < deadline, 'Policy startup timed out')
            time.sleep(2)
        for task, spec_path in selected:
            for label, gpu, replay in [('live-a', 0, None), ('live-b', 1, None),
                                       ('replay-a', 0, ROOT/(task+'-live-a')),
                                       ('replay-b', 1, ROOT/(task+'-live-a'))]:
                output = ROOT/(task+'-'+label)
                control.atomic_json(ROOT/'probe-status.json', {'status': 'RUNNING_TRACE', 'trace': output.name, 'utc': control.stamp()})
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                    TOKENIZERS_PARALLELISM='false', IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg',
                    PYTHONPATH=str(control.REPO)+':/projects/openpi/packages/openpi-client/src:'+str(control.REPO/'scripts/cabi_vla'))
                command = ['/workspace/envs/fresh-libero/bin/python',
                    str(control.REPO/'scripts/dsol_paper1/diagnostics/trace_view_repeatability_v1.py'), '--spec', str(spec_path),
                    '--output', str(output), '--bank', release['old_release_content']['noise_bank']['path'],
                    '--render-gpu', str(gpu)]
                if replay:
                    command.extend(['--replay', str(replay)])
                with (ROOT/(output.name+'.log')).open('a') as stream:
                    child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    code = child.wait(timeout=360)
                    control.require(code == 0, 'Diagnostic trace failed: '+str(output))
                child = None
        control.atomic_json(ROOT/'probe-status.json', {'status': 'PASS_ALL_TRACES_COMPLETE', 'utc': control.stamp()})
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for server in servers:
            if server.poll() is None: server.terminate()
        for server in servers:
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired: server.kill(); server.wait()
        for stream in streams: stream.close()
        for gpu in stopped:
            if subprocess.run(['tmux', 'has-session', '-t', f'gpu-keepalive-{gpu}'], capture_output=True).returncode:
                subprocess.run(['bash', '/workspace/ai2r/gpu_compute_keepalive/start.sh', '1', '8192', f'gpu-keepalive-{gpu}', str(gpu)], check=True)


if __name__ == '__main__':
    main()
