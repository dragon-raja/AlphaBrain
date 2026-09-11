#!/usr/bin/env python3
"""Bounded full-trajectory validation of an explicitly separate render protocol."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[4]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import scripts.dsol_paper1.operations.controllers.run_full64_view_landscape_v2 as control
from scripts.dsol_paper1.operations.controllers.run_early_landscape_speed_gate_v1 import process_identity

ROOT = control.ROOT/'repeatability-root-cause-v1'
WIN = ROOT/'render-fix-window-v1'

def main():
    release = control.preflight()
    control.require(os.getpgrp() == os.getpid(), 'Must launch in an owned process group')
    control.write_new_json(WIN/'process.json', process_identity(os.getpid()))
    control.write_new_json(WIN/'noaa-validation-entry.json', {
        'utc': control.stamp(), 'protocol': 'diagnostic-noaa-v1', 'formal_samples_added': 0,
        'tasks': ['goal_wine_rack', 'libero10_mug_microwave'], 'render_gpus': [0, 1],
        'purpose': 'Full closed-loop bitwise comparison; not old-protocol equivalence'})
    servers, children, streams, stopped = [], [], [], []
    def interrupt(signum, frame): raise InterruptedError('Bounded validation interrupted')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    try:
        control.resources_available(29600, 2)
        for gpu in (0, 1):
            name = f'gpu-keepalive-{gpu}'
            if subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True).returncode == 0:
                subprocess.run(['tmux', 'kill-session', '-t', name], check=True); stopped.append(gpu)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                PRETRAINED_MODELS_DIR='/share/longjunyu/alphabrain/pretrained_models', ALPHABRAIN_DISABLE_AUTO_DOWNLOAD='1',
                PYTHONPATH=str(control.REPO)+':/projects/openpi/src:/projects/openpi/packages/openpi-client/src')
            stream = (WIN/f'noaa-policy-{gpu}.log').open('a'); streams.append(stream)
            servers.append(subprocess.Popen(['/alphabrain/.venv/bin/python', str(control.REPO/'scripts/cabi_vla/serve_alphabrain_pi05_websocket.py'),
                '--checkpoint', release['old_release_content']['canonical_checkpoint']['path'], '--port', str(29600+gpu),
                '--device', 'cuda:0', '--cpu-threads', '2'], env=env, stdout=stream, stderr=subprocess.STDOUT))
        deadline = time.time()+180
        while True:
            control.require(all(p.poll() is None for p in servers), 'Policy startup failed')
            ready=[]
            for port in (29600,29601):
                with socket.socket() as sock: ready.append(sock.connect_ex(('127.0.0.1',port))==0)
            if all(ready): break
            control.require(time.time()<deadline, 'Startup timeout'); time.sleep(2)
        for task in ('goal_wine_rack','libero10_mug_microwave'):
            for gpu in (0,1):
                output=WIN/f'{task}-noaa-gpu{gpu}'
                env=os.environ.copy()
                env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    PYTHONPATH=str(control.REPO)+':/projects/openpi/packages/openpi-client/src:'+str(control.REPO/'scripts/cabi_vla'))
                stream=(WIN/(output.name+'.log')).open('a'); streams.append(stream)
                children.append(subprocess.Popen(['/workspace/envs/fresh-libero/bin/python',
                    str(control.REPO/'scripts/dsol_paper1/diagnostics/trace_view_noaa_v1.py'), '--spec', str(ROOT/(task+'-spec.json')),
                    '--output', str(output), '--bank', release['old_release_content']['noise_bank']['path'],
                    '--render-gpu', str(gpu)], env=env, stdout=stream, stderr=subprocess.STDOUT))
        for child in children: control.require(child.wait(timeout=360)==0, 'Trace failed')
        control.write_new_json(WIN/'noaa-validation-complete.json', {'utc':control.stamp(),'status':'ALL_FOUR_TRACES_COMPLETE'})
    finally:
        for child in children+servers:
            if child.poll() is None: child.terminate()
        for child in children+servers:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for stream in streams: stream.close()
        for gpu in stopped:
            if subprocess.run(['tmux','has-session','-t',f'gpu-keepalive-{gpu}'],capture_output=True).returncode:
                subprocess.run(['bash','/workspace/ai2r/gpu_compute_keepalive/start.sh','1','8192',f'gpu-keepalive-{gpu}',str(gpu)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__': main()
