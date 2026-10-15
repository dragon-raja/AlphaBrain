"""Run a frozen bounded LIBERO panel; never retry failed scientific outcomes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def run(root):
    plan = read(root/'plan.json')
    native = plan.get('format') == 'libero_native_reference_panel_v1'
    expected_gpus = list(range(8))*2 if native else [1, 3, 5, 7]
    if plan['count'] != 640 or plan['gpus'] != expected_gpus or plan['workers'] != len(expected_gpus):
        raise ValueError('Budget or GPU assignment changed')
    if sha(plan['runner']) != plan['runner_sha256']:
        raise ValueError('Frozen runner changed')
    gate = read(root/'smoke-audit.json')
    if gate['status'] != 'PASS' or gate['plan_sha256'] != sha(root/'plan.json'):
        raise ValueError('Whole-rollout smoke audit missing')
    # Existing users of selected GPUs may only be the verified keepalive program.
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    selected = {line.split(',')[1].strip() for line in devices.splitlines() if int(line.split(',')[0]) in plan['gpus']}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    for line in apps.splitlines():
        uuid, pid = [x.strip() for x in line.split(',')]
        if uuid in selected:
            cmd = Path('/proc')/pid/'cmdline'
            if 'gpu_compute_keepalive.py' not in cmd.read_bytes().decode(errors='replace'):
                raise ValueError('Selected GPU has an unrelated process: '+pid)
    post = [Path(__file__), Path(__file__).parents[2]/'AlphaBrain/research/dsol/analysis/libero_baseline_panel.py',
            Path(__file__).parents[2]/'AlphaBrain/research/dsol/plotting/libero_baseline_panel.py']
    identity = {str(p): sha(p) for p in post}
    write(root/'postprocessing-source-identity.json', identity)
    children = []
    logs = []
    started = time.time()
    try:
        for worker in range(plan['workers']):
            path = root/'workers'/str(worker)/'driver.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open('a')
            logs.append(log)
            env = os.environ.copy()
            parent = Path(plan['parent'])
            env.update(PYTHONPATH=f"{parent}/runtime/policy/lib/python3.12/site-packages:{parent}/repo",
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', ALPHABRAIN_DISABLE_AUTO_DOWNLOAD='1',
                       PRETRAINED_MODELS_DIR='/share/longjunyu/alphabrain/pretrained_models')
            children.append(subprocess.Popen([sys.executable, plan['runner'], 'worker', '--root', str(root),
                             '--worker', str(worker)], env=env, stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [p.poll() for p in children]
            count = len(list((root/'episodes').glob('*.json')))
            write(root/'status.json', dict(status='RUNNING', completed=count, total=640,
                         child_pids=[p.pid for p in children], started=started, heartbeat=time.time()))
            if any(c not in (None, 0) for c in codes):
                raise RuntimeError('Worker failed; inspect worker logs')
            if all(c == 0 for c in codes):
                break
            if time.time()-started > 6*3600:
                raise TimeoutError('Six-hour execution bound reached')
            time.sleep(15)
        if count != 640:
            raise ValueError('Incomplete bounded matrix')
        if any(sha(p) != digest for p, digest in identity.items()):
            raise ValueError('Postprocessing source changed')
        from AlphaBrain.research.dsol.analysis.libero_baseline_panel import summarize
        summarize(root)
        write(root/'status.json', dict(status='COMPLETE', completed=640, total=640, finished=time.time()))
    except BaseException as exc:
        write(root/'status.json', dict(status='ERROR', error=repr(exc), heartbeat=time.time()))
        # Do not signal other jobs; workers already running have their own finite lists.
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required')
    run(a.root)
