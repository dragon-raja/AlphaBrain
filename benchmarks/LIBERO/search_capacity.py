"""Real-render, synchronized batch-one capacity worker; launched by execution queue.

Two calls per worker: a checked warmup and a synchronized timed query. No
manipulation actions or scientific success labels are produced by this test.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from .search_features import load_observations
from .search_live import simulator, verify_example, check_simulator_dependencies
from .search_runtime import PolicyRuntime, exact_prediction


def wait_barrier(path, timeout=600):
    start = time.monotonic()
    while not Path(path).is_file():
        if time.monotonic()-start > timeout:
            raise TimeoutError('Capacity barrier deadline exceeded')
        time.sleep(.1)


def run(config_path, output, *, execute=False):
    if not execute:
        raise ValueError('Explicit --execute required')
    config = read(config_path); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = None
    try:
        for path, digest in config['input_hashes'].items():
            if sha(path) != digest:
                raise ValueError('Bound capacity source changed: '+path)
        check_simulator_dependencies(config)
        runtime = PolicyRuntime(config, output, 2)
        initial, candidate, member = 0, 65, 0
        record = read(config['inputs'])['records'][initial]
        cached = load_observations(record)
        with np.load(config['anchors'], allow_pickle=False) as z:
            noise = z['first_score_noise'][initial,member].copy()
        with np.load(Path(config['native'])/'initials'/'00.npz', allow_pickle=False) as z:
            reference = {k:z[k][candidate,member].copy() for k in ('trace','actions')}
        with np.load(Path(config['native'])/'siglip-scores.npz', allow_pickle=False) as z:
            reference.update(score=z['scores'][initial,candidate,member],
                external=z['external'][initial,candidate], wrist=z['wrist'][initial,candidate])
        with simulator(config_path,config,initial,output) as request:
            example = request(dict(kind='render', candidate=candidate))['example']
            verify_example(example,cached,candidate)
            value = runtime.infer(example,noise,dict(stage='warmup'))
            exact_prediction(value,reference)
            write(output/'ready.json',dict(status='READY',warmup_seconds=value['seconds']))
            wait_barrier(config['capacity_barrier'])
            started = time.time()
            example = request(dict(kind='render',candidate=candidate))['example']
            verify_example(example,cached,candidate)
            value = runtime.infer(example,noise,dict(stage='timed'))
            finished = time.time()
            exact_prediction(value,reference)
            write(output/'timing.json',dict(started=started,finished=finished,model_seconds=value['seconds']))
        write(output/'status.json',dict(status='COMPLETE',calls=runtime.calls,config_sha256=sha(config_path),
            exact=True,timing_sha256=sha(output/'timing.json'),peak_allocated_bytes=runtime.torch.cuda.max_memory_allocated()))
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),calls=0 if runtime is None else runtime.calls))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true'); args=parser.parse_args()
    run(args.config,args.output,execute=args.execute)
