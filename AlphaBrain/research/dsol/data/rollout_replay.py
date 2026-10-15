"""Opt-in, lossless engineering replay records; no simulator/model imports."""
import hashlib
from pathlib import Path

import numpy as np

from .artifacts import read, sha, write


def array_identity(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


class ActionReplay:
    """Explicit diagnostic action source; never represents live policy success."""
    def __init__(self, record_file):
        self.record_file = Path(record_file)
        self.record = read(self.record_file)
        payload = self.record_file.parent / 'arrays.npz'
        if sha(payload) != self.record['arrays_sha256']:
            raise ValueError('Replay payload has changed')
        with np.load(payload, allow_pickle=False) as arrays:
            self.actions = [arrays[f'plan_{i:04d}/actions'].copy() for i in range(len(self.record['plans']))]

    def plan(self, index, step, noise):
        if index >= len(self.record['plans']):
            raise ValueError('Action replay exhausted')
        row = self.record['plans'][index]
        if row['step'] != step or row['arrays']['noise'] != array_identity(noise):
            raise ValueError('Action replay timing or noise identity mismatch')
        if row['arrays']['actions'] != array_identity(self.actions[index]):
            raise ValueError('Action replay identity mismatch')
        return self.actions[index].copy()


class ReplayRecorder:
    def __init__(self):
        self.plans, self.steps, self.arrays = [], [], {}

    def record_plan(self, step, request, noise, actions, physical_sha256):
        index = len(self.plans)
        arrays = {key: value for key, value in request.items() if isinstance(value, np.ndarray)}
        arrays.update(noise=noise, actions=actions)
        identities = {}
        for key, value in arrays.items():
            name = f'plan_{index:04d}/' + key
            self.arrays[name] = np.array(value, copy=True)
            identities[key] = array_identity(value)
        self.plans.append(dict(index=index, step=step, prompt=request['prompt'],
                               physical_sha256=physical_sha256, arrays=identities))

    def record_step(self, step, action, physical_sha256, success):
        self.steps.append(dict(step=step, action=array_identity(action),
                               physical_sha256=physical_sha256, success=bool(success)))

    def save(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        payload = root / 'arrays.npz'
        with payload.open('xb') as stream:
            np.savez_compressed(stream, **self.arrays)
        result = dict(scope='diagnostic only; recording overhead excluded from capacity estimates',
                      plans=self.plans, steps=self.steps, arrays_sha256=sha(payload))
        write(root / 'replay.json', result)
        return str(root / 'replay.json')
