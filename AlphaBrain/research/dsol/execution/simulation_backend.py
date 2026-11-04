"""Persistent, bounded CPU fake for exercising the real queue start boundary.

No subprocess, SSH, GPU package, simulator, model or result-directory access.
All outputs are explicitly synthetic records under a separate backend root.
"""
from copy import deepcopy
import json
from pathlib import Path

from .manifest import digest
from .queue import MissingResult, _durable_write


class SimulationBackend:
    def __init__(self, root, limits, *, faults=None):
        self.root, self.limits = Path(root), deepcopy(limits)
        self.faults = dict(faults or {})
        self.offline = set()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'simulation.json'
        if self.path.exists():
            self.read()
        else:
            if list(self.root.iterdir()):
                raise ValueError('Simulation requires an empty or existing simulation directory')
            _durable_write(self.path, dict(schema='CPU_FAKE_BACKEND_ONLY', attempts={}))

    def read(self):
        state = json.loads(self.path.read_text())
        if state.get('schema') != 'CPU_FAKE_BACKEND_ONLY':
            raise ValueError('Not a simulation backend store')
        return state

    def probe(self, peer):
        return peer not in self.offline

    def admit(self, slot):
        return self.probe(slot['host'])

    def start(self, task, slot, attempt):
        state = self.read()
        key = task['key']
        if key in state['attempts']:
            raise RuntimeError('Simulation detects duplicate backend.start')
        mode = self.faults.get(key, 'complete')
        handle = dict(key=key, attempt=attempt)
        row = dict(handle=handle, task_sha256=digest(task), slot=slot, mode=mode,
                   alive=mode == 'unknown', exitcode=1 if mode == 'partial' else 0)
        if mode == 'complete':
            row['receipt'] = dict(simulation_only=True, task_sha256=digest(task),
                                  budget_consumed=deepcopy(self.limits[key]), scientific_success=False)
        state['attempts'][key] = row
        _durable_write(self.path, state)
        if mode == 'unknown':
            raise ConnectionError('Injected loss after backend launch was recorded')
        return handle

    def locate(self, task, slot, attempt):
        row = self.read()['attempts'].get(task['key'])
        if row is None:
            return None
        if row['slot'] != slot or row['handle']['attempt'] != attempt:
            raise ValueError('Simulation attempt identity changed')
        return deepcopy(row['handle'])

    def alive(self, handle):
        row = self.read()['attempts'][handle['key']]
        if row['handle'] != handle:
            raise ValueError('Simulation process identity changed')
        return row['alive']

    def poll(self, handle):
        return None if self.alive(handle) else self.read()['attempts'][handle['key']]['exitcode']

    def cancel(self, handle):
        self.alive(handle)
        state = self.read()
        row = state['attempts'][handle['key']]
        row['alive'] = False
        if 'receipt' not in row:
            row['exitcode'] = 143
        _durable_write(self.path, state)

    def validate(self, task):
        row = self.read()['attempts'].get(task['key'])
        if row is None or 'receipt' not in row:
            raise MissingResult('No complete synthetic result')
        if row['task_sha256'] != digest(task) or row['alive'] or row['exitcode'] != 0:
            raise ValueError('Synthetic result not complete or identity changed')
        return deepcopy(row['receipt'])
