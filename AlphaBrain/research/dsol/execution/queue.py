"""Single-coordinator persistent queue with explicit recovery and backend gates.

Only the coordinator writes the ledger. Workers may reside on any admitted
host. Backend adapters own subprocess/SSH details and scientific validation.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Protocol

from .manifest import digest, validate, verify_artifacts


class MissingResult(Exception):
    """No complete receipt yet. Invalid complete results must use other errors."""


class Backend(Protocol):
    def admit(self, slot: dict) -> bool: ...
    def start(self, task: dict, slot: dict, attempt: int) -> dict: ...
    def locate(self, task: dict, slot: dict, attempt: int) -> dict | None: ...
    def poll(self, handle: dict) -> int | None: ...
    def alive(self, handle: dict) -> bool: ...
    def cancel(self, handle: dict) -> None: ...
    def validate(self, task: dict) -> dict: ...


def _durable_write(path, value):
    """Persist launch intent before launch; atomic visibility alone is not enough."""
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def exclusive(root):
    """Nonblocking coordinator lock; use a local, flock-capable control volume."""
    with (Path(root) / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load(root):
    root = Path(root)
    plan = validate(json.loads((root / 'manifest.json').read_text()))
    state = json.loads((root / 'state.json').read_text())
    if state['manifest_sha256'] != digest(plan) or set(state['tasks']) != {t['key'] for t in plan['tasks']}:
        raise ValueError('Frozen manifest or task coverage changed')
    allowed = {'PENDING', 'LAUNCHING', 'RUNNING', 'COMPLETE'}
    for row in state['tasks'].values():
        if row.get('status') not in allowed or type(row.get('attempt')) is not int or row['attempt'] < 0:
            raise ValueError('Invalid persisted task state')
        if row['status'] in ('LAUNCHING', 'RUNNING') and row.get('slot') not in plan['slots']:
            raise ValueError('Worker slot is outside the frozen manifest')
        if row['status'] == 'RUNNING' and not isinstance(row.get('handle'), dict):
            raise ValueError('Running task lost its durable process identity')
        if row['status'] == 'COMPLETE' and not isinstance(row.get('receipt'), dict):
            raise ValueError('Completed task lost its validated receipt')
    return plan, state


def _save(root, state, event, **details):
    state['events'].append(dict(event=event, unix=time.time(), **details))
    state['updated_unix'] = time.time()
    _durable_write(Path(root) / 'state.json', state)


def prepare(plan, root):
    """Freeze a new queue, without loading an adapter or starting any process."""
    plan = deepcopy(validate(plan))
    verify_artifacts(plan)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    _durable_write(root / 'manifest.json', plan)
    state = dict(status='PREPARED', manifest_sha256=digest(plan),
                 tasks={t['key']: dict(status='PENDING', attempt=0) for t in plan['tasks']}, events=[])
    _save(root, state, 'prepared')
    return state


def _receipt(backend, task):
    receipt = backend.validate(deepcopy(task))
    if not isinstance(receipt, dict) or not receipt:
        raise ValueError('Adapter must return a nonempty validated receipt')
    digest(receipt)
    return deepcopy(receipt)


def summary(root):
    plan, state = load(root)
    counts = {name: sum(t['status'] == name for t in state['tasks'].values())
              for name in ('PENDING', 'LAUNCHING', 'RUNNING', 'COMPLETE')}
    return dict(status=state['status'], experiment_id=plan['experiment_id'],
                manifest_sha256=state['manifest_sha256'], blocks=len(plan['tasks']),
                trials=sum(len(t['addresses']) for t in plan['tasks']), counts=counts,
                updated_unix=state['updated_unix'])


def recover(root, backend):
    """Explicit recovery: prove no old workers live; preserve validated results.

    Missing outputs return to PENDING. Corrupt outputs stop recovery. No
    success-based filtering and no automatic retry of a failed experiment.
    """
    with exclusive(root):
        if (Path(root)/'retired.json').exists(): raise ValueError('Retired queue must not restart')
        plan, old = load(root)
        if old['status'] not in ('RUNNING', 'FAILED', 'INTERRUPTED'):
            raise ValueError('Only interrupted/failed queues can be recovered')
        verify_artifacts(plan)
        state = deepcopy(old)
        retained, pending = [], []
        for task in plan['tasks']:
            row = state['tasks'][task['key']]
            handle = row.get('handle')
            if row['status'] == 'LAUNCHING':
                # Adapter MUST prove absence or return the durable launch receipt.
                handle = backend.locate(deepcopy(task), deepcopy(row['slot']), row['attempt'])
            if handle is not None and backend.alive(handle):
                raise RuntimeError('Old owned worker still alive; recovery refused')
            try:
                receipt = _receipt(backend, task)
            except MissingResult:
                if row['status'] == 'COMPLETE':
                    raise ValueError('Previously completed result disappeared')
                row.update(status='PENDING')
                row.pop('handle', None)
                pending.append(task['key'])
            else:
                if row['status'] == 'COMPLETE' and receipt != row['receipt']:
                    raise ValueError('Previously completed result changed')
                row.update(status='COMPLETE', receipt=receipt)
                retained.append(task['key'])
        state['status'] = 'PREPARED'
        state.pop('error', None)
        _save(root, state, 'explicit_recovery', retained=retained, pending=pending)
        return state


def run(root, backend, *, poll_seconds=1.0, timeout_seconds=3600.0,
        clock=time.monotonic, sleep=time.sleep):
    """Outcome-independent FIFO drain across configured, admitted idle slots."""
    if not math.isfinite(poll_seconds) or not math.isfinite(timeout_seconds) or poll_seconds < 0 or timeout_seconds <= 0:
        raise ValueError('Invalid polling interval or timeout')
    with exclusive(root):
        if (Path(root)/'retired.json').exists(): raise ValueError('Retired queue must not restart')
        plan, state = load(root)
        if state['status'] != 'PREPARED':
            raise ValueError('Queue requires preparation or explicit recovery')
        verify_artifacts(plan)
        # Reuse is permitted only after the adapter rechecks the exact receipt.
        for task in plan['tasks']:
            row = state['tasks'][task['key']]
            if row['status'] == 'COMPLETE' and _receipt(backend, task) != row['receipt']:
                raise ValueError('Retained result identity changed')
        state['status'] = 'RUNNING'
        _save(root, state, 'started')
        started = clock()
        try:
            while any(t['status'] != 'COMPLETE' for t in state['tasks'].values()):
                if clock() - started > timeout_seconds:
                    raise TimeoutError('Queue deadline exceeded')
                for task in plan['tasks']:
                    row = state['tasks'][task['key']]
                    if row['status'] != 'RUNNING':
                        continue
                    code = backend.poll(row['handle'])
                    if code is None:
                        continue
                    if code != 0:
                        raise RuntimeError(f'Worker failed: {task["key"]}, exit {code}')
                    if backend.alive(row['handle']):
                        raise RuntimeError('Worker exit reported with live owned children')
                    row.update(status='COMPLETE', receipt=_receipt(backend, task))
                    _save(root, state, 'completed', key=task['key'])
                occupied = {(t['slot']['host'], t['slot']['id']) for t in state['tasks'].values()
                            if t['status'] in ('LAUNCHING', 'RUNNING')}
                from .resources import dispatch_guard
                for slot in plan['slots']:
                    if (slot['host'], slot['id']) in occupied:
                        continue
                    with dispatch_guard(root,plan,slot) as allowed:
                        if not allowed or not backend.admit(deepcopy(slot)):
                            continue
                        task = next((t for t in plan['tasks'] if state['tasks'][t['key']]['status'] == 'PENDING'), None)
                        if task is None:
                            break
                        row = state['tasks'][task['key']]
                        row.update(status='LAUNCHING', slot=slot, attempt=row['attempt'] + 1)
                        _save(root, state, 'launch_intent', key=task['key'], slot=slot, attempt=row['attempt'])
                        handle = backend.start(deepcopy(task), deepcopy(slot), row['attempt'])
                        if not isinstance(handle, dict) or not handle:
                            raise ValueError('Durable JSON launch handle required')
                        digest(handle)
                        row.update(status='RUNNING', handle=deepcopy(handle))
                        _save(root, state, 'launched', key=task['key'])
                sleep(poll_seconds)
            verify_artifacts(plan)
            if digest(load(root)[0]) != state['manifest_sha256']:
                raise ValueError('Manifest changed during execution')
            state['status'] = 'COMPLETE'
            _save(root, state, 'queue_complete')
        except BaseException as exc:
            errors = []
            for task in plan['tasks']:
                row = state['tasks'][task['key']]
                if row['status'] not in ('LAUNCHING', 'RUNNING'):
                    continue
                try:
                    handle = row.get('handle')
                    if handle is None:
                        handle = backend.locate(deepcopy(task), deepcopy(row['slot']), row['attempt'])
                    if handle is not None:
                        backend.cancel(handle)
                        if backend.alive(handle):
                            raise RuntimeError('Owned worker did not stop')
                except BaseException as cleanup:
                    errors.append(dict(key=task['key'], error=repr(cleanup)))
            state.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                         error=repr(exc), cleanup_errors=errors)
            _save(root, state, 'stopped')
            raise
        return state
