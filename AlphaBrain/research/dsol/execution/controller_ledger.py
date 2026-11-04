"""Durable single-owner custody and bounded CPU dispatch rehearsals.

This owns custody and accounting; controlled_queue wraps the standard queue's
launch boundary. New production custody requires LIVE_MODEL_V1 and a separate permit. Frozen
executors do not acquire this ownership automatically.
Live authority roots must be local; immutable checkpoints may use shared storage.
"""
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from .manifest import digest
from .queue import _durable_write, exclusive


def validate_plan(plan):
    """Bind disjoint scientific addresses, stages and integer call ceilings."""
    if not plan['campaign_sha256'] or not plan['stages'] or not plan['peers']:
        raise ValueError('Missing campaign, stages or peers')
    if len(set(plan['stages'])) != len(plan['stages']) or len(set(plan['peers'])) != len(plan['peers']):
        raise ValueError('Duplicate stages or peers')
    addresses, keys = set(), set()
    totals = {k: 0 for k in plan['limits']}
    for task in plan['tasks']:
        if task['key'] in keys or task['stage'] not in plan['stages']:
            raise ValueError('Duplicate task or unknown stage')
        keys.add(task['key'])
        if not task['addresses'] or set(task['limits']) != set(totals):
            raise ValueError('Missing addresses or budget dimensions')
        for address in task['addresses']:
            identity = digest(address)
            if identity in addresses:
                raise ValueError('Overlapping scientific addresses')
            addresses.add(identity)
        for name, value in task['limits'].items():
            if type(value) is not int or value < 0:
                raise ValueError('Invalid budget')
            totals[name] += value
    if not keys or totals != plan['limits']:
        raise ValueError('Budget coverage changed')
    if any(type(v) is not int or v < 0 for v in plan['limits'].values()):
        raise ValueError('Invalid total budget')
    return plan


def prepare(root, plan, *, simulation_only=False, execution_contract=None):
    if type(simulation_only) is not bool:
        raise ValueError('Explicit boolean simulation scope required')
    if execution_contract not in (None, 'LIVE_MODEL_V1') or (simulation_only and execution_contract):
        raise ValueError('Invalid explicit execution contract')
    plan = deepcopy(validate_plan(plan))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    _durable_write(root / 'plan.json', plan)
    _durable_write(root / 'ledger.json', dict(
        schema='controller_custody_v1', plan_sha256=digest(plan),
        dispatch_enabled=False, simulation_only=simulation_only, status='PAUSED', owner=None, epoch=0,
        stage=None, tasks={}, events=[], **({'execution_contract': execution_contract} if execution_contract else {})))


class ControllerLedger:
    def __init__(self, root):
        self.root = Path(root)
        self.failed = False
        self.session = None

    @contextmanager
    def launcher_lock(self):
        """Exclude other integrated controllers/recovery for the whole queue run.

        Keep this separate from the short ledger transaction lock. Never unlink
        either lock, and never take over on heartbeat expiry.
        """
        with (self.root / 'launcher.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def read(self):
        plan = validate_plan(json.loads((self.root / 'plan.json').read_text()))
        state = json.loads((self.root / 'ledger.json').read_text())
        live = state.get('execution_contract') == 'LIVE_MODEL_V1' and state.get('simulation_only') is False
        if state['plan_sha256'] != digest(plan) or (state['dispatch_enabled'] is not False and not (
                live and state['dispatch_enabled'] is True and state['status'] == 'MODEL_ACTIVE')):
            raise ValueError('Changed plan or unsupported real dispatch')
        tasks = {t['key']: t for t in plan['tasks']}
        if not set(state['tasks']).issubset(tasks):
            raise ValueError('Unknown ledger task')
        for key, row in state['tasks'].items():
            if row['reserved'] != tasks[key]['limits'] or row['status'] not in ('RESERVED', 'COMPLETE'):
                raise ValueError('Changed reservation')
            if row['status'] == 'COMPLETE':
                self._counts(row['consumed'], row['reserved'])
            if 'launch_status' in row:
                if row['launch_status'] not in ('RESERVED', 'RUNNING', 'UNKNOWN', 'COMPLETE'):
                    raise ValueError('Invalid launch state')
                if (row['status'] == 'COMPLETE') != (row['launch_status'] == 'COMPLETE'):
                    raise ValueError('Inconsistent launch completion')
        return plan, state

    def _save(self, state, event):
        state['events'].append(event)
        try:
            _durable_write(self.root / 'ledger.json', state)
            if getattr(self, 'authority_mirror', None) is not None:
                from .authority_wire import checkpoint
                checkpoint(self)
        except BaseException:
            self.failed = True
            raise

    @staticmethod
    def _owner(state, owner, epoch):
        if type(epoch) is not int or state['owner'] != owner or state['epoch'] != epoch:
            raise RuntimeError('Owner or fencing epoch differs')

    @staticmethod
    def _counts(counts, limits):
        if set(counts) != set(limits) or any(
                type(v) is not int or not 0 <= v <= limits[k] for k, v in counts.items()):
            raise ValueError('Budget exceeded or malformed')

    def claim(self, owner):
        """Never steal custody on timeout, missing heartbeat or process restart."""
        if not isinstance(owner, dict) or not all(owner.get(k) for k in ('host', 'boot_id', 'session')):
            raise ValueError('Explicit controller identity required')
        with exclusive(self.root):
            _, state = self.read()
            if state['owner'] is not None:
                raise RuntimeError('Existing owner requires explicit recovery')
            state.update(owner=deepcopy(owner), epoch=1, status='PAUSED')
            self._save(state, dict(event='claim', owner=owner, epoch=1))
            self.session = (deepcopy(owner), 1)
            return 1

    def recover(self, previous_owner, previous_epoch, new_owner, *, reconcile):
        """Reconcile live processes/results externally, retain all reservations.

        Callback runs under the lock, returns a nonempty evidence digest only
        after proving old controller/workers stopped and budgets reconciled.
        Integrated launch reservations additionally require structured stopped,
        complete-result and consumed-budget evidence for every uncertain task.
        Recovery never clears ambiguous reservations or enables dispatch.
        """
        if not isinstance(new_owner, dict) or not all(new_owner.get(k) for k in ('host', 'boot_id', 'session')):
            raise ValueError('Explicit controller identity required')
        with self.launcher_lock(), exclusive(self.root):
            _, state = self.read()
            self._owner(state, previous_owner, previous_epoch)
            proof = reconcile(deepcopy(state))
            launched = {k for k, row in state['tasks'].items()
                        if 'launch_status' in row and row['status'] != 'COMPLETE'}
            if launched:
                if not isinstance(proof, dict) or set(proof.get('completed', {})) != launched:
                    raise RuntimeError('Every uncertain launch requires complete reconciled evidence')
                for key, evidence in proof['completed'].items():
                    row = state['tasks'][key]
                    self._counts(evidence['counts'], row['reserved'])
                    if evidence.get('stopped') is not True or not evidence.get('receipt'):
                        raise RuntimeError('Stopped worker and complete result proof required')
                    digest(evidence['receipt'])
                    row.update(status='COMPLETE', launch_status='COMPLETE',
                               consumed=deepcopy(evidence['counts']), receipt=deepcopy(evidence['receipt']))
            if isinstance(proof, dict):
                proof = proof.get('evidence_sha256')
            if not isinstance(proof, str) or len(proof) != 64 or any(c not in '0123456789abcdef' for c in proof):
                raise RuntimeError('Live reconciliation evidence required')
            state.update(owner=deepcopy(new_owner), epoch=previous_epoch + 1, status='PAUSED', stage=None, dispatch_enabled=False)
            self._save(state, dict(event='explicit_recovery', evidence_sha256=proof,
                                   previous_owner=previous_owner, epoch=state['epoch']))
            self.failed = False
            self.session = (deepcopy(new_owner), state['epoch'])
            return state['epoch']

    def rehearse_stage(self, owner, epoch, stage, *, explicit=False):
        """Each simulated stage needs explicit approval; real dispatch stays off."""
        with exclusive(self.root):
            plan, state = self.read()
            self._owner(state, owner, epoch)
            if (not explicit or self.failed or state['status'] == 'HALTED'
                    or self.session != (owner, epoch) or 'active_queue_sha256' in state):
                raise RuntimeError('Explicit recovery or rehearsal approval required')
            if stage not in plan['stages']:
                raise ValueError('Unknown stage')
            previous = plan['stages'][:plan['stages'].index(stage)]
            if any(state['tasks'].get(t['key'], {}).get('status') != 'COMPLETE'
                   for t in plan['tasks'] if t['stage'] in previous):
                raise RuntimeError('Previous stages incomplete')
            state.update(stage=stage, status='REHEARSAL')
            self._save(state, dict(event='rehearsal_stage', stage=stage))

    def reserve(self, owner, epoch, key, *, probe, execute=False):
        """Record intent once before a hypothetical launch; never start a worker.

        probe(peer) is a bounded, fresh, authenticated read, called for EVERY
        peer immediately before each reservation. Only literal True is healthy.
        Disconnects latch HALTED; reconnect alone cannot resume dispatch.
        """
        if execute:
            raise RuntimeError('Real dispatch unavailable in custody preparation')
        if self.failed:
            raise RuntimeError('Storage/probe fault requires explicit recovery')
        if self.session != (owner, epoch):
            raise RuntimeError('New controller instance requires explicit recovery')
        try:
            with exclusive(self.root):
                plan, state = self.read()
                self._owner(state, owner, epoch)
                if state['status'] != 'REHEARSAL':
                    raise RuntimeError('Dispatch paused')
                task = next((t for t in plan['tasks'] if t['key'] == key), None)
                if task is None or task['stage'] != state['stage']:
                    raise ValueError('Unknown task or unapproved stage')
                if key in state['tasks']:
                    raise RuntimeError('Duplicate dispatch/ambiguous attempt forbidden')
                try:
                    for peer in plan['peers']:
                        if probe(peer) is not True:
                            raise RuntimeError('Peer unavailable')
                except Exception:
                    state['status'] = 'HALTED'
                    self._save(state, dict(event='disconnect_stop', key=key))
                    self.failed = True
                    raise RuntimeError('Disconnected: explicit recovery required') from None
                row = dict(status='RESERVED', epoch=epoch, reserved=deepcopy(task['limits']))
                state['tasks'][key] = row
                self._save(state, dict(event='reserve_dry_run', key=key, epoch=epoch))
                return deepcopy(row)
        except OSError:
            self.failed = True
            raise

    def complete_rehearsal(self, owner, epoch, key, counts):
        if self.failed or self.session != (owner, epoch):
            raise RuntimeError('Explicit recovery required')
        with exclusive(self.root):
            _, state = self.read()
            self._owner(state, owner, epoch)
            row = state['tasks'][key]
            if 'launch_status' in row:
                raise RuntimeError('Integrated launches require backend result validation')
            if row['status'] != 'RESERVED' or row['epoch'] != epoch:
                raise RuntimeError('Completion requires original reservation epoch')
            self._counts(counts, row['reserved'])
            row.update(status='COMPLETE', consumed=deepcopy(counts))
            self._save(state, dict(event='complete_dry_run', key=key))

    def assert_real_dispatch(self):
        raise RuntimeError('Real dispatch unavailable: new binding and adapter admission required')
