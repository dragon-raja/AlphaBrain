"""Ledger-fenced adapter for the standard queue's actual backend.start path.

Only registered local/remote CPU contracts and explicitly permitted model adapters are accepted. A paused
custody record is not a permit. Legacy queue/benchmark entrypoints remain outside
this contract; production requires a separate new binding and explicit stage permit.
"""
from copy import deepcopy
import json
from pathlib import Path
import socket
import time

from . import queue
from .manifest import digest, verify_artifacts
from .dispatch_policy import _controlled_scope
from .worker_fence import check_permit, start_simulated_worker


def _simulation_only(backend, simulate, execute):
    from .simulation_backend import SimulationBackend
    from .authority_backend import AuthoritySimulationBackend
    from .model_authority import ModelAuthorityBackend
    if type(backend) is ModelAuthorityBackend:
        if (backend.production and execute is True and simulate is False) or (not backend.production and simulate is True and not execute):
            return
    elif type(backend) in (SimulationBackend, AuthoritySimulationBackend) and simulate is True and not execute:
        return
    raise RuntimeError('Production dispatch disabled without the explicit live model contract')


def _registered(backend):
    production = getattr(backend, 'production', False)
    _simulation_only(backend, not production, production)


def _ledger_mode(ledger, backend):
    if not getattr(backend, 'production', False):
        return _simulation_ledger(ledger)
    state = ledger.read()[1]
    if state.get('execution_contract') != 'LIVE_MODEL_V1' or state.get('simulation_only') is not False:
        raise RuntimeError('Original paused custody cannot become a new model run')


def _simulation_ledger(ledger):
    if ledger.read()[1].get('simulation_only') is not True:
        raise RuntimeError('CPU simulation must use an explicitly isolated simulation ledger')


class LedgerBackend:
    """Wrap every actual start, with a durable reservation and no retry window.

    Caller holds ledger.launcher_lock for the complete standard queue run. The
    controller reserves under the ledger lock. The local fake then checks the
    same ledger; the SSH fake obtains a live grant from this controller and
    never reads shared custody. Transport operations are bounded. Neither
    CPU backends cannot invoke models; the fixed model adapter requires its own live permit.
    """
    def __init__(self, ledger, owner, epoch, plan, permit, backend, *, clock=time.time, queue_root=None):
        _registered(backend)
        self.ledger, self.owner, self.epoch = ledger, deepcopy(owner), epoch
        self.plan, self.fingerprint = deepcopy(plan), digest(plan)
        self.permit, self.backend, self.clock = Path(permit), backend, clock
        self.queue_root = queue_root
        self.parallel_launches = getattr(backend, 'parallel_launches', 1)

    def _check(self, state, ledger_plan, *, dispatch=True):
        if self.ledger.failed or self.ledger.session != (self.owner, self.epoch):
            raise RuntimeError('Controller session requires explicit recovery')
        permit = json.loads(self.permit.read_text())
        stage = check_permit(state, ledger_plan, self.plan, permit, self.owner,
                             self.epoch, self.clock(), dispatch=dispatch, production=getattr(self.backend, 'production', False))
        for peer in ledger_plan['peers']:
            if self.backend.probe(peer) is not True:
                raise ConnectionError('Peer disconnected; explicit recovery required')
        return stage

    def activate(self):
        with queue.exclusive(self.ledger.root):
            plan, state = self.ledger.read()
            self.ledger._owner(state, self.owner, self.epoch)
            if state['status'] != 'PAUSED':
                raise RuntimeError('New invocation requires paused state or explicit recovery')
            permit = json.loads(self.permit.read_text())
            stage = permit.get('stage')
            if stage not in plan['stages']:
                raise ValueError('Unknown permitted stage')
            previous = plan['stages'][:plan['stages'].index(stage)]
            if any(state['tasks'].get(t['key'], {}).get('status') != 'COMPLETE'
                   for t in plan['tasks'] if t['stage'] in previous):
                raise RuntimeError('Previous stages incomplete')
            from .model_authority import ModelAuthorityBackend
            if type(self.backend) is ModelAuthorityBackend:
                self.backend.verify_previous(plan, state, previous)
            production = getattr(self.backend, 'production', False)
            state.update(status='MODEL_ACTIVE' if production else 'SIMULATION', dispatch_enabled=production,
                         stage=stage, active_queue_sha256=self.fingerprint)
            self._check(state, plan)
            # Verify all configured limits before invoking even admit/start.
            expected = {t['key']: t['limits'] for t in plan['tasks'] if t['key'] in {q['key'] for q in self.plan['tasks']}}
            if self.backend.limits != expected:
                raise ValueError('Simulation backend budget differs')
            self.ledger._save(state, dict(event='activate_controlled_simulation', queue_sha256=self.fingerprint))

    def check(self, *, dispatch=True):
        with queue.exclusive(self.ledger.root):
            plan, state = self.ledger.read()
            self._check(state, plan, dispatch=dispatch)
        return True

    def admit(self, slot):
        self.check()
        return self.backend.admit(slot)

    def _reserve(self, task, slot, attempt):
        with queue.exclusive(self.ledger.root):
            plan, state = self.ledger.read()
            self._check(state, plan)
            expected = next(t for t in self.plan['tasks'] if t['key'] == task['key'])
            if (task != expected or slot not in task.get('eligible_slots', self.plan['slots'])
                    or type(attempt) is not int or attempt < 1):
                raise ValueError('Unbound launch request')
            if task['key'] in state['tasks']:
                raise RuntimeError('Duplicate or uncertain launch: never automatically retry')
            spec = next(t for t in plan['tasks'] if t['key'] == task['key'])
            request = dict(schema='CPU_WORKER_REQUEST_V1', key=task['key'],
                custody_host=socket.gethostname(),
                custody=str(self.ledger.root.resolve()), backend=str(self.backend.root.resolve()),
                owner=deepcopy(self.owner), epoch=self.epoch, queue_sha256=self.fingerprint,
                permit_sha256=digest(json.loads(self.permit.read_text())), task_sha256=digest(task),
                slot=deepcopy(slot), attempt=attempt)
            row = dict(status='RESERVED', launch_status='RESERVED', epoch=self.epoch,
                       reserved=deepcopy(spec['limits']), queue_sha256=self.fingerprint,
                       slot=deepcopy(slot), attempt=attempt, worker_request=request)
            state['tasks'][task['key']] = row
            self.ledger._save(state, dict(event='reserve_before_backend_start', key=task['key'], epoch=self.epoch))
        return request

    def begin_start(self, task, slot, attempt):
        if self.parallel_launches <= 1 or self.queue_root is None:
            raise RuntimeError('Parallel launch requires a bound queue root')
        request = self._reserve(task, slot, attempt)
        try:
            return self.backend.begin_authorized(self.ledger, self.plan, self.permit,
                request, queue_root=self.queue_root, clock=self.clock)
        except BaseException:
            self._launch_failed(task['key'])
            raise

    def poll_start(self, ticket):
        try:
            handle = self.backend.poll_authorized(ticket)
            return None if handle is None else dict(key=ticket['key'], backend=handle)
        except BaseException:
            self._launch_failed(ticket['key'])
            raise

    def _launch_failed(self, key):
        self.ledger.failed = True
        try:
            with queue.exclusive(self.ledger.root):
                _, current = self.ledger.read()
                self.ledger._owner(current, self.owner, self.epoch)
                current['tasks'][key]['launch_status'] = 'UNKNOWN'
                current['status'] = 'HALTED'
                current['dispatch_enabled'] = False
                self.ledger._save(current, dict(event='launch_outcome_unknown', key=key))
        except Exception:
            # Never reconstruct inaccessible custody or refund uncertain work.
            pass

    def start(self, task, slot, attempt):
        request = self._reserve(task, slot, attempt)
        try:
            # Release the controller transaction before crossing the worker
            # boundary. The independent worker acquires that same lock and
            # persists its own single-use claim before backend.start.
            from .authority_backend import AuthoritySimulationBackend
            if isinstance(self.backend, AuthoritySimulationBackend):
                handle = self.backend.start_authorized(self.ledger, self.plan, self.permit,
                                                       request, clock=self.clock)
            else:
                handle = start_simulated_worker(self.ledger.root, self.plan, self.permit,
                                                self.backend, request, clock=self.clock)
            with queue.exclusive(self.ledger.root):
                plan, state = self.ledger.read()
                self._check(state, plan, dispatch=False)
                row = state['tasks'][task['key']]
                claim = row.get('worker_claim', {})
                if (row['worker_request'] != request or claim.get('outcome') != 'ACKNOWLEDGED'
                        or claim.get('handle') != handle):
                    raise RuntimeError('Worker acknowledgement differs from durable claim')
                row.update(launch_status='RUNNING', handle=deepcopy(handle))
                self.ledger._save(state, dict(event='backend_start_ack', key=task['key']))
                return dict(key=task['key'], backend=handle)
        except BaseException:
            self._launch_failed(task['key'])
            raise

    def poll(self, handle):
        self.check(dispatch=False)
        return self.backend.poll(handle['backend'])

    def poll_many(self, handles):
        from .model_authority import ModelAuthorityBackend
        if type(self.backend) is ModelAuthorityBackend:
            return self.backend.poll_many([h['backend'] for h in handles])
        return [self.poll(handle) for handle in handles]

    def alive(self, handle):
        return self.backend.alive(handle['backend'])

    def locate(self, task, slot, attempt):
        handle = self.backend.locate(task, slot, attempt)
        return None if handle is None else dict(key=task['key'], backend=handle)

    def cancel(self, handle):
        if handle.get('kind') == 'PENDING_AUTHORITY_LAUNCH_V1':
            return self.backend.cancel_pending(handle)
        return self.backend.cancel(handle['backend'])

    def validate(self, task):
        receipt = self.backend.validate(task)
        with queue.exclusive(self.ledger.root):
            plan, state = self.ledger.read()
            self._check(state, plan, dispatch=False)
            row = state['tasks'][task['key']]
            if row['status'] == 'COMPLETE':
                if row['receipt'] != receipt:
                    raise ValueError('Completed result changed')
                return receipt
            self.ledger._counts(receipt['budget_consumed'], row['reserved'])
            row.update(status='COMPLETE', launch_status='COMPLETE',
                       consumed=deepcopy(receipt['budget_consumed']), receipt=deepcopy(receipt))
            self.ledger._save(state, dict(event='validated_backend_completion', key=task['key']))
        return receipt

    def stop(self, status):
        self.ledger.failed = status == 'HALTED'
        try:
            with queue.exclusive(self.ledger.root):
                _, state = self.ledger.read()
                self.ledger._owner(state, self.owner, self.epoch)
                state['status'] = status
                state['dispatch_enabled'] = False
                self.ledger._save(state, dict(event='controlled_queue_stop', status=status))
        finally:
            from .model_authority import ModelAuthorityBackend
            if type(self.backend) is ModelAuthorityBackend:
                self.backend.close_channels()


def run_controlled_queue(root, ledger, owner, epoch, permit, backend, *,
                         simulate=False, execute=False, clock=time.time,
                         poll_seconds=0.01, timeout_seconds=30):
    """The actual new start entry: standard queue + the ledger adapter."""
    _simulation_only(backend, simulate, execute)
    _ledger_mode(ledger, backend)
    with ledger.launcher_lock():
        from .authority_backend import AuthoritySimulationBackend
        if isinstance(backend, AuthoritySimulationBackend):
            backend.bind_authority(ledger, permit)
        _, state = ledger.read()
        ledger._owner(state, owner, epoch)
        if ledger.session != (owner, epoch):
            raise RuntimeError('Controller instance requires explicit recovery')
        plan, _ = queue.load(root)
        verify_artifacts(plan)
        wrapped = LedgerBackend(ledger, owner, epoch, plan, permit, backend, clock=clock, queue_root=root)
        try:
            wrapped.activate()
            with _controlled_scope(root, wrapped, 'run'):
                result = queue.run(root, wrapped, poll_seconds=poll_seconds,
                                   timeout_seconds=timeout_seconds, allow_dispatch=wrapped.check)
        except BaseException:
            # Storage can be unavailable. Never replace the original fault or
            # permit another dispatch if durable stop recording also fails.
            ledger.failed = True
            try:
                wrapped.stop('HALTED')
            except Exception:
                pass
            raise
        wrapped.stop('PAUSED')
        return result


class LedgerRecoveryBackend:
    """CPU-only recovery view, entered after ledger reconciliation under its locks."""
    def __init__(self, backend):
        _registered(backend)
        self.backend = backend

    def locate(self, task, slot, attempt):
        handle = self.backend.locate(task, slot, attempt)
        return None if handle is None else dict(key=task['key'], backend=handle)

    def alive(self, handle):
        return self.backend.alive(handle['backend'])

    def validate(self, task):
        return self.backend.validate(task)


def recover_controlled_queue(root, ledger, old_owner, old_epoch, new_owner, backend, *,
                             simulate=False, execute=False):
    """Only complete, stopped attempts can be reconciled; partials stay held.

    Standard queue recovery may reset only tasks with no ledger reservation.
    No refund, partial-block retry or automatic recovery is offered here.
    The ledger recovery holds both launcher and transaction locks throughout.
    """
    _simulation_only(backend, simulate, execute)
    _ledger_mode(ledger, backend)
    from .authority_backend import AuthoritySimulationBackend
    if isinstance(backend, AuthoritySimulationBackend):
        backend.bind_authority(ledger)

    def reconcile(state):
        budget, _ = ledger.read()
        for peer in budget['peers']:
            if backend.probe(peer) is not True:
                raise ConnectionError('Disconnected peer prevents recovery reconciliation')
        plan, queue_state = queue.load(root)
        verify_artifacts(plan)
        known = {t['key']: t for t in plan['tasks']}
        completed = {}
        for key, row in state['tasks'].items():
            if 'launch_status' not in row:
                raise RuntimeError('Rehearsal reservations cannot become launch permission')
            if key not in known:
                if row['status'] != 'COMPLETE':
                    raise RuntimeError('Other queue has unresolved budget reservations')
                continue
            if row['queue_sha256'] != digest(plan):
                raise ValueError('Recovery queue manifest changed')
            handle = backend.locate(known[key], row['slot'], row['attempt'])
            if handle is not None and backend.alive(handle):
                raise RuntimeError('Old worker still alive; recovery refused')
            try:
                receipt = backend.validate(known[key])
            except queue.MissingResult:
                raise RuntimeError('Partial/unknown consumption: reservation retained; no retry') from None
            ledger._counts(receipt['budget_consumed'], row['reserved'])
            if row['status'] == 'COMPLETE' and row['receipt'] != receipt:
                raise ValueError('Completed result changed')
            if row['status'] != 'COMPLETE':
                completed[key] = dict(stopped=True, receipt=receipt, counts=receipt['budget_consumed'])
        # Refuse queue launch records lacking a ledger reservation, including a
        # queue that may have been run through a legacy bypass.
        for key, row in queue_state['tasks'].items():
            if row['attempt'] and key not in state['tasks']:
                raise RuntimeError('Unledgered legacy attempt requires manual reconciliation')
            if key not in state['tasks']:
                for slot in plan['slots']:
                    if backend.locate(known[key], slot, 1) is not None:
                        raise RuntimeError('Unledgered backend attempt requires manual reconciliation')
        if queue_state['status'] in ('RUNNING', 'FAILED', 'INTERRUPTED'):
            recovery_backend = LedgerRecoveryBackend(backend)
            with _controlled_scope(root, recovery_backend, 'recover'):
                queue.recover(root, recovery_backend)
        elif queue_state['status'] not in ('PREPARED', 'COMPLETE'):
            raise RuntimeError('Unsupported queue recovery state')
        return dict(evidence_sha256=digest(dict(queue=queue.load(root)[1], completed=completed)), completed=completed)

    try:
        return ledger.recover(old_owner, old_epoch, new_owner, reconcile=reconcile)
    finally:
        from .model_authority import ModelAuthorityBackend
        if type(backend) is ModelAuthorityBackend:
            backend.close_channels()
