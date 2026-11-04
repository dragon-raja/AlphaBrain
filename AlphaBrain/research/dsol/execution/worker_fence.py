"""Worker-side durable launch claims, currently restricted to the CPU fake.

The worker reopens custody independently instead of trusting a controller's
in-memory approval. No real subprocess, remote transport or model is enabled.
Remote custody requires an authoritative handshake; shared paths alone do not
establish timely visibility. This version admits same-host fakes only.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import socket
import time

from .controller_ledger import ControllerLedger
from .manifest import digest, verify_artifacts
from .queue import exclusive


def check_binding(ledger_plan, plan, stage):
    if plan['protocol_sha256'] != ledger_plan['campaign_sha256']:
        raise ValueError('Campaign/ledger binding differs')
    known = {t['key']: t for t in ledger_plan['tasks']}
    for task in plan['tasks']:
        old = known.get(task['key'])
        if old is None or old['stage'] != stage or old['addresses'] != task['addresses']:
            raise ValueError('Queue task/address/stage differs from ledger')
    if any(s['host'] not in ledger_plan['peers'] for s in plan['slots']):
        raise ValueError('Queue peer outside ledger')


def check_permit(state, ledger_plan, plan, permit, owner, epoch, now, *, dispatch=True, production=False):
    """Same persisted authorization checks at controller and worker boundaries."""
    if production and (state.get('simulation_only') is not False
                       or state.get('execution_contract') != 'LIVE_MODEL_V1'
                       or permit.get('binding_sha256') != plan.get('binding_sha256')
                       or not plan.get('binding_sha256')):
        raise RuntimeError('A newly bound production custody and permit are required')
    if not production and state.get('simulation_only') is not True:
        raise RuntimeError('Production custody cannot receive simulation consumption')
    ControllerLedger._owner(state, owner, epoch)
    stage = permit.get('stage')
    times = [permit.get(k) for k in ('issued_at', 'dispatch_until', 'finish_until')]
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
        raise ValueError('Invalid permit time window')
    issued, cutoff, finish = times
    fingerprint = digest(plan)
    if (permit.get('authorization') != ('EXPLICIT_MODEL_STAGE_V1' if production else 'CPU_SIMULATION_ONLY')
            or permit.get('ledger_plan_sha256') != state['plan_sha256']
            or permit.get('queue_sha256') != fingerprint
            or permit.get('owner') != owner
            or type(permit.get('epoch')) is not int or permit['epoch'] != epoch
            or stage not in ledger_plan['stages']
            or not issued < cutoff < finish or not issued <= now < (cutoff if dispatch else finish)):
        raise RuntimeError('Missing, stale or mismatched explicit permit')
    check_binding(ledger_plan, plan, stage)
    if (state['status'] != ('MODEL_ACTIVE' if production else 'SIMULATION') or state.get('active_queue_sha256') != fingerprint
            or state['stage'] != stage):
        raise RuntimeError('Ledger is paused or not armed for this queue')
    return stage


def start_simulated_worker(custody, plan, permit_path, backend, request, *, clock=time.time):
    """Claim once under the custody lock, then enter the actual fake start.

    The controller must release its transaction lock before this call. The
    worker takes the same lock, so epoch changes cannot overlap its launch.
    Once intent is persisted, an unknown outcome is never retried or refunded.
    Paths are trusted deployment arguments; the request cannot select custody.
    """
    from .simulation_backend import SimulationBackend
    if type(backend) is not SimulationBackend:
        raise RuntimeError('Production worker disabled; CPU simulation only')
    ledger = ControllerLedger(custody)  # independent view, no controller session
    request = deepcopy(request)
    with exclusive(ledger.root):
        budget, state = ledger.read()
        permit = json.loads(Path(permit_path).read_text())
        check_permit(state, budget, plan, permit, request['owner'], request['epoch'], clock())
        task = next((t for t in plan['tasks'] if t['key'] == request['key']), None)
        if task is None:
            raise ValueError('Unknown worker task')
        row = state['tasks'].get(task['key'])
        expected = dict(schema='CPU_WORKER_REQUEST_V1', key=task['key'],
            custody_host=socket.gethostname(),
            custody=str(ledger.root.resolve()), backend=str(backend.root.resolve()),
            owner=state['owner'], epoch=state['epoch'], queue_sha256=digest(plan),
            permit_sha256=digest(permit), task_sha256=digest(task),
            slot=request.get('slot'), attempt=request.get('attempt'))
        if (request != expected or request['slot'] not in plan['slots']
                or type(request['attempt']) is not int or request['attempt'] < 1
                or row is None or row.get('worker_request') != request
                or row['status'] != 'RESERVED' or row['launch_status'] != 'RESERVED'
                or row['epoch'] != request['epoch'] or row['slot'] != request['slot']
                or row['attempt'] != request['attempt'] or row['queue_sha256'] != digest(plan)):
            raise RuntimeError('Worker request lacks its exact durable reservation')
        if 'worker_claim' in row:
            raise RuntimeError('Duplicate or unknown worker claim; explicit reconciliation required')
        if backend.limits.get(task['key']) != row['reserved']:
            raise ValueError('Worker budget differs from durable reservation')
        verify_artifacts(plan)
        for peer in budget['peers']:
            if backend.probe(peer) is not True:
                raise ConnectionError('Worker peer disconnected')
        row['worker_claim'] = dict(request_sha256=digest(request), outcome='INTENT')
        ledger._save(state, dict(event='worker_claim_before_start', key=task['key']))
        try:
            # Reopen the durable record and authorization after fsync, before
            # any start. Never silently recreate a missing ledger from memory.
            fresh_budget, fresh = ledger.read()
            current_permit = json.loads(Path(permit_path).read_text())
            check_permit(fresh, fresh_budget, plan, current_permit, request['owner'], request['epoch'], clock())
            if fresh['tasks'][task['key']] != row or digest(current_permit) != request['permit_sha256']:
                raise RuntimeError('Worker intent or permit changed before start')
            for peer in fresh_budget['peers']:
                if backend.probe(peer) is not True:
                    raise ConnectionError('Worker peer disconnected before start')
            handle = backend.start(deepcopy(task), deepcopy(request['slot']), request['attempt'])
            if not isinstance(handle, dict) or not handle:
                raise ValueError('Durable worker handle required')
            digest(handle)
            row['worker_claim'].update(outcome='ACKNOWLEDGED', handle=deepcopy(handle))
            ledger._save(state, dict(event='worker_start_ack', key=task['key']))
            return handle
        except BaseException:
            try:
                _, fresh = ledger.read()
                ledger._owner(fresh, request['owner'], request['epoch'])
                fresh['tasks'][task['key']]['worker_claim']['outcome'] = 'UNKNOWN'
                fresh['status'] = 'HALTED'
                ledger._save(fresh, dict(event='worker_outcome_unknown', key=task['key']))
            except Exception:
                pass
            raise
