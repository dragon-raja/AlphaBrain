"""Exercise the actual standard queue launch entry with a persistent CPU fake."""
from copy import deepcopy
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from AlphaBrain.research.dsol.execution import queue
from AlphaBrain.research.dsol.execution.controller_ledger import ControllerLedger, prepare
from AlphaBrain.research.dsol.execution.controlled_queue import (
    recover_controlled_queue, run_controlled_queue,
)
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key
from AlphaBrain.research.dsol.execution.simulation_backend import SimulationBackend


from tests.dsol_paper1.helpers.controller_cases import (
    OWNER, NEXT, make_case, make_authority_case, run, recover, update_permit,
)
REPO = Path(__file__).resolve().parents[3]
ENTRY = REPO / 'scripts/dsol_paper1/operations/manage_controlled_queue.py'




@pytest.fixture
def case(tmp_path):
    return make_case(tmp_path)




def test_real_entry_reserves_before_every_backend_start_and_preserves_negative_results(case, monkeypatch):
    original = SimulationBackend.start
    observed = []
    def inspect_start(self, task, slot, attempt):
        state = case['ledger'].read()[1]; row = state['tasks'][task['key']]
        assert row['status'] == row['launch_status'] == 'RESERVED'
        assert row['reserved'] == dict(calls=2, episodes=1)
        observed.append(task['key'])
        return original(self, task, slot, attempt)
    monkeypatch.setattr(SimulationBackend, 'start', inspect_start)
    assert run(case)['status'] == 'COMPLETE'
    assert len(observed) == 3
    state = case['ledger'].read()[1]
    assert state['status'] == 'PAUSED' and state['dispatch_enabled'] is False
    assert sum(t['reserved']['calls'] for t in state['tasks'].values()) == 6
    assert all(t['receipt']['scientific_success'] is False for t in state['tasks'].values())
    assert not (case['root'] / 'unused-output').exists()


@pytest.mark.parametrize('fault', ['absent', 'expired', 'epoch', 'owner', 'queue', 'budget', 'custody_only', 'nan'])
def test_missing_or_invalid_permit_never_reaches_start(case, fault):
    changes = {
        'expired': dict(dispatch_until=time.time() - 1), 'epoch': dict(epoch=0),
        'owner': dict(owner=NEXT), 'queue': dict(queue_sha256='a' * 64),
        'budget': dict(ledger_plan_sha256='b' * 64),
        'custody_only': dict(authorization='PAUSED'), 'nan': dict(finish_until=float('nan')),
    }
    if fault == 'absent': case['permit'].unlink()
    else: update_permit(case, **changes[fault])
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)): run(case)
    assert case['backend'].read()['attempts'] == {}
    assert case['ledger'].read()[1]['tasks'] == {}


def test_stale_owner_and_real_execution_rejected_before_start(case):
    with pytest.raises(RuntimeError): run(case, epoch=0)
    assert case['ledger'].read()[1]['status'] == 'PAUSED'
    with pytest.raises(RuntimeError, match='disabled'): run(case, execute=True)
    assert case['backend'].read()['attempts'] == {}


def test_original_custody_cannot_be_used_for_simulated_consumption(case):
    path = case['ledger'].root / 'ledger.json'
    state = json.loads(path.read_text()); state.pop('simulation_only')
    path.write_text(json.dumps(state)); before = file_digest(path)
    with pytest.raises(RuntimeError, match='isolated'): run(case)
    with pytest.raises(RuntimeError, match='isolated'): recover(case)
    assert file_digest(path) == before and case['backend'].read()['attempts'] == {}


def test_offline_peer_prevents_first_start_and_recovery(case):
    case['backend'].offline.add('video')
    with pytest.raises(ConnectionError): run(case)
    with pytest.raises(ConnectionError): recover(case)
    assert case['backend'].read()['attempts'] == {}


def test_second_controller_and_recovery_excluded_while_actual_start_runs(case, monkeypatch):
    original = SimulationBackend.start
    observed = []
    def compete(self, task, slot, attempt):
        with pytest.raises(BlockingIOError): run(case)
        with pytest.raises(BlockingIOError): recover(case, ControllerLedger(case['ledger'].root))
        observed.append(task['key'])
        return original(self, task, slot, attempt)
    monkeypatch.setattr(SimulationBackend, 'start', compete)
    run(case)
    assert len(observed) == 3


def test_different_queue_cannot_duplicate_scientific_tasks(case):
    run(case)
    # New control directory is not a way to obtain new scientific ownership.
    second = case['root'] / 'another-queue'; queue.prepare(case['plan'], second)
    with pytest.raises(RuntimeError, match='Duplicate'):
        run_controlled_queue(second, case['ledger'], OWNER, 1, case['permit'], case['backend'],
                             simulate=True, poll_seconds=0)
    assert len(case['backend'].read()['attempts']) == 3


def crash_child(root, when):
    root = Path(root)
    ledger = ControllerLedger(root / 'custody'); ledger.claim(OWNER)
    budget, _ = ledger.read()
    backend = SimulationBackend(root / 'backend', {t['key']: t['limits'] for t in budget['tasks']})
    original = SimulationBackend.start
    def crash(self, task, slot, attempt):
        if when == 'after_start': original(self, task, slot, attempt)
        os._exit(73)  # actual controller death, no finally/cleanup handlers
    SimulationBackend.start = crash
    run_controlled_queue(root / 'queue', ledger, OWNER, 1, root / 'permit.json', backend, simulate=True)


@pytest.mark.parametrize('when', ['before_start', 'after_start'])
def test_process_crash_between_reservation_and_ack(tmp_path, when):
    case = make_case(tmp_path, claim=False)
    p = get_context('spawn').Process(target=crash_child, args=(str(tmp_path), when))
    p.start(); p.join(15)
    assert p.exitcode == 73
    state = case['ledger'].read()[1]
    assert len(state['tasks']) == 1
    assert next(iter(state['tasks'].values()))['reserved'] == dict(calls=2, episodes=1)
    assert next(iter(state['tasks'].values()))['launch_status'] == 'RESERVED'
    if when == 'before_start':
        with pytest.raises(RuntimeError, match='Partial/unknown'): recover(case)
        assert case['backend'].read()['attempts'] == {}
    else:
        assert recover(case) == 2
        assert case['ledger'].read()[1]['status'] == 'PAUSED'
        assert len(case['backend'].read()['attempts']) == 1
        update_permit(case, owner=NEXT, epoch=2)
        assert run(case, NEXT, 2)['status'] == 'COMPLETE'
        assert len(case['backend'].read()['attempts']) == 3


@pytest.mark.parametrize('mode', ['unknown', 'partial'])
def test_unknown_launch_and_partial_block_never_refunded_or_reissued(case, mode):
    key = case['plan']['tasks'][0]['key']; case['backend'].faults[key] = mode
    with pytest.raises((ConnectionError, RuntimeError)): run(case)
    with pytest.raises(RuntimeError, match='Partial/unknown'): recover(case)
    state = case['ledger'].read()[1]
    assert state['tasks'][key]['status'] == 'RESERVED'
    assert state['tasks'][key]['reserved'] == dict(calls=2, episodes=1)
    assert len(case['backend'].read()['attempts']) == 1
    with pytest.raises(RuntimeError):
        case['ledger'].recover(OWNER, 1, NEXT, reconcile=lambda _: 'a' * 64)
    with pytest.raises(RuntimeError): run(case)
    assert len(case['backend'].read()['attempts']) == 1


def test_disconnect_after_one_completed_start_needs_explicit_recovery(case, monkeypatch):
    original = SimulationBackend.start
    def disconnect(self, task, slot, attempt):
        result = original(self, task, slot, attempt); self.offline.add('gnu'); return result
    monkeypatch.setattr(SimulationBackend, 'start', disconnect)
    with pytest.raises(ConnectionError): run(case)
    assert len(case['backend'].read()['attempts']) == 1
    case['backend'].offline.clear()
    with pytest.raises(RuntimeError): run(case)
    monkeypatch.setattr(SimulationBackend, 'start', original)
    assert recover(case) == 2
    with pytest.raises(RuntimeError): run(case, OWNER, 1)
    assert case['ledger'].read()[1]['status'] == 'PAUSED'
    update_permit(case, owner=NEXT, epoch=2)
    assert run(case, NEXT, 2)['status'] == 'COMPLETE'
    assert len(case['backend'].read()['attempts']) == 3


@pytest.mark.parametrize('fault', ['permit', 'connectivity', 'write', 'missing_ledger'])
def test_final_launch_recheck_and_unavailable_ledger_fail_closed(case, monkeypatch, fault):
    ledger = case['ledger']; original = ledger._save
    def fault_at_reservation(state, event):
        if event['event'] != 'reserve_before_backend_start': return original(state, event)
        if fault == 'write': raise OSError('injected fsync failure')
        original(state, event)
        if fault == 'permit': update_permit(case, epoch=99)
        if fault == 'connectivity': case['backend'].offline.add('fresh')
        if fault == 'missing_ledger': (ledger.root / 'ledger.json').rename(ledger.root / 'offline-ledger.json')
    monkeypatch.setattr(ledger, '_save', fault_at_reservation)
    with pytest.raises((OSError, RuntimeError)): run(case)
    assert case['backend'].read()['attempts'] == {}
    if fault == 'missing_ledger':
        assert not (ledger.root / 'ledger.json').exists()  # no stale-memory recreation
    else:
        assert ledger.read()[1]['status'] == 'HALTED'


def test_lost_ack_recovers_completed_attempt_without_second_start(case, monkeypatch):
    ledger = case['ledger']; original = ledger._save
    def lose_ack(state, event):
        if event['event'] == 'backend_start_ack': raise OSError('injected lost launch acknowledgement')
        return original(state, event)
    monkeypatch.setattr(ledger, '_save', lose_ack)
    with pytest.raises(OSError): run(case)
    monkeypatch.setattr(ledger, '_save', original)
    assert len(case['backend'].read()['attempts']) == 1
    assert recover(case) == 2
    update_permit(case, owner=NEXT, epoch=2)
    assert run(case, NEXT, 2)['status'] == 'COMPLETE'
    assert len(case['backend'].read()['attempts']) == 3


def test_scientific_queue_binding_tamper_rejected(case):
    path = case['root'] / 'queue/manifest.json'
    value = json.loads(path.read_text()); value['slots'][0]['host'] = 'unbound'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): run(case)
    assert case['backend'].read()['attempts'] == {}


def test_unledgered_backend_bypass_is_not_adopted_during_recovery(case):
    task = case['plan']['tasks'][0]
    case['backend'].start(task, case['plan']['slots'][0], 1)
    with pytest.raises(RuntimeError, match='Unledgered'): recover(case)
    assert case['ledger'].read()[1]['tasks'] == {}


def test_backend_receipt_over_budget_keeps_full_reservation(case, monkeypatch):
    original = SimulationBackend.validate
    def over_budget(self, task):
        receipt = original(self, task)
        receipt['budget_consumed']['calls'] = 3
        return receipt
    monkeypatch.setattr(SimulationBackend, 'validate', over_budget)
    with pytest.raises(ValueError, match='Budget'): run(case)
    row = next(iter(case['ledger'].read()[1]['tasks'].values()))
    assert row['status'] == 'RESERVED' and row['reserved']['calls'] == 2


def test_unknown_or_live_worker_blocks_explicit_recovery(case, monkeypatch):
    key = case['plan']['tasks'][0]['key']; case['backend'].faults[key] = 'unknown'
    monkeypatch.setattr(SimulationBackend, 'cancel', lambda *_: None)
    with pytest.raises(ConnectionError): run(case)
    with pytest.raises(RuntimeError, match='still alive'): recover(case)
    def unknown(*_): raise ConnectionError('Cannot establish worker identity')
    monkeypatch.setattr(SimulationBackend, 'locate', unknown)
    with pytest.raises(ConnectionError): recover(case)
    assert case['ledger'].read()[1]['tasks'][key]['reserved']['calls'] == 2


def test_cli_uses_same_entry_and_rejects_production_or_second_owner(tmp_path):
    case = make_case(tmp_path, claim=False)
    owner = tmp_path / 'owner.json'; owner.write_text(json.dumps(OWNER))
    cmd = [sys.executable, str(ENTRY), 'run', '--root', str(tmp_path / 'queue'),
           '--custody', str(tmp_path / 'custody'), '--owner', str(owner), '--epoch', '1',
           '--permit', str(case['permit']), '--backend-state', str(tmp_path / 'backend'), '--simulate']
    rejected = subprocess.run(cmd + ['--execute'], capture_output=True, text=True, timeout=10)
    assert rejected.returncode != 0 and 'disabled' in rejected.stderr
    assert case['backend'].read()['attempts'] == {}
    accepted = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)['dispatch_enabled'] is False
    repeated = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    assert repeated.returncode != 0 and 'explicit recovery' in repeated.stderr
    assert len(case['backend'].read()['attempts']) == 3


@pytest.mark.parametrize('fault', ['owner', 'epoch', 'permit', 'task', 'slot', 'attempt',
                                  'custody', 'custody_host', 'backend', 'offline', 'missing', 'artifact'])
def test_worker_reopens_custody_and_rejects_changed_launch(case, monkeypatch, fault):
    from AlphaBrain.research.dsol.execution import controlled_queue
    original = controlled_queue.start_simulated_worker
    def intercepted(custody, plan, permit, backend, request, **kwargs):
        request = deepcopy(request)
        if fault in ('owner', 'epoch'):
            path = Path(custody) / 'ledger.json'
            state = json.loads(path.read_text())
            state[fault] = NEXT if fault == 'owner' else 2
            queue._durable_write(path, state)
        elif fault == 'permit':
            update_permit(case, dispatch_until=time.time() - 1)
        elif fault == 'task': request['task_sha256'] = 'f' * 64
        elif fault == 'slot': request['slot'] = dict(host='gnu', id='cpu-fake')
        elif fault == 'attempt': request['attempt'] = 2
        elif fault == 'custody': request['custody'] = str(case['root'] / 'other')
        elif fault == 'custody_host': request['custody_host'] = 'another-compute-node'
        elif fault == 'backend': request['backend'] = str(case['root'] / 'other')
        elif fault == 'offline': backend.offline.add('video')
        elif fault == 'missing': (Path(custody) / 'ledger.json').rename(Path(custody) / 'saved.json')
        elif fault == 'artifact': (case['root'] / 'protocol.json').write_text('changed')
        return original(custody, plan, permit, backend, request, **kwargs)
    monkeypatch.setattr(controlled_queue, 'start_simulated_worker', intercepted)
    with pytest.raises((RuntimeError, ValueError, OSError)): run(case)
    assert case['backend'].read()['attempts'] == {}
    if fault != 'missing':
        row = next(iter(case['ledger'].read()[1]['tasks'].values()))
        assert row['reserved'] == dict(calls=2, episodes=1)


def test_worker_claim_prevents_duplicate_delivery_even_without_controller_ack(case, monkeypatch):
    from AlphaBrain.research.dsol.execution import controlled_queue
    original = controlled_queue.start_simulated_worker
    observed = []
    def deliver_twice(*args, **kwargs):
        handle = original(*args, **kwargs)
        with pytest.raises(RuntimeError, match='Duplicate'):
            original(*args, **kwargs)
        observed.append(handle)
        return handle
    monkeypatch.setattr(controlled_queue, 'start_simulated_worker', deliver_twice)
    assert run(case)['status'] == 'COMPLETE'
    assert len(observed) == len(case['backend'].read()['attempts']) == 3


@pytest.mark.parametrize('fault', ['write', 'disconnect', 'missing_ledger', 'ack'])
def test_worker_intent_failures_keep_budget_and_forbid_repeat(case, monkeypatch, fault):
    original = ControllerLedger._save
    def fail(self, state, event):
        if event['event'] == 'worker_claim_before_start':
            if fault == 'write': raise OSError('worker fsync failed')
            original(self, state, event)
            if fault == 'disconnect': case['backend'].offline.add('gnu')
            if fault == 'missing_ledger': (self.root / 'ledger.json').rename(self.root / 'saved.json')
            return
        if event['event'] == 'worker_start_ack' and fault == 'ack':
            raise OSError('worker acknowledgement lost')
        original(self, state, event)
    monkeypatch.setattr(ControllerLedger, '_save', fail)
    with pytest.raises((RuntimeError, OSError)): run(case)
    assert len(case['backend'].read()['attempts']) == (1 if fault == 'ack' else 0)
    if fault != 'missing_ledger':
        row = next(iter(case['ledger'].read()[1]['tasks'].values()))
        assert row['reserved']['calls'] == 2 and row['status'] == 'RESERVED'
    monkeypatch.setattr(ControllerLedger, '_save', original)
    if fault == 'ack':
        assert recover(case) == 2
        assert len(case['backend'].read()['attempts']) == 1


def independent_worker(connection, custody, plan, permit, backend_root, limits, request):
    from AlphaBrain.research.dsol.execution.worker_fence import start_simulated_worker
    try:
        backend = SimulationBackend(backend_root, limits)
        handle = start_simulated_worker(custody, plan, permit, backend, request)
        connection.send(dict(handle=handle, pid=os.getpid()))
    except Exception as error:
        connection.send(dict(error=repr(error)))
    finally:
        connection.close()


def test_actual_entry_crosses_independent_worker_process_without_lock_deadlock(case, monkeypatch):
    from AlphaBrain.research.dsol.execution import controlled_queue
    pids = []
    def transport(custody, plan, permit, backend, request, **kwargs):
        context = get_context('spawn')
        parent, child = context.Pipe()
        worker = context.Process(target=independent_worker, args=(child, str(custody), plan,
            str(permit), str(backend.root), backend.limits, json.loads(json.dumps(request))))
        worker.start(); child.close()
        try:
            assert parent.poll(10), 'Worker lock acquisition deadlocked'
            value = parent.recv()
            worker.join(5)
            assert worker.exitcode == 0 and 'error' not in value, value
            pids.append(value['pid'])
            return value['handle']
        finally:
            if worker.is_alive(): worker.terminate(); worker.join(5)
            parent.close()
    monkeypatch.setattr(controlled_queue, 'start_simulated_worker', transport)
    assert run(case)['status'] == 'COMPLETE'
    assert len(pids) == 3 and os.getpid() not in pids
    state = case['ledger'].read()[1]
    assert all(r['worker_claim']['outcome'] == 'ACKNOWLEDGED' for r in state['tasks'].values())


def test_real_backend_rejected_at_worker_boundary(case):
    from AlphaBrain.research.dsol.execution.worker_fence import start_simulated_worker
    with pytest.raises(RuntimeError, match='Production worker disabled'):
        start_simulated_worker(case['ledger'].root, case['plan'], case['permit'], object(), {})




@pytest.fixture
def authority_case(tmp_path):
    return make_authority_case(tmp_path)


def test_actual_queue_uses_live_authority_not_poisoned_shared_snapshot(authority_case):
    case = authority_case
    mirror = Path(case['backend'].config['checkpoint_directory'])
    (mirror/'ledger.json').write_text('{"epoch":999,"dispatch_enabled":true}')
    assert run(case, timeout_seconds=30)['status'] == 'COMPLETE'
    state = case['ledger'].read()[1]
    assert len(state['tasks']) == 3 and all(r['consumed']['calls'] == 2 for r in state['tasks'].values())
    assert len(list(mirror.glob('*.json'))) > 10
    for p in (case['root']/'remote-store/tasks').glob('*.json'):
        row = json.loads(p.read_text())
        assert row['starts'] == 1 and row['status'] == 'COMPLETE'
        assert 'custody' not in row['launch'] and 'permit' not in row['launch']


@pytest.mark.parametrize('point', ['before_go', 'lost_ack', 'mirror_failure', 'revoke_after_ready'])
def test_authority_handshake_faults_keep_reservations(authority_case, monkeypatch, point):
    from AlphaBrain.research.dsol.execution import authority_wire
    case = authority_case
    original_send, original_receive = authority_wire.Wire.send, authority_wire.Wire.receive
    original_checkpoint = authority_wire.checkpoint
    def send(self, message):
        if point == 'before_go' and message.get('kind') == 'GO':
            raise ConnectionError('Injected authority disconnect before GO')
        return original_send(self, message)
    def receive(self):
        value = original_receive(self)
        if point == 'lost_ack' and value.get('kind') == 'ACK':
            raise ConnectionError('Injected lost ACK')
        if point == 'revoke_after_ready' and value.get('kind') == 'READY':
            update_permit(case, dispatch_until=time.time()-1)
        return value
    def checkpoint(ledger):
        if point == 'mirror_failure' and ledger.read()[1]['events'][-1]['event'] == 'authority_grant_before_go':
            raise OSError('Injected persistent mirror failure')
        return original_checkpoint(ledger)
    monkeypatch.setattr(authority_wire.Wire, 'send', send)
    monkeypatch.setattr(authority_wire.Wire, 'receive', receive)
    monkeypatch.setattr(authority_wire, 'checkpoint', checkpoint)
    with pytest.raises((RuntimeError, OSError)): run(case, timeout_seconds=30)
    state = case['ledger'].read()[1]
    assert state['status'] == 'HALTED' and len(state['tasks']) == 1
    row = next(iter(state['tasks'].values()))
    assert row['status'] == 'RESERVED' and row['reserved']['calls'] == 2
    remote = json.loads(next((case['root']/'remote-store/tasks').glob('*.json')).read_text())
    assert remote['starts'] == (1 if point == 'lost_ack' else 0)
    monkeypatch.setattr(authority_wire.Wire, 'send', original_send)
    monkeypatch.setattr(authority_wire.Wire, 'receive', original_receive)
    monkeypatch.setattr(authority_wire, 'checkpoint', original_checkpoint)
    if point == 'lost_ack':
        assert recover(case) == 2
        update_permit(case, owner=NEXT, epoch=2)
        assert run(case, NEXT, 2, timeout_seconds=30)['status'] == 'COMPLETE'
        assert all(json.loads(p.read_text())['starts'] == 1 for p in (case['root']/'remote-store/tasks').glob('*.json'))
    else:
        with pytest.raises(RuntimeError, match='Partial/unknown'): recover(case)


def test_remote_duplicate_and_stale_session_cannot_start_twice(authority_case):
    from AlphaBrain.research.dsol.execution.authority_wire import Channel
    case = authority_case
    run(case, timeout_seconds=30)
    entry = next(iter(json.loads(case['backend'].path.read_text())['attempts'].values()))
    target = case['backend'].config['peers']['fresh']
    for stale in [False, True]:
        with Channel(target) as channel:
            channel.wire.send(dict(kind='PREPARE', key=entry['launch']['key'], launch=entry['launch'],
                                   session='old-session' if stale else channel.hello['session']))
            with pytest.raises(RuntimeError): channel.wire.receive()
    assert all(json.loads(p.read_text())['starts'] == 1 for p in (case['root']/'remote-store/tasks').glob('*.json'))


def test_direct_remote_backend_cannot_skip_center_grant(authority_case):
    with pytest.raises(RuntimeError, match='authority boundary'):
        authority_case['backend'].start({}, {}, 1)


@pytest.mark.parametrize('field', ['identity', 'store_id', 'source'])
def test_changed_remote_binding_refused(authority_case, field):
    case = authority_case
    expected = case['backend'].config['peers']['fresh']['expected']
    expected[field] = {} if field == 'identity' else 'changed'
    with pytest.raises(RuntimeError): run(case, timeout_seconds=30)
    assert not list((case['root']/'remote-store/tasks').glob('*.json'))


def authority_crash_child(root, point):
    from AlphaBrain.research.dsol.execution.authority_backend import AuthoritySimulationBackend
    from AlphaBrain.research.dsol.execution.authority_wire import Wire
    root = Path(root)
    ledger = ControllerLedger(root/'custody'); ledger.claim(OWNER)
    budget, _ = ledger.read()
    backend = AuthoritySimulationBackend(root/'authority-backend', {t['key']: t['limits'] for t in budget['tasks']})
    original = Wire.send
    def die(self, value):
        if value.get('kind') == 'GO':
            if point == 'before_go': os._exit(81)
            original(self, value)
            os._exit(82)
        return original(self, value)
    Wire.send = die
    run_controlled_queue(root/'queue', ledger, OWNER, 1, root/'permit.json', backend,
                         simulate=True, timeout_seconds=30)


@pytest.mark.parametrize('point,code', [('before_go',81),('after_go',82)])
def test_controller_process_death_has_no_automatic_replay(tmp_path, point, code):
    case = make_authority_case(tmp_path, claim=False)
    child = get_context('spawn').Process(target=authority_crash_child, args=(str(tmp_path),point))
    child.start(); child.join(20)
    assert child.exitcode == code
    row = next(iter(case['ledger'].read()[1]['tasks'].values()))
    assert row['reserved']['calls'] == 2
    # Explicit read-only reconciliation waits for the bounded remote handler.
    deadline = time.monotonic()+5
    while True:
        try:
            case['backend']._inspect('fresh', next(iter(case['ledger'].read()[1]['tasks'])))
            break
        except RuntimeError:
            if time.monotonic() >= deadline: raise
            time.sleep(.05)
    if point == 'before_go':
        with pytest.raises(RuntimeError, match='Partial/unknown'): recover(case)
    else:
        assert recover(case) == 2
        assert len(case['ledger'].read()[1]['tasks']) == 1


def test_authority_live_state_rejects_network_filesystem(tmp_path, monkeypatch):
    from AlphaBrain.research.dsol.execution import authority_wire
    monkeypatch.setattr(authority_wire.subprocess, 'check_output', lambda *a, **k: 'nfs4\n')
    with pytest.raises(RuntimeError, match='local filesystem'):
        authority_wire.require_local(tmp_path)


def test_missing_authority_launch_history_never_auto_recreated(authority_case):
    from AlphaBrain.research.dsol.execution.authority_backend import AuthoritySimulationBackend
    backend = authority_case['backend']
    backend.path.unlink()
    with pytest.raises(RuntimeError, match='records missing'):
        AuthoritySimulationBackend(backend.root, backend.limits)
    assert not backend.path.exists()


def test_expiration_during_checkpoint_never_sends_go(authority_case, monkeypatch):
    from AlphaBrain.research.dsol.execution import authority_wire
    case = authority_case
    original = authority_wire.checkpoint
    def checkpoint(ledger):
        if ledger.read()[1]['events'][-1]['event'] == 'authority_grant_before_go':
            update_permit(case, dispatch_until=time.time()-1)
        return original(ledger)
    monkeypatch.setattr(authority_wire, 'checkpoint', checkpoint)
    with pytest.raises(RuntimeError): run(case, timeout_seconds=30)
    remote = json.loads(next((case['root']/'remote-store/tasks').glob('*.json')).read_text())
    assert remote['starts'] == 0
    assert next(iter(case['ledger'].read()[1]['tasks'].values()))['reserved']['calls'] == 2


def test_remote_partial_block_never_refunded_or_replayed(authority_case, monkeypatch):
    from AlphaBrain.research.dsol.execution import authority_wire
    case = authority_case
    original = authority_wire.Wire.receive
    def receive(self):
        value = original(self)
        if value.get('kind') == 'ACK':
            path = next((case['root']/'remote-store/tasks').glob('*.json'))
            row = json.loads(path.read_text()); row.update(status='INTENT'); row.pop('receipt')
            path.write_text(json.dumps(row))
            raise ConnectionError('Injected partial block without complete receipt')
        return value
    monkeypatch.setattr(authority_wire.Wire, 'receive', receive)
    with pytest.raises(ConnectionError): run(case, timeout_seconds=30)
    monkeypatch.setattr(authority_wire.Wire, 'receive', original)
    with pytest.raises(RuntimeError, match='Partial/unknown'): recover(case)
    assert next(iter(case['ledger'].read()[1]['tasks'].values()))['reserved']['calls'] == 2
    with pytest.raises(RuntimeError): run(case, timeout_seconds=30)
    assert len(list((case['root']/'remote-store/tasks').glob('*.json'))) == 1


def test_authority_cli_uses_live_protocol_and_rejects_execute(tmp_path):
    case = make_authority_case(tmp_path, claim=False)
    config, owner = tmp_path/'authority.json', tmp_path/'owner.json'
    config.write_text(json.dumps(case['backend'].config)); owner.write_text(json.dumps(OWNER))
    args = [sys.executable, '-B', str(ENTRY), 'run', '--root', str(tmp_path/'queue'),
            '--custody', str(case['ledger'].root), '--owner', str(owner), '--epoch', '1',
            '--permit', str(case['permit']), '--backend-state', str(case['backend'].root),
            '--authority-config', str(config), '--simulate']
    result = subprocess.run(args, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['queue']['status'] == 'COMPLETE'
    result = subprocess.run(args+['--execute'], text=True, capture_output=True, timeout=5)
    assert result.returncode != 0 and 'Production dispatch is disabled' in result.stderr


@pytest.mark.parametrize('lost_session', [True, False])
def test_direct_authority_rejects_lost_session_or_failure_latch(authority_case, lost_session):
    case = authority_case
    if lost_session:
        case['ledger'] = ControllerLedger(case['ledger'].root)
    else:
        case['ledger'].failed = True
    with pytest.raises(RuntimeError, match='requires explicit recovery'):
        case['backend'].start_authorized(case['ledger'], case['plan'], case['permit'],
                                          dict(owner=OWNER, epoch=1))
    assert not list((case['root']/'remote-store/tasks').glob('*.json'))
