"""Real standard entry and CPU worker protocol; no model/GPU side effects."""
import json
import os
from pathlib import Path
import threading
import time

import pytest

from AlphaBrain.research.dsol.execution.authority_wire import Wire
from AlphaBrain.research.dsol.execution.parallel_launch import ParallelLaunches
from AlphaBrain.research.dsol.execution.controlled_queue import recover_controlled_queue
from AlphaBrain.research.dsol.execution.resources import set_drain
from tests.dsol_paper1.helpers.model_cases import make_model_case, run_model_case as run
from tests.dsol_paper1.helpers.controller_cases import OWNER, NEXT


def make(root, **kwargs):
    return make_model_case(root, parallel_launches=4, resident=True, persistent_inspection=True,
        slots=[dict(host='fresh', id=str(i)) for i in range(4)], task_count=8, **kwargs)


def records(case):
    return [json.loads(p.read_text()) for p in (case['worker']/'tasks').glob('*.json')]


def test_parallel_actual_entry_completes_exactly_once_and_reuses(tmp_path):
    case = make(tmp_path, checks=8)
    assert run(case)['status'] == 'COMPLETE'
    rows = records(case)
    assert len(rows) == 8 and all(r['starts'] == 1 and r['status'] == 'COMPLETE' for r in rows)
    metrics = case['backend'].metrics
    assert 1 < metrics['max_pending_launches'] <= 4
    assert metrics['continuation_grants'] == 64
    assert metrics['worker_processes'] <= 4 and metrics['resident_reuses'] >= 4
    _, state = case['ledger'].read()
    assert all(r['status'] == 'COMPLETE' and r['consumed'] == r['reserved'] for r in state['tasks'].values())
    assert state['dispatch_enabled'] is False


def test_slow_preparation_does_not_block_other_worker_continuations(tmp_path, monkeypatch):
    case = make(tmp_path, checks=16)
    original = Wire.receive_ready
    delayed = set(); progressed = []
    def receive(wire):
        # Defer one actual READY until another real worker has computed.
        pending = case['backend'].pending_launches
        item = next((v for v in pending.pending.values() if v.get('channel') is not None
                     and v['channel'].wire is wire), None) if pending else None
        if item and item['phase'] == 'PREPARING' and (not delayed or item['ticket']['key'] in delayed):
            delayed.add(item['ticket']['key'])
            if case['backend'].metrics['continuation_grants'] < 4:
                return None
            progressed.append(case['backend'].metrics['continuation_grants'])
        return original(wire)
    monkeypatch.setattr(Wire, 'receive_ready', receive)
    assert run(case)['status'] == 'COMPLETE'
    assert progressed and min(progressed) >= 4
    assert all(r['starts'] == 1 for r in records(case))


def test_connections_overlap_and_pending_bound_is_enforced(tmp_path, monkeypatch):
    case = make(tmp_path)
    original = ParallelLaunches._open
    barrier = threading.Barrier(4, timeout=10)
    lock = threading.Lock(); arrivals = []
    def connect(engine, target):
        with lock: arrivals.append(threading.get_ident())
        barrier.wait()
        return original(engine, target)
    monkeypatch.setattr(ParallelLaunches, '_open', connect)
    assert run(case)['status'] == 'COMPLETE'
    assert len(set(arrivals)) == 4
    assert case['backend'].metrics['max_pending_launches'] == 4


@pytest.mark.parametrize('fault', ['epoch', 'permit', 'ledger_loss', 'drain', 'disconnect', 'lost_ack', 'bad_ack'])
def test_uncertain_parallel_start_halts_without_refund_or_retry(tmp_path, monkeypatch, fault):
    case = make(tmp_path, checks=8)
    original = Wire.receive_ready; injected = []
    def receive(wire):
        value = original(wire)
        target = 'ACK' if fault in ('lost_ack', 'bad_ack') else 'READY'
        if value and value.get('kind') == target and not injected:
            injected.append(True)
            if fault == 'epoch':
                path = case['ledger'].root/'ledger.json';state = json.loads(path.read_text())
                state['epoch'] += 1;path.write_text(json.dumps(state))
            elif fault == 'permit':
                path = case['permit'];value_ = json.loads(path.read_text());value_['dispatch_until'] = 0
                path.write_text(json.dumps(value_))
            elif fault == 'ledger_loss':
                (case['ledger'].root/'ledger.json').unlink()
            elif fault == 'drain':
                for slot in case['plan']['slots']:
                    set_drain(case['queue'], slot['host'], slot['id'], reason='injected before GO')
            elif fault == 'disconnect':
                monkeypatch.setattr(case['backend'], 'probe_many', lambda hosts: (_ for _ in ()).throw(ConnectionError('injected peer loss')))
            elif fault == 'lost_ack':
                raise ConnectionError('ACK lost after remote start')
            else:
                value['session'] = 'another-session'
        return value
    monkeypatch.setattr(Wire, 'receive_ready', receive)
    with pytest.raises((RuntimeError, ValueError, ConnectionError, FileNotFoundError)):
        run(case)
    assert injected and case['backend'].pending_launches.closed
    assert all(r['starts'] <= 1 for r in records(case))
    if fault not in ('lost_ack', 'bad_ack'):
        assert all(r['starts'] == 0 for r in records(case))
    if fault != 'ledger_loss':
        state = case['ledger'].read()[1]
        assert all(r['status'] == 'RESERVED' and 'consumed' not in r for r in state['tasks'].values())
    # A second run cannot replay even when a transient fault has disappeared.
    with pytest.raises((RuntimeError, ValueError, ConnectionError, FileNotFoundError)):
        run(case)


def test_wire_partial_frame_is_nonblocking_and_preserved():
    read_fd, write_fd = os.pipe()
    try:
        wire = Wire(read_fd, write_fd, timeout=.1)
        os.write(write_fd, b'{"kind":')
        assert wire.receive_ready() is None
        os.write(write_fd, b'"READY"}\n{"kind":"ACK"}\n')
        assert wire.receive_ready() == dict(kind='READY')
        assert wire.receive_ready() == dict(kind='ACK')
        assert wire.receive_ready() is None
        os.close(write_fd);write_fd = None
        with pytest.raises(ConnectionError):wire.receive_ready()
    finally:
        os.close(read_fd)
        if write_fd is not None:os.close(write_fd)


def test_cancelled_connect_cannot_send_late_prepare(tmp_path, monkeypatch):
    case = make(tmp_path)
    original = ParallelLaunches._open
    gate = threading.Event(); finished = threading.Event(); channels = []
    def connect(engine, target):
        gate.wait(10)
        channel = original(engine, target);channels.append(channel);finished.set()
        return channel
    monkeypatch.setattr(ParallelLaunches, '_open', connect)
    case['backend'].launch_timeout = .15
    with pytest.raises(TimeoutError):run(case)
    gate.set()
    assert finished.wait(10)
    deadline = time.monotonic()+10
    expected = len(case['backend'].pending_launches.pending)
    while (len(channels) < expected or any(not c.closed for c in channels)) and time.monotonic() < deadline:
        time.sleep(.01)
    assert channels and all(c.closed for c in channels)
    assert records(case) == []
    assert all(r['status'] == 'RESERVED' for r in case['ledger'].read()[1]['tasks'].values())


def test_same_gpu_preparations_are_serialized_but_workers_overlap(tmp_path, monkeypatch):
    case = make_model_case(tmp_path, parallel_launches=4, resident=True, task_count=4, checks=16,
                          slots=[dict(host='fresh', id='0', gpu_id='0'), dict(host='fresh', id='0:1', gpu_id='0')])
    original = ParallelLaunches.poll; saw_two = []
    def poll(engine, ticket):
        result = original(engine, ticket)
        assert sum(v['phase'] in ('PREPARING', 'WAIT_ACK') for v in engine.pending.values()) <= 1
        if len(engine.backend.channels) == 2:saw_two.append(True)
        return result
    monkeypatch.setattr(ParallelLaunches, 'poll', poll)
    assert run(case)['status'] == 'COMPLETE'
    assert saw_two and case['backend'].metrics['continuation_grants'] == 64


@pytest.mark.parametrize('fault', ['lost_done', 'partial_result'])
def test_parallel_partial_completion_requires_explicit_recovery(tmp_path, monkeypatch, fault):
    case = make_model_case(tmp_path, parallel_launches=2, resident=True, task_count=1)
    original = Wire.receive_ready
    def receive(wire):
        value = original(wire)
        if value and value.get('kind') == 'DONE':
            if fault == 'partial_result':
                p = case['worker']/'tasks'/(value['row']['launch']['key']+'.json')
                row = json.loads(p.read_text());row['status'] = 'UNKNOWN';row.pop('receipt');p.write_text(json.dumps(row))
            raise ConnectionError('injected completion loss')
        return value
    monkeypatch.setattr(Wire, 'receive_ready', receive)
    with pytest.raises(ConnectionError):run(case)
    monkeypatch.setattr(Wire, 'receive_ready', original)
    assert all(r['status'] == 'RESERVED' and 'consumed' not in r for r in case['ledger'].read()[1]['tasks'].values())
    # Recovery is a new controller transport; a poisoned resident connection
    # must never be promoted to healthy merely because a file says COMPLETE.
    old = case['backend']
    recovery = type(old)(old.root, old.limits, config=old.config)
    if fault == 'lost_done':
        assert recover_controlled_queue(case['queue'], case['ledger'], OWNER, 1, NEXT, recovery, simulate=True) == 2
    else:
        with pytest.raises(RuntimeError):
            recover_controlled_queue(case['queue'], case['ledger'], OWNER, 1, NEXT, recovery, simulate=True)
    assert len(records(case)) == 1 and records(case)[0]['starts'] == 1


def test_another_controller_is_rejected_during_parallel_prepare(tmp_path, monkeypatch):
    case = make(tmp_path)
    original = ParallelLaunches.poll; checked = []
    def poll(engine, ticket):
        if not checked:
            with pytest.raises(BlockingIOError):run(case)
            checked.append(True)
        return original(engine, ticket)
    monkeypatch.setattr(ParallelLaunches, 'poll', poll)
    assert run(case)['status'] == 'COMPLETE' and checked


def test_permit_rechecked_after_grant_fsync_before_go(tmp_path, monkeypatch):
    case = make(tmp_path)
    original = case['ledger']._save; expired = []
    def save(state, event):
        original(state, event)
        if event['event'] == 'authority_grant_before_go' and not expired:
            expired.append(True)
            p = case['permit'];v = json.loads(p.read_text());v['dispatch_until'] = 0;p.write_text(json.dumps(v))
    monkeypatch.setattr(case['ledger'], '_save', save)
    with pytest.raises(RuntimeError):run(case)
    assert expired and all(r['starts'] == 0 for r in records(case))
    assert all(r['status'] == 'RESERVED' for r in case['ledger'].read()[1]['tasks'].values())


def test_completion_burst_yields_to_live_channels_between_receipts(tmp_path, monkeypatch):
    case = make(tmp_path)
    backend = case['backend'];poll = backend.poll_many;validate = backend.validate
    events = [];released = [];heartbeated_waiting = []
    def poll_many(handles):
        waiting = [c for c in backend.channels.values() if c.get('done')]
        events.append('POLL');codes = poll(handles)
        if waiting:
            assert all(c.get('last_idle_probe', 0) > 0 for c in waiting)
            heartbeated_waiting.append(True)
        if not released:
            if codes.count(0) < 4:return [None]*len(codes)
            released.append(True)
        return codes
    def receipt(task):
        events.append('RECEIPT');return validate(task)
    monkeypatch.setattr(backend, 'poll_many', poll_many)
    monkeypatch.setattr(backend, 'validate', receipt)
    assert run(case)['status'] == 'COMPLETE' and released and heartbeated_waiting
    assert events.count('RECEIPT') == 8
    assert all(pair != ['RECEIPT', 'RECEIPT'] for pair in [events[i:i+2] for i in range(len(events)-1)])
