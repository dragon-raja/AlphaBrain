"""Real CPU transport: amortize inspection setup without cached acceptance."""
import fcntl
import json

import pytest

from AlphaBrain.research.dsol.execution import authority_wire
from AlphaBrain.research.dsol.execution.controlled_queue import run_controlled_queue
from tests.dsol_paper1.helpers.controller_cases import OWNER
from tests.dsol_paper1.helpers.model_cases import make_model_case


def run(case):
    return run_controlled_queue(case['queue'], case['ledger'], OWNER, 1,
                                case['permit'], case['backend'], simulate=True,
                                poll_seconds=.01, timeout_seconds=30)


def test_one_inspection_process_for_many_fresh_receipts(tmp_path, monkeypatch):
    case = make_model_case(tmp_path, resident=True, persistent_inspection=True)
    backend = case['backend']; original = backend._inspect_persistent; births = []
    def inspect(host, key):
        row = original(host, key)
        births.append(backend.inspection_channels[host].process.pid)
        backend.inspection_last_probe[host] = 0.
        backend.probe_inspections()
        return row
    monkeypatch.setattr(backend, '_inspect_persistent', inspect)
    assert run(case)['status'] == 'COMPLETE'
    assert len(births) == 3 and len(set(births)) == 1
    assert backend.metrics['inspection_connections'] == 1
    assert backend.metrics['completion_inspections'] == 3
    assert not backend.inspection_channels
    assert all(r['status'] == 'COMPLETE' for r in case['ledger'].read()[1]['tasks'].values())


@pytest.mark.parametrize('fault', ['disconnect', 'wrong_key', 'wrong_session', 'changed_receipt'])
def test_failed_second_inspection_keeps_budget_and_never_replays(tmp_path, monkeypatch, fault):
    case = make_model_case(tmp_path, resident=True, persistent_inspection=True)
    backend = case['backend']; original = backend._inspect; calls = []
    receive = authority_wire.Wire.receive
    def corrupt(wire):
        value = receive(wire)
        if value.get('kind') == 'STATE' and len(calls) == 2:
            if fault == 'wrong_key': value['key'] = 'foreign-key'
            if fault == 'wrong_session': value['session'] = 'old-session'
        return value
    monkeypatch.setattr(authority_wire.Wire, 'receive', corrupt)
    def inspect(host, key):
        calls.append(key)
        if len(calls) == 2:
            if fault == 'disconnect': backend.inspection_channels[host].close()
            if fault == 'changed_receipt':
                path = case['worker'] / 'tasks' / (key + '.json')
                row = json.loads(path.read_text())
                row['receipt']['budget_consumed']['calls'] += 1
                path.write_text(json.dumps(row))
        return original(host, key)
    monkeypatch.setattr(backend, '_inspect', inspect)
    with pytest.raises((RuntimeError, OSError, ValueError)): run(case)
    state = case['ledger'].read()[1]
    assert not state['dispatch_enabled']
    assert sum(r['status'] == 'COMPLETE' for r in state['tasks'].values()) == 1
    held = [r for r in state['tasks'].values() if r['status'] != 'COMPLETE']
    assert len(held) == 1 and held[0]['status'] == 'RESERVED' and 'consumed' not in held[0]
    assert backend.metrics['inspection_connections'] == 1
    with pytest.raises(RuntimeError): run(case)
    if fault != 'changed_receipt':
        with pytest.raises(ConnectionError, match='explicit recovery'):
            backend._inspect_persistent('fresh', calls[-1])
    assert len(list((case['worker'] / 'tasks').glob('*.json'))) == 2


def test_inspection_channel_rejects_launch_messages(tmp_path):
    case = make_model_case(tmp_path, persistent_inspection=True)
    backend = case['backend']; target = backend._target('fresh')
    with backend.channel(target) as channel:
        backend._hello(channel, target)
        session = channel.hello['session']
        channel.wire.send(dict(kind='INSPECT_SESSION', session=session))
        assert channel.wire.receive() == dict(kind='INSPECTION_READY', session=session)
        channel.wire.send(dict(kind='PREPARE', session=session, key=next(iter(case['binding']['tasks']))))
        with pytest.raises((RuntimeError, OSError)): channel.wire.receive()
    assert not list((case['worker'] / 'tasks').glob('*.json'))
    assert not case['ledger'].read()[1]['tasks']


def test_inspection_rechecks_task_lock(tmp_path):
    case = make_model_case(tmp_path, persistent_inspection=True)
    backend = case['backend']; key = next(iter(case['binding']['tasks']))
    try:
        with (case['worker'] / 'locks' / (key + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match='task lock'):
                backend._inspect_persistent('fresh', key)
        with pytest.raises(ConnectionError, match='explicit recovery'):
            backend._inspect_persistent('fresh', key)
    finally:
        backend.close_channels()


def test_inspection_heartbeat_loss_does_not_reconnect(tmp_path):
    case = make_model_case(tmp_path, persistent_inspection=True)
    backend = case['backend']; key = next(iter(case['binding']['tasks']))
    try:
        assert backend._inspect_persistent('fresh', key) is None
        backend.inspection_channels['fresh'].close()
        backend.inspection_last_probe['fresh'] = 0.
        with pytest.raises((RuntimeError, OSError)): backend.probe_inspections()
        with pytest.raises(ConnectionError, match='explicit recovery'):
            backend._inspect_persistent('fresh', key)
        assert backend.metrics['inspection_connections'] == 1
    finally:
        backend.close_channels()
