"""CPU-only custody, duplicate/uncertain launch and disconnect fault tests."""
from copy import deepcopy
import json
from multiprocessing import get_context

import pytest

from AlphaBrain.research.dsol.execution.controller_ledger import (
    ControllerLedger, prepare, validate_plan,
)
from AlphaBrain.research.dsol.execution.queue import exclusive


OWNER = dict(host='center', boot_id='boot-a', session='session-a')
NEXT = dict(host='center', boot_id='boot-b', session='session-b')


@pytest.fixture
def ledger(tmp_path):
    plan = dict(campaign_sha256='f' * 64, stages=['admission', 'capacity', 'formal'],
                peers=['fresh', 'gnu', 'video'], limits=dict(calls=6, episodes=3), tasks=[
                    dict(key=str(i), stage=stage, addresses=[dict(initial=i)],
                         limits=dict(calls=2, episodes=1))
                    for i, stage in enumerate(['admission', 'capacity', 'formal'])])
    prepare(tmp_path / 'custody', plan)
    return ControllerLedger(tmp_path / 'custody')


def start(ledger):
    epoch = ledger.claim(OWNER)
    ledger.rehearse_stage(OWNER, epoch, 'admission', explicit=True)
    return epoch


def competing_claim(root, result):
    try:
        ControllerLedger(root).claim(NEXT)
        result.put('acquired')
    except (RuntimeError, BlockingIOError):
        result.put('rejected')


def test_independent_process_lock_and_persistent_owner(ledger):
    ctx = get_context('spawn')
    with exclusive(ledger.root):
        q = ctx.Queue(); p = ctx.Process(target=competing_claim, args=(ledger.root, q))
        p.start(); p.join(10)
        assert p.exitcode == 0 and q.get(timeout=1) == 'rejected'
    ledger.claim(OWNER)
    q = ctx.Queue(); p = ctx.Process(target=competing_claim, args=(ledger.root, q))
    p.start(); p.join(10)
    assert p.exitcode == 0 and q.get(timeout=1) == 'rejected'


def test_no_real_dispatch_and_no_implicit_stage(ledger):
    epoch = ledger.claim(OWNER)
    with pytest.raises(RuntimeError):
        ledger.rehearse_stage(OWNER, epoch, 'admission')
    with pytest.raises(RuntimeError):
        ledger.reserve(OWNER, epoch, '0', probe=lambda _: True)
    with pytest.raises(RuntimeError):
        ledger.reserve(OWNER, epoch, '0', probe=lambda _: True, execute=True)
    with pytest.raises(RuntimeError):
        ledger.assert_real_dispatch()
    assert ledger.read()[1]['dispatch_enabled'] is False


def test_bounded_stages_and_duplicates_survive_restart(ledger):
    epoch = start(ledger)
    with pytest.raises(RuntimeError, match='incomplete'):
        ledger.rehearse_stage(OWNER, epoch, 'capacity', explicit=True)
    peers = []
    for key, stage in enumerate(['admission', 'capacity', 'formal']):
        ledger.rehearse_stage(OWNER, epoch, stage, explicit=True)
        ledger.reserve(OWNER, epoch, str(key), probe=lambda p: peers.append(p) or True)
        reopened = ControllerLedger(ledger.root)
        with pytest.raises(RuntimeError, match='recovery'):
            reopened.reserve(OWNER, epoch, str(key), probe=lambda _: True)
        with pytest.raises(RuntimeError, match='Duplicate'):
            ledger.reserve(OWNER, epoch, str(key), probe=lambda _: True)
        with pytest.raises(ValueError, match='Budget'):
            ledger.complete_rehearsal(OWNER, epoch, str(key), dict(calls=3, episodes=1))
        ledger.complete_rehearsal(OWNER, epoch, str(key), dict(calls=2, episodes=1))
    state = ledger.read()[1]
    assert sum(t['reserved']['calls'] for t in state['tasks'].values()) == 6
    assert len(peers) == 9


@pytest.mark.parametrize('fault', [False, None, 'stale', 'exception'])
def test_disconnect_latches_until_explicit_recovery(ledger, fault):
    epoch = start(ledger)
    def probe(peer):
        if peer == 'gnu':
            if fault == 'exception':
                raise TimeoutError('injected SSH loss')
            return fault
        return True
    with pytest.raises(RuntimeError, match='Disconnected'):
        ledger.reserve(OWNER, epoch, '0', probe=probe)
    reopened = ControllerLedger(ledger.root)
    with pytest.raises(RuntimeError):
        reopened.reserve(OWNER, epoch, '0', probe=lambda _: True)
    with pytest.raises(RuntimeError):
        reopened.rehearse_stage(OWNER, epoch, 'admission', explicit=True)
    with pytest.raises(RuntimeError):
        reopened.recover(OWNER, epoch, NEXT, reconcile=lambda _: '')
    new_epoch = reopened.recover(OWNER, epoch, NEXT, reconcile=lambda _: 'a' * 64)
    assert new_epoch == 2
    with pytest.raises(RuntimeError, match='fencing'):
        reopened.rehearse_stage(OWNER, epoch, 'admission', explicit=True)
    reopened.rehearse_stage(NEXT, new_epoch, 'admission', explicit=True)
    reopened.reserve(NEXT, new_epoch, '0', probe=lambda _: True)


def test_recovery_never_refunds_or_retries_uncertain_attempt(ledger):
    epoch = start(ledger)
    ledger.reserve(OWNER, epoch, '0', probe=lambda _: True)
    new_epoch = ledger.recover(OWNER, epoch, NEXT, reconcile=lambda _: 'a' * 64)
    ledger.rehearse_stage(NEXT, new_epoch, 'admission', explicit=True)
    with pytest.raises(RuntimeError, match='Duplicate'):
        ledger.reserve(NEXT, new_epoch, '0', probe=lambda _: True)
    with pytest.raises(RuntimeError):
        ledger.complete_rehearsal(NEXT, new_epoch, '0', dict(calls=0, episodes=0))
    assert ledger.read()[1]['tasks']['0']['reserved']['calls'] == 2


def test_plan_overlap_budget_tamper_and_missing_storage(ledger):
    plan, _ = ledger.read()
    changed = deepcopy(plan); changed['tasks'][1]['addresses'] = changed['tasks'][0]['addresses']
    with pytest.raises(ValueError, match='Overlapping'): validate_plan(changed)
    changed = deepcopy(plan); changed['limits']['calls'] += 1
    with pytest.raises(ValueError, match='coverage'): validate_plan(changed)
    epoch = start(ledger)
    path = ledger.root / 'plan.json'; original = path.read_text()
    changed = deepcopy(plan); changed['campaign_sha256'] = 'e' * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Changed plan'): ledger.read()
    path.write_text(original)
    path.rename(ledger.root / 'plan-hidden.json')
    with pytest.raises(FileNotFoundError):
        ledger.reserve(OWNER, epoch, '0', probe=lambda _: True)
    (ledger.root / 'plan-hidden.json').rename(path)
    with pytest.raises(RuntimeError, match='fault'):
        ledger.reserve(OWNER, epoch, '0', probe=lambda _: True)


def test_write_failure_prevents_reservation_return(ledger, monkeypatch):
    epoch = start(ledger)
    from AlphaBrain.research.dsol.execution import controller_ledger as module
    original = module._durable_write
    def unavailable(*_): raise OSError('injected shared storage loss')
    monkeypatch.setattr(module, '_durable_write', unavailable)
    with pytest.raises(OSError): ledger.reserve(OWNER, epoch, '0', probe=lambda _: True)
    monkeypatch.setattr(module, '_durable_write', original)
    with pytest.raises(RuntimeError, match='fault'):
        ledger.reserve(OWNER, epoch, '0', probe=lambda _: True)
