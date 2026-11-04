import json
from threading import Event, Thread

import pytest

from AlphaBrain.research.dsol.execution import phase_handoff as handoff
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.phase_budget import PhaseBudget


def test_barrier_blocks_new_reservation_without_touching_budget(tmp_path):
    a = tmp_path/'auth.json'
    a.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL', total_model_call_limit=10)))
    c = tmp_path/'campaign.json'
    c.write_text(json.dumps(dict(limits=dict(score=2, paired_execution=0, nominal_execution=0))))
    budget = PhaseBudget(tmp_path/'local', tmp_path/'mirror', a, authorization_sha256=file_digest(a))
    budget.reserve('prior', c)
    entered, finished = Event(), Event()
    rejected = []
    def reserve():
        entered.set()
        try: budget.reserve('next', c)
        except BlockingIOError: rejected.append(True)
        finally: finished.set()
    with handoff.reservation_barrier([budget.root]):
        thread = Thread(target=reserve); thread.start()
        assert entered.wait(1) and finished.wait(1)
        assert rejected == [True]
        assert not (budget.root/'next.json').exists()
        assert json.loads((budget.root/'prior.json').read_text())['status'] == 'RESERVED_BEFORE_DISPATCH'
    thread.join(2)
    assert finished.is_set()
    budget.reserve('next', c)


def test_barrier_timeout_releases_partial_locks(tmp_path):
    roots = [tmp_path/'a', tmp_path/'b']
    for p in roots:
        p.mkdir(); (p/'coordinator.lock').touch()
    with handoff.reservation_barrier([roots[1]]):
        with pytest.raises(TimeoutError):
            with handoff.reservation_barrier(roots, timeout=.05): pass
        with handoff.reservation_barrier([roots[0]], timeout=.05): pass


def test_unknown_attempt_is_not_replayed_and_completed_is_not_repeated():
    assert handoff.remaining_schedules(['a', 'b'], ['a'], ['a']) == ['b']
    with pytest.raises(ValueError, match='Unsettled'):
        handoff.remaining_schedules(['a', 'b'], ['a'], ['a', 'b'])
    with pytest.raises(ValueError): handoff.remaining_schedules(['a'], ['other'], ['other'])
    with pytest.raises(ValueError): handoff.remaining_schedules(['a', 'a'], [], [])


@pytest.mark.parametrize('fault', [None, 'pending', 'changed', 'calls', 'dispatch', 'task', 'cpu'])
def test_phase_handoff_uses_local_settled_ledger(tmp_path, monkeypatch, fault):
    counts = dict(score=2, paired_execution=3, nominal_execution=0, episodes=1)
    campaign = tmp_path/'campaign.json'
    campaign.write_text(json.dumps(dict(limits=counts)))
    row = dict(campaign=str(campaign), campaign_sha256=file_digest(campaign), reserved_model_calls=5)
    cfg = dict(output=str(tmp_path/'output'), local_root=str(tmp_path/'local'))
    root = tmp_path/'output/execution/one-real'; root.mkdir(parents=True)
    local = tmp_path/'local/one-real/center/custody'; local.mkdir(parents=True)
    (local/'ledger.json').write_text('{}')
    result = dict(status='COMPLETE', phase='one', cpu=False, dispatch_enabled=False,
                  counts=counts, actual_model_calls=5, actual_episodes=1)
    state = dict(dispatch_enabled=False, tasks={'x': dict(status='COMPLETE', consumed=counts)})
    plan = dict(campaign_sha256=row['campaign_sha256'], tasks=[dict(key='x')])
    monkeypatch.setattr(handoff.ControllerLedger, 'read', lambda self: (plan, state))
    if fault == 'changed': campaign.write_text('{}')
    if fault == 'calls': result['actual_model_calls'] = 4
    if fault == 'dispatch': state['dispatch_enabled'] = True
    if fault == 'task': state['tasks']['x']['status'] = 'RUNNING'
    if fault == 'cpu': result['cpu'] = True
    if fault != 'pending': (root/'completion.json').write_text(json.dumps(result))
    if fault == 'pending': assert handoff.completed_phase(row, cfg, 'one') is None
    elif fault is None: assert handoff.completed_phase(row, cfg, 'one')['result'] == result
    else:
        with pytest.raises(ValueError): handoff.completed_phase(row, cfg, 'one')
