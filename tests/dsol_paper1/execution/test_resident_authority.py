"""Actual standard queue/ledger/remote agent, with CPU-only resident resources."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from AlphaBrain.research.dsol.execution import authority_wire
from AlphaBrain.research.dsol.execution.controlled_queue import run_controlled_queue
from AlphaBrain.research.dsol.execution.resident_runtime import ResidentRuntimeCache
from tests.dsol_paper1.helpers.controller_cases import OWNER
from tests.dsol_paper1.helpers.model_cases import make_model_case


def run(case):
    return run_controlled_queue(case['queue'],case['ledger'],OWNER,1,case['permit'],case['backend'],
                                simulate=True,poll_seconds=.01,timeout_seconds=30)


def test_resident_actual_queue_reuses_process_and_resource_with_separate_budgets(tmp_path):
    case=make_model_case(tmp_path,resident=True)
    assert run(case)['status']=='COMPLETE'
    rows=[json.loads(p.read_text()) for p in (case['worker']/'tasks').glob('*.json')]
    assert len(rows)==3 and len({(r['process']['pid'],r['process']['start_ticks']) for r in rows})==1
    assert all(r['status']=='COMPLETE' and r['starts']==1 for r in rows)
    assert sorted(r['resident_cache']['reuses'] for r in rows)==[0,1,2]
    assert all(r['resident_cache']['loads']==1 for r in rows)
    assert case['backend'].metrics['worker_processes']==1
    assert case['backend'].metrics['resident_reuses']==2
    assert case['backend'].metrics['resident_done_alive_checks']==3
    assert case['backend'].metrics['completion_inspections']==3
    _,state=case['ledger'].read()
    assert all(r['status']=='COMPLETE' and r['consumed']==r['reserved'] for r in state['tasks'].values())
    assert state['dispatch_enabled'] is False
    assert all(json.loads(p.read_text())['status']=='CLOSED' for p in (case['worker']/'residents').glob('*.json'))


def test_live_done_does_not_skip_fresh_receipt_validation(tmp_path,monkeypatch):
    case=make_model_case(tmp_path,resident=True)
    alive=case['backend'].alive
    def tamper_after_done(handle):
        value=alive(handle)
        if case['backend'].channels.get(handle['key'],{}).get('done'):
            path=case['worker']/'tasks'/(handle['key']+'.json')
            row=json.loads(path.read_text())
            row['receipt']['budget_consumed']['calls']=row['launch']['limits']['calls']+1
            path.write_text(json.dumps(row))
        return value
    monkeypatch.setattr(case['backend'],'alive',tamper_after_done)
    with pytest.raises(ValueError,match='budget exceeded'):run(case)
    state=case['ledger'].read()[1]
    assert all(r['status']=='RESERVED' and 'consumed' not in r for r in state['tasks'].values())
    assert case['backend'].metrics['completion_inspections']>=1


def test_nonresident_and_missing_done_still_inspect(tmp_path,monkeypatch):
    case=make_model_case(tmp_path,resident=True);backend=case['backend'];seen=[]
    monkeypatch.setattr(backend,'_inspect',lambda host,key:seen.append((host,key)))
    handle=dict(host='fresh',key='synthetic',session='session')
    backend.channels[handle['key']]=dict(handle=handle,done=False)
    assert backend.alive(handle) is False and len(seen)==1
    backend.channels[handle['key']]['done']=True
    with pytest.raises(RuntimeError,match='handle'):backend.alive(dict(handle,session='other'))
    assert backend.alive(handle) is False and len(seen)==1
    backend.resident=False
    assert backend.alive(handle) is False and len(seen)==2
    backend.channels.clear()  # Synthetic entry owns no process or transport.


@pytest.mark.parametrize('fault',['second_ack_lost','second_done_lost','second_check_epoch','idle_disconnect'])
def test_resident_partial_completion_does_not_replay_or_refund(tmp_path,monkeypatch,fault):
    case=make_model_case(tmp_path,resident=True)
    receive=authority_wire.Wire.receive
    seen={'ACK':0,'DONE':0,'CHECK':0}
    def injected(wire):
        value=receive(wire);kind=value.get('kind')
        if kind in seen:seen[kind]+=1
        if fault=='second_ack_lost' and kind=='ACK' and seen[kind]==2:raise ConnectionError('lost second ACK')
        if fault=='second_done_lost' and kind=='DONE' and seen[kind]==2:raise ConnectionError('lost second completion')
        if fault=='second_check_epoch' and kind=='CHECK' and seen['ACK']==2:
            permit=json.loads(case['permit'].read_text());permit['epoch']=0
            case['permit'].write_text(json.dumps(permit))
        return value
    monkeypatch.setattr(authority_wire.Wire,'receive',injected)
    if fault=='idle_disconnect':
        original=case['backend'].start_authorized
        def start(*args,**kwargs):
            for client in case['backend'].idle_channels.values():client['channel'].close()
            return original(*args,**kwargs)
        monkeypatch.setattr(case['backend'],'start_authorized',start)
    with pytest.raises((RuntimeError,OSError)):run(case)
    _,state=case['ledger'].read()
    assert sum(r['status']=='COMPLETE' for r in state['tasks'].values())==1
    unsettled=[r for r in state['tasks'].values() if r['status']!='COMPLETE']
    assert len(unsettled)==1 and unsettled[0]['status']=='RESERVED'
    assert 'consumed' not in unsettled[0]
    assert len(list((case['worker']/'tasks').glob('*.json')))<=2
    with pytest.raises(RuntimeError):run(case)


def test_runtime_lease_rejects_identity_changes_and_failure_reuse():
    cache=ResidentRuntimeCache();created=[]
    def factory():
        value=object();created.append(value);return value
    with cache.lease('frozen',factory,lambda r:None) as (one,reused):assert not reused
    with cache.lease('frozen',factory,lambda r:None) as (two,reused):assert reused and one is two
    assert len(created)==1
    with pytest.raises(RuntimeError,match='identity'):
        with cache.lease('other',factory,lambda r:None):pass
    with pytest.raises(RuntimeError,match='recovery'):
        with cache.lease('frozen',factory,lambda r:None):pass


def test_runtime_partial_inference_poison_and_parallel_lease():
    cache=ResidentRuntimeCache()
    with pytest.raises(RuntimeError,match='partial inference'):
        with cache.lease('frozen',object,lambda r:None):
            with pytest.raises(RuntimeError,match='Concurrent'):
                with cache.lease('frozen',object,lambda r:None):pass
            raise RuntimeError('partial inference')
    assert cache.failed
    with pytest.raises(RuntimeError,match='recovery'):
        with cache.lease('frozen',object,lambda r:None):pass
