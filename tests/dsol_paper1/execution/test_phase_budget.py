from pathlib import Path
import json
import pytest
from AlphaBrain.research.dsol.execution.phase_budget import PhaseBudget
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.fleet_slots import FleetSlots


def test_unknown_reservation_never_refunds_or_replays(tmp_path):
    a=tmp_path/'a.json';a.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL',total_model_call_limit=10)))
    c=tmp_path/'c.json';c.write_text(json.dumps(dict(limits=dict(score=6,paired_execution=0,nominal_execution=0))))
    b=PhaseBudget(tmp_path/'local',tmp_path/'mirror',a,authorization_sha256=file_digest(a));b.reserve('one',c)
    with pytest.raises(ValueError):b.reserve('one',c)
    with pytest.raises(ValueError):b.reserve('two',c)
    assert b.status()['held_unsettled']==6
    b.settle('one',dict(status='COMPLETE',actual_model_calls=2))
    assert b.status()['reserved']==6 and b.status()['known_consumed']==2
    with pytest.raises(ValueError):b.reserve('two',c)


def test_mirror_failure_keeps_local_reservation(tmp_path,monkeypatch):
    import AlphaBrain.research.dsol.execution.phase_budget as module
    a=tmp_path/'a.json';a.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL',total_model_call_limit=10)))
    c=tmp_path/'c.json';c.write_text(json.dumps(dict(limits=dict(score=6,paired_execution=0,nominal_execution=0))))
    b=PhaseBudget(tmp_path/'local',tmp_path/'mirror',a,authorization_sha256=file_digest(a));original=module._durable_write
    def write(p,v):
        if p.parent==tmp_path/'mirror':raise OSError('disconnected')
        original(p,v)
    monkeypatch.setattr(module,'_durable_write',write)
    with pytest.raises(OSError):b.reserve('one',c)
    assert b.status()['held_unsettled']==6
    with pytest.raises(ValueError):b.reserve('one',c)


def test_failed_phase_quarantines_slots_until_audit():
    pool=FleetSlots(dict(a=[0,1,2,3],b=[0,1,2,3]))
    with pytest.raises(ConnectionError):
        with pool.lease('one',['a']):raise ConnectionError('unknown processes')
    assert pool.available['a']==set() and pool.held['one']['gpu_ids']==[0,1,2,3]
    with pool.lease('two',['b']) as result:assert result==('b',[0,1,2,3])
    assert len(pool.available['b'])==4


def test_quarantined_only_host_fails_without_blocking_others():
    pool=FleetSlots(dict(a=[0,1,2,3],b=[0,1,2,3]))
    with pytest.raises(RuntimeError):
        with pool.lease('failed',['a']):raise RuntimeError('unknown')
    with pytest.raises(RuntimeError,match='quarantined'):
        with pool.lease('blocked',['a']):pass
    with pool.lease('healthy',['b']) as (host,ids):assert host=='b' and len(ids)==4
    assert len(pool.snapshot()['held'])==1


def test_child_cannot_spend_unreserved_parent_budget(tmp_path):
    a=tmp_path/'a.json';a.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL',total_model_call_limit=10)))
    c=tmp_path/'c.json';c.write_text(json.dumps(dict(limits=dict(score=6,paired_execution=0,nominal_execution=0))))
    parent=PhaseBudget(tmp_path/'parent',tmp_path/'parent-mirror',a,authorization_sha256=file_digest(a));parent.reserve('delegate',c)
    allocation=dict(path=str(tmp_path/'parent'/'delegate.json'),sha256=file_digest(tmp_path/'parent'/'delegate.json'))
    child=PhaseBudget(tmp_path/'child',tmp_path/'child-mirror',a,authorization_sha256=file_digest(a),allocation=allocation)
    child.reserve('one',c)
    with pytest.raises(ValueError):child.reserve('two',c)
    assert child.status()['hard_limit']==6 and parent.status()['held_unsettled']==6


def test_resource_grants_never_reactivate_leased_or_previously_granted_cards():
    pool=FleetSlots(dict(a=[0,1,2,3]),initial_capacity=dict(a=[0]))
    pool.grant('a',[1],evidence_sha256='a'*64)
    with pool.lease('one',['a'],count=2):
        with pytest.raises(ValueError):pool.grant('a',[1],evidence_sha256='a'*64)
    pool.grant('a',[2,3],evidence_sha256='b'*64)
    assert len(pool.snapshot()['available']['a'])==4


def test_explicit_surplus_cannot_double_spend_or_release_unknown(tmp_path):
    a=tmp_path/'auth.json';a.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL',total_model_call_limit=20)))
    c=tmp_path/'phase.json';c.write_text(json.dumps(dict(limits=dict(score=10,paired_execution=0,nominal_execution=0))))
    recovery=tmp_path/'recovery.json';recovery.write_text(json.dumps(dict(limits=dict(score=6,paired_execution=0,nominal_execution=0))))
    b=PhaseBudget(tmp_path/'parent',tmp_path/'mirror',a,authorization_sha256=file_digest(a))
    b.reserve('complete',c);b.reserve('unknown',c)
    with pytest.raises(ValueError,match='complete settlements'):b.delegate_settled_surplus('invalid',recovery,['unknown'])
    b.settle('complete',dict(status='COMPLETE',actual_model_calls=4))
    before=file_digest(b.root/'complete.json');unknown=file_digest(b.root/'unknown.json')
    allocation=b.delegate_settled_surplus('repair',recovery,['complete'])
    row=json.loads(Path(allocation['path']).read_text())
    assert row['combined_maximum_consumption']==20
    assert file_digest(b.root/'complete.json')==before and file_digest(b.root/'unknown.json')==unknown
    assert b.status()['reserved']==20 and b.status()['held_unsettled']==10
    child=PhaseBudget(tmp_path/'child',tmp_path/'child-mirror',a,authorization_sha256=file_digest(a),allocation=allocation)
    child.reserve('repair',recovery)
    with pytest.raises(ValueError):child.reserve('again',recovery)
    with pytest.raises(ValueError,match='already delegated'):b.delegate_settled_surplus('second',recovery,['complete'])
    with pytest.raises(ValueError,match='already exists'):b.delegate_settled_surplus('repair',recovery,['complete'])


def test_surplus_checks_evidence_and_never_obstructs_parent_lock(tmp_path):
    from AlphaBrain.research.dsol.execution.queue import exclusive
    a=tmp_path/'auth.json';a.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL',total_model_call_limit=10)))
    c=tmp_path/'phase.json';c.write_text(json.dumps(dict(limits=dict(score=10,paired_execution=0,nominal_execution=0))))
    recovery=tmp_path/'recovery.json';recovery.write_text(json.dumps(dict(limits=dict(score=4,paired_execution=0,nominal_execution=0))))
    b=PhaseBudget(tmp_path/'parent',tmp_path/'mirror',a,authorization_sha256=file_digest(a))
    b.reserve('complete',c);b.settle('complete',dict(status='COMPLETE',actual_model_calls=7))
    with pytest.raises(ValueError,match='Insufficient'):b.delegate_settled_surplus('too-much',recovery,['complete'])
    recovery.write_text(json.dumps(dict(limits=dict(score=3,paired_execution=0,nominal_execution=0))))
    with exclusive(b.root):allocation=b.delegate_settled_surplus('exact',recovery,['complete'])
    assert json.loads(Path(allocation['path']).read_text())['combined_maximum_consumption']==10
