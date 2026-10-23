import json
from contextlib import contextmanager
import pytest
from AlphaBrain.research.dsol.execution.campaign_control import CampaignGuard
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.phase_timing import PhaseTiming


def test_permit_deadline_peer_failure_and_finished_hosts(tmp_path):
    manifest=tmp_path/'campaign.json';control=tmp_path/'control';control.mkdir()
    manifest.write_text(json.dumps(dict(control=str(control),hosts=['fresh','gnu','video'],restart_deadline_unix=30000)))
    clock=[100.];g=CampaignGuard(manifest,'fresh','admission',clock=lambda:clock[0])
    with pytest.raises(FileNotFoundError):g.check()
    permit=dict(authorization='EXPLICIT_CAMPAIGN_EXECUTION',manifest_sha256=file_digest(manifest),stage='admission',issued_at=90,dispatch_until=1000,finish_until=2000)
    (control/'permit.json').write_text(json.dumps(permit));assert g.check(dispatch=True)
    clock[0]=300
    with pytest.raises(FileNotFoundError):g.check()
    for host in ['fresh','gnu','video']:CampaignGuard(manifest,host,'admission',clock=lambda:clock[0]).heartbeat()
    assert g.check()
    clock[0]=370
    with pytest.raises(RuntimeError,match='heartbeat'):g.check()
    for host in ['gnu','video']:
        (control/f'done-{host}-admission.json').write_text(json.dumps(dict(status='COMPLETE',manifest_sha256=g.digest,stage='admission')))
    g.heartbeat();assert g.check()
    clock[0]=1000;g.heartbeat();assert not g.check(dispatch=True) and g.check()
    g.fail('injected peer failure')
    with pytest.raises(RuntimeError,match='failed'):g.check()


def test_reserve_and_changed_permit_fail_closed(tmp_path):
    manifest=tmp_path/'m.json';manifest.write_text(json.dumps(dict(control=str(tmp_path),hosts=['fresh'],restart_deadline_unix=30000)))
    g=CampaignGuard(manifest,'fresh','formal',clock=lambda:100)
    p=dict(authorization='EXPLICIT_CAMPAIGN_EXECUTION',manifest_sha256=g.digest,stage='formal',issued_at=0,dispatch_until=9000,finish_until=10000)
    (tmp_path/'permit.json').write_text(json.dumps(p))
    with pytest.raises(RuntimeError,match='reserve'):g.check()
    p.update(dispatch_until=1000,finish_until=2000,manifest_sha256='bad');(tmp_path/'permit.json').write_text(json.dumps(p))
    with pytest.raises(RuntimeError):g.check()


def test_phase_timing_closes_on_exception():
    ticks=iter([0,2,3,7,11,12]);timing=PhaseTiming(clock=lambda:next(ticks))
    with timing.measure('ledger'):pass
    @contextmanager
    def context():yield 'value'
    with pytest.raises(ValueError):
        with timing.context(context(),'simulator') as value:
            assert value=='value';raise ValueError('test')
    assert timing.rows['ledger']==dict(seconds=2,count=1)
    assert timing.rows['simulator_start']['seconds']==4
    assert timing.rows['simulator_close']['seconds']==1
