from copy import deepcopy
import json
import pytest
from AlphaBrain.research.dsol.execution.node_health import NodeHealth,require_fresh_health


def snapshot():
    expected=dict(hostname='borrowed',boot_id='new',devices=['GPU-1'],keeper=dict(pid=3,start_ticks=4,executable_sha256='a'*64))
    return expected,dict(expected,status='PASS',owned_root_accessible=True,shared_root_accessible=True)


def test_disconnect_is_latched_and_changed_birth_never_auto_recovers():
    expected,s=snapshot();monitor=NodeHealth(expected,'observer')
    assert monitor.observe(s,now=10)['status']=='HEALTHY'
    assert monitor.observe({'status':'FAILED'},now=11)['status']=='HOLD'
    assert monitor.observe(s,now=12)['status']=='HOLD'
    another=NodeHealth(expected,'new-observer');bad=deepcopy(s);bad['keeper']['start_ticks']=999
    assert another.observe(bad,now=10)['reason']=='keeper_identity_changed'


@pytest.mark.parametrize('change',['stale','future','observer','hold','missing'])
def test_local_guard_refuses_untrusted_or_stale_monitor(tmp_path,change):
    expected,s=snapshot();state=NodeHealth(expected,'observer').observe(s,now=10)
    p=tmp_path/'local.json';cfg=dict(path=str(p),observer_id='observer',max_age_seconds=30)
    if change=='stale':state['checked_monotonic']=-30
    if change=='future':state['checked_monotonic']=40
    if change=='observer':state['observer_id']='replaced'
    if change=='hold':state['status']='HOLD'
    if change!='missing':p.write_text(json.dumps(state))
    with pytest.raises((ConnectionError,OSError)):require_fresh_health(cfg,now=20)


def test_fresh_local_health_augments_actual_authority_probe(tmp_path):
    from AlphaBrain.research.dsol.execution.model_authority import ModelAuthorityBackend
    expected,s=snapshot();p=tmp_path/'health.json';p.write_text(json.dumps(NodeHealth(expected,'o').observe(s)))
    backend=ModelAuthorityBackend.__new__(ModelAuthorityBackend);backend.production=True
    backend.config=dict(node_health={'cy':dict(path=str(p),observer_id='o',max_age_seconds=30)})
    backend._node_health('cy')
    value=json.loads(p.read_text());value['status']='HOLD';p.write_text(json.dumps(value))
    with pytest.raises(ConnectionError):backend._node_health('cy')
def test_trusted_sampling_race_is_bounded_and_identity_failure_never_retried(tmp_path):
    from AlphaBrain.research.dsol.execution.node_health import trusted_keeper_snapshot
    from AlphaBrain.research.dsol.execution.gpu_guard import TransientMonitorError,MonitorError
    c=dict(gpu_guard={},keeper_gpu_uuid='gpu',owned_root=str(tmp_path),shared_root=str(tmp_path))
    snapshot=dict(hostname='host',boot_id='boot',gpus=[dict(uuid='gpu')])
    calls=[]
    def collector(*args,**kwargs):
        calls.append(1)
        if len(calls)<3:raise TransientMonitorError('membership race')
        return snapshot
    result=trusted_keeper_snapshot(c,collector=collector,verifier=lambda *_:dict(supervisor_state='PROTECTED'),sleep=lambda _:None)
    assert result['status']=='PASS' and result['attempts']==3 and len(calls)==3
    def identity(*args,**kwargs):raise MonitorError('wrong executable')
    result=trusted_keeper_snapshot(c,collector=identity,sleep=lambda _:None)
    assert result['status']=='FAILED' and result['attempts']==1
    def unstable(*args,**kwargs):raise TransientMonitorError('race')
    result=trusted_keeper_snapshot(c,collector=unstable,sleep=lambda _:None)
    assert result['status']=='FAILED' and result['attempts']==3



def test_transient_missing_proc_fd_is_bounded_not_a_healthy_cache(tmp_path):
    from AlphaBrain.research.dsol.execution.node_health import trusted_keeper_snapshot
    calls=[]
    def vanished(*args,**kwargs):
        calls.append(1);raise FileNotFoundError('proc fd closed during fresh sampling')
    c=dict(gpu_guard={},keeper_gpu_uuid='gpu',owned_root=str(tmp_path),shared_root=str(tmp_path))
    result=trusted_keeper_snapshot(c,collector=vanished,sleep=lambda _:None)
    assert result['status']=='FAILED' and result['attempts']==3 and len(calls)==3


def test_live_verified_supervisor_can_manage_anchor_birth_without_changing_trust():
    expected,s=snapshot();expected['keeper']=dict(supervisor_pid=7,supervisor_start_ticks=9,executable_sha256='a'*64)
    s['keeper']=dict(expected['keeper'],pid=33,start_ticks=44,gpu_uuid='GPU-1')
    monitor=NodeHealth(expected,'explicit-new-managed-supervisor-observer')
    assert monitor.observe(s,now=10)['status']=='HEALTHY'
    changed=deepcopy(s);changed['keeper'].update(pid=55,start_ticks=66)
    assert monitor.observe(changed,now=11)['status']=='HEALTHY'
    changed['keeper']['supervisor_start_ticks']=99
    assert monitor.observe(changed,now=12)['status']=='HOLD'
