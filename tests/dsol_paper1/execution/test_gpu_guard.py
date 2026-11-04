"""CPU-only ownership, restart, monitoring and memory admission regressions."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from AlphaBrain.research.dsol.execution.gpu_guard import (
    MonitorError, TransientMonitorError, classify, preflight, process, verify_status,
)


@pytest.fixture
def case():
    release='/trusted/releases/v6'
    files={'anchor':'anchor-hash','supervisor.py':'supervisor-hash'}
    devices=[dict(index=0,uuid='GPU-0'),dict(index=3,uuid='GPU-3')]
    config=dict(hostname='node',profile='fresh',status_path='/state/status.json',devices=devices,
        max_status_age_seconds=15,attempts=3,retry_delay_seconds=0,
        worker_peak_mib=13500,safety_margin_mib=2048,minimum_free_mib=13500,
        supervisor_executable=dict(path='/python',sha256='python-hash'),
        trusted_releases={release:dict(files=files)})
    sup=dict(pid=10,state='S',ppid=9,pgid=9,sid=9,start_ticks=10000,
        executable='/python',executable_sha256='python-hash',argv=['python',release+'/supervisor.py','run','fresh'])
    worker=dict(pid=11,state='S',ppid=10,pgid=9,sid=9,start_ticks=20000,
        executable=release+'/anchor',executable_sha256='anchor-hash',argv=[release+'/anchor','--device','3'])
    status=dict(profile='fresh',pid=10,checked_at=19999,boot_id='boot',release=release,
        process_identity={k:sup[k] for k in ('pid','start_ticks','pgid','sid')},
        state='PROTECTED_OWN_WORKER',job_coordination='coexist',worker_pid=11,worker_gpu=3,gpus=devices)
    snapshot=dict(hostname='node',boot_id='boot',profile_file='fresh',status=status,supervisor=sup,worker=worker,
        now=20000,boottime=10000,clock_ticks=100,release_hashes=files,supervisor_children=[11],
        supervisor_lock_held=True,apps=[dict(gpu_uuid='GPU-3',pid=11,used_mib=900)],processes={11:worker},
        gpus=[dict(g,free_mib=31000,used_mib=900,total_mib=32607) for g in devices])
    return config,snapshot


def test_current_worker_is_known_on_any_gpu(case):
    c,s=case
    result=classify(c,s)
    assert [r['category'] for r in result['rows']]==['AVAILABLE','KNOWN_KEEPALIVE']
    assert result['rows'][1]['required_free_mib']==15548
    assert result['keepalive']['start_ticks']==20000


def test_live_low_utilization_worker_still_requires_identity_and_memory(case):
    c,s=case;s['status'].update(state='WORKER_THRESHOLD_LOW',continuous_worker=True)
    assert classify(c,s)['keepalive']['supervisor_state']=='WORKER_THRESHOLD_LOW'
    s['gpus'][1]['free_mib']=15547
    assert classify(c,s)['rows'][1]['category']=='INSUFFICIENT_MEMORY'
    s['worker']['executable_sha256']='impostor'
    assert preflight(c,collector=lambda _:s)['status']=='MONITOR_ERROR'


@pytest.mark.parametrize('state',['PAUSED','ERROR','WORKER_STARTING','YIELD_MEMORY_PRESSURE','unknown'])
def test_non_running_or_unready_states_remain_blocked(case,state):
    c,s=case;s['status'].update(state=state,continuous_worker=True)
    result=preflight(c,collector=lambda _:s,sleep=lambda _:None)
    assert result['status']=='MONITOR_ERROR' and state in result['error']


def test_low_threshold_requires_continuous_worker_contract(case):
    c,s=case;s['status']['state']='WORKER_THRESHOLD_LOW'
    assert preflight(c,collector=lambda _:s,sleep=lambda _:None)['status']=='MONITOR_ERROR'


def test_explicitly_trusted_old_unified_release_is_supported(case):
    c,s=case
    old='/trusted/releases/v5';new=s['status']['release']
    c['trusted_releases'][old]=c['trusted_releases'].pop(new)
    s['status']['release']=old
    s['supervisor']['argv'][1]=old+'/supervisor.py'
    s['worker']['executable']=old+'/anchor';s['worker']['argv'][0]=old+'/anchor'
    assert classify(c,s)['status']=='PASS'


def test_old_python_name_is_not_a_whitelist(case):
    c,s=case;s['worker']['executable']='/python'
    s['worker']['argv']=['python','/gpu_compute_keepalive/gpu_compute_keepalive.py','--device','3']
    with pytest.raises(MonitorError,match='trusted release anchor'):
        verify_status(c,s)


@pytest.mark.parametrize('field,value',[
    ('hostname','other'),('profile_file','other'),('boot_id','other'),('supervisor_lock_held',False)])
def test_bad_host_or_supervisor_proof_is_blocked(case,field,value):
    c,s=case;s[field]=value
    assert preflight(c,collector=lambda _:s,sleep=lambda _:None)['status']=='MONITOR_ERROR'


@pytest.mark.parametrize('mutation',[
    lambda s:s['status'].update(profile='other'),
    lambda s:s['status']['process_identity'].update(start_ticks=10001),
    lambda s:s['worker'].update(ppid=12),
    lambda s:s['worker'].update(executable='/fake/anchor'),
    lambda s:s['worker'].update(executable_sha256='replaced'),
    lambda s:s['supervisor'].update(executable_sha256='replaced'),
    lambda s:s['supervisor']['argv'].append('--fake'),
    lambda s:s.update(supervisor_children=[]),
    lambda s:s['status']['gpus'].__setitem__(1,dict(index=3,uuid='GPU-other')),
    lambda s:s['worker']['argv'].__setitem__(-1,'0'),
    lambda s:s['status'].update(worker_start_ticks=100),
])
def test_identity_mismatches_fail_closed(case,mutation):
    c,s=case;s=deepcopy(s);mutation(s)
    result=preflight(c,collector=lambda _:s,sleep=lambda _:None)
    assert result['status']=='MONITOR_ERROR'
    assert all(r['category']=='MONITOR_ERROR' and not r['allowed'] for r in result['rows'])


def test_worker_pid_reused_after_status_publication(case):
    c,s=case;s['worker']['start_ticks']=1000001
    with pytest.raises(TransientMonitorError,match='birth'):
        verify_status(c,s)


def test_stale_status_retries_exactly_three_times(case):
    c,s=case;s['status']['checked_at']=19900
    attempts=[];sleeps=[]
    def collect(_):attempts.append(1);return s
    result=preflight(c,collector=collect,sleep=sleeps.append)
    assert result['status']=='MONITOR_ERROR' and len(attempts)==3 and len(sleeps)==2


def test_worker_restart_requires_new_fresh_consistent_status(case):
    c,s=case;fresh=deepcopy(s)
    fresh['status'].update(worker_pid=12,checked_at=20000)
    fresh['worker'].update(pid=12,start_ticks=999900)
    fresh['supervisor_children']=[12]
    fresh['apps']=[dict(gpu_uuid='GPU-3',pid=12,used_mib=900)]
    fresh['processes']={12:fresh['worker']}
    calls=[]
    def collect(_):
        calls.append(1)
        if len(calls)==1:raise TransientMonitorError('worker restarting')
        return fresh
    result=preflight(c,collector=collect,sleep=lambda _:None)
    assert result['status']=='PASS' and result['attempts']==2 and result['keepalive']['pid']==12


@pytest.mark.parametrize('stable_after', [2, 3, None])
def test_membership_churn_has_bounded_fresh_sampling_not_task_retry(case, stable_after):
    c, snapshot = case; observations = []; sleeps = []
    def sample(_):
        observations.append(1)
        if stable_after is None or len(observations) < stable_after:
            raise TransientMonitorError('GPU ownership changed during preflight')
        return deepcopy(snapshot)
    result = preflight(c, collector=sample, sleep=sleeps.append)
    assert len(observations) == (stable_after or 3)
    assert len(sleeps) == len(observations) - 1
    if stable_after is None:
        assert result['status'] == 'MONITOR_ERROR' and result['model_calls'] == 0
        assert all(not row['allowed'] for row in result['rows'])
    else:
        assert result['status'] == 'PASS' and result['attempts'] == stable_after


def test_mixed_same_named_foreign_process_blocks_only_occupied_gpu(case):
    c,s=case
    s['apps'].append(dict(gpu_uuid='GPU-3',pid=99,used_mib=50))
    s['processes'][99]=dict(s['worker'],pid=99,argv=['/fake/anchor','keepalive'])
    result=classify(c,s)
    assert result['rows'][0]['allowed']
    assert result['rows'][1]['category']=='UNKNOWN_OCCUPANCY'
    assert result['rows'][1]['unknown_pids']==[99]


def test_keepalive_memory_is_not_subtracted(case):
    c,s=case;s['gpus'][1]['free_mib']=15547
    result=classify(c,s)
    assert result['rows'][1]['category']=='INSUFFICIENT_MEMORY'
    assert result['rows'][1]['coexisting_pids']==[11]


def test_telemetry_errors_cannot_mean_idle(case):
    c,_=case
    def broken(_):raise OSError('NVML unavailable')
    result=preflight(c,collector=broken)
    assert result['status']=='MONITOR_ERROR' and len(result['rows'])==2
    assert not any(r['allowed'] for r in result['rows'])


def test_release_current_symlink_is_not_used_for_trust(case):
    c,s=case;c['current_release']='/trusted/current-now-points-to-v7'
    assert verify_status(c,s)['executable']=='/trusted/releases/v6/anchor'
    s['status']['release']='/trusted/current'
    with pytest.raises(MonitorError,match='Untrusted'):
        verify_status(c,s)


def test_owned_group_requires_live_leader_birth_not_only_pgid(case):
    c,s=case
    leader=dict(s['worker'],pid=20,ppid=1,pgid=20,sid=20,start_ticks=50000)
    s['processes'][20]=leader;s['apps'].append(dict(gpu_uuid='GPU-0',pid=20,used_mib=9000))
    owner=dict(pid=20,start='50000',boot='boot',group=20,gpu_uuid='GPU-0')
    assert classify(c,s,[owner])['rows'][0]['allowed']
    owner['start']='40000'
    assert classify(c,s,[owner])['rows'][0]['category']=='UNKNOWN_OCCUPANCY'


def test_process_snapshot_reads_executable_and_start_identity():
    value=process(os.getpid())
    assert value['pid']==os.getpid() and value['start_ticks']>0
    assert Path(value['executable']).is_absolute() and len(value['executable_sha256'])==64


def test_loading_peer_peak_is_reserved_before_actual_gpu_allocation(case):
    c,s=case
    s['gpus'][0]['free_mib']=20000
    leader=dict(s['worker'],pid=20,ppid=1,pgid=20,sid=20,start_ticks=50000)
    s['processes'][20]=leader  # A model loader not yet present in GPU process telemetry.
    owner=dict(pid=20,start='50000',boot='boot',group=20,gpu_uuid='GPU-0')
    row=classify(c,s,[owner])['rows'][0]
    assert row['peer_peak_reserve_mib']==13500
    assert row['required_free_mib']==29048
    assert row['category']=='INSUFFICIENT_MEMORY'


@pytest.mark.parametrize('field',['worker_peak_mib','safety_margin_mib','minimum_free_mib'])
def test_nonfinite_memory_threshold_never_bypasses_admission(case,field):
    c,s=case;c[field]=float('nan')
    assert preflight(c,collector=lambda _:s)['status']=='MONITOR_ERROR'


def test_resident_continuation_retains_two_peaks_without_charging_a_third(case):
    c,s=case;owners=[]
    for pid in (20,21):
        leader=dict(s['worker'],pid=pid,ppid=1,pgid=pid,sid=pid,start_ticks=50000+pid)
        s['processes'][pid]=leader
        s['apps'].append(dict(gpu_uuid='GPU-0',pid=pid,used_mib=8000))
        owners.append(dict(pid=pid,start=str(50000+pid),boot='boot',group=pid,gpu_uuid='GPU-0'))
    s['gpus'][0]['free_mib']=16000
    cold=classify(c,s,owners)['rows'][0]
    warm=classify(c,s,owners,continuing_pid=20)['rows'][0]
    assert cold['category']=='INSUFFICIENT_MEMORY' and cold['required_free_mib']==26548
    assert warm['allowed'] and warm['required_free_mib']==13048
    assert warm['peer_peak_reserve_mib']==11000 and warm['resident_continuation']
    s['gpus'][0]['free_mib']=13047
    assert not classify(c,s,owners,continuing_pid=20)['rows'][0]['allowed']
    with pytest.raises(MonitorError):classify(c,s,owners,continuing_pid=999)
    s['gpus'][0]['free_mib']=31000
    s['apps'].append(dict(gpu_uuid='GPU-0',pid=99,used_mib=1));s['processes'][99]=dict(s['worker'],pid=99)
    assert classify(c,s,owners,continuing_pid=20)['rows'][0]['category']=='UNKNOWN_OCCUPANCY'


@pytest.mark.parametrize('after_state,allowed',[
    ('PROTECTED_ASSIST',True),('WORKER_THRESHOLD_LOW',True),
    ('YIELD_MEMORY_PRESSURE',False),('WORKER_STARTING',False)])
def test_collector_checks_fresh_health_instead_of_freezing_utilization_state(case,monkeypatch,after_state,allowed):
    from AlphaBrain.research.dsol.execution import gpu_guard as guard
    c,s=case;after=deepcopy(s['status'])
    after.update(state=after_state,continuous_worker=True,checked_at=s['now'])
    reports=iter([json.dumps(s['status']),json.dumps(after)])
    def read(path,*args,**kwargs):
        values={'/proc/sys/kernel/random/boot_id':'boot','/state/profile':'fresh',
                '/proc/10/task/10/children':'11'}
        return next(reports) if str(path)==c['status_path'] else values[str(path)]
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(guard,'telemetry',lambda:(deepcopy(s['gpus']),deepcopy(s['apps'])))
    monkeypatch.setattr(guard,'process',lambda pid:deepcopy({10:s['supervisor'],11:s['worker']}[pid]))
    monkeypatch.setattr(guard,'sha_file',lambda path:s['release_hashes'][Path(path).name])
    monkeypatch.setattr(guard,'holds_supervisor_lock',lambda *a:True)
    monkeypatch.setattr(guard.socket,'gethostname',lambda:'node')
    monkeypatch.setattr(guard.time,'time',lambda:s['now'])
    monkeypatch.setattr(guard.time,'clock_gettime',lambda _:s['boottime'])
    monkeypatch.setattr(guard.os,'sysconf',lambda _:s['clock_ticks'])
    if allowed:
        result=guard.classify(c,guard.collect(c))
        assert result['status']=='PASS'
        assert result['keepalive']['supervisor_state']==after_state
        assert result['keepalive']['status_age_seconds']==0
    else:
        with pytest.raises(TransientMonitorError,match='not in an admitted live state'):
            guard.collect(c)


@pytest.mark.parametrize('field,value',[('worker_pid',12),('worker_gpu',0),('boot_id','other'),
                                     ('release','untrusted'),('process_identity',{})])
def test_status_transition_cannot_replace_worker_identity(case,field,value):
    from AlphaBrain.research.dsol.execution.gpu_guard import reconcile_status
    _,s=case;after=deepcopy(s['status']);after[field]=value
    with pytest.raises(TransientMonitorError,match=field):reconcile_status(s,after)


@pytest.mark.parametrize('target,changed_gpu,allowed',[
    ('GPU-0','GPU-3',True),('GPU-0','GPU-0',False),(None,'GPU-3',False)])
def test_actual_collector_scopes_cuda_membership_churn(case,monkeypatch,target,changed_gpu,allowed):
    from AlphaBrain.research.dsol.execution import gpu_guard as guard
    c,s=case
    def read(path,*args,**kwargs):
        values={'/proc/sys/kernel/random/boot_id':'boot','/state/profile':'fresh',
                '/proc/10/task/10/children':'11',c['status_path']:json.dumps(s['status'])}
        return values[str(path)]
    samples=iter([(deepcopy(s['gpus']),deepcopy(s['apps'])),
        (deepcopy(s['gpus']),deepcopy(s['apps'])+[dict(gpu_uuid=changed_gpu,pid=99,used_mib=1)])])
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(guard,'telemetry',lambda:next(samples))
    monkeypatch.setattr(guard,'process',lambda pid:deepcopy({10:s['supervisor'],11:s['worker']}[pid]))
    monkeypatch.setattr(guard,'sha_file',lambda path:s['release_hashes'][Path(path).name])
    monkeypatch.setattr(guard,'holds_supervisor_lock',lambda *a:True)
    monkeypatch.setattr(guard.socket,'gethostname',lambda:'node')
    monkeypatch.setattr(guard.time,'time',lambda:s['now'])
    monkeypatch.setattr(guard.time,'clock_gettime',lambda _:s['boottime'])
    monkeypatch.setattr(guard.os,'sysconf',lambda _:s['clock_ticks'])
    if allowed:
        snapshot=guard.collect(c,target_gpu_uuid=target)
        result=guard.classify(c,snapshot)
        assert result['status']=='PASS' and [r['uuid'] for r in result['rows']]==['GPU-0']
        assert result['keepalive']['pid']==11
    else:
        with pytest.raises(TransientMonitorError,match='ownership changed'):
            guard.collect(c,target_gpu_uuid=target)


def test_scoped_snapshot_still_rejects_unknown_target_occupant(case):
    c,s=case;s['target_gpu_uuid']='GPU-0'
    s['apps'].append(dict(gpu_uuid='GPU-0',pid=99,used_mib=1));s['processes'][99]=dict(s['worker'],pid=99)
    result=classify(c,s)
    assert len(result['rows'])==1 and result['rows'][0]['category']=='UNKNOWN_OCCUPANCY'
    s['worker']['executable_sha256']='changed'
    with pytest.raises(MonitorError,match='trusted release anchor'):classify(c,s)
