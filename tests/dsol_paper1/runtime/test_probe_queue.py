import os
import subprocess
from pathlib import Path

import pytest

from benchmarks.Robocasa365.probe_queue import ProbeBackend,process_identity,group_members


@pytest.mark.skipif(process_identity(os.getpid()) is None,reason='Sandbox does not expose process /proc; run host admission separately')
def test_process_identity_is_current_and_stable():
    row=process_identity(os.getpid())
    assert row['pid']==os.getpid()
    assert row==process_identity(os.getpid())
    assert any(x['pid']==os.getpid() for x in group_members(row['group']))


def test_no_receipt_never_relaunches(tmp_path):
    backend=ProbeBackend({'runtime':{}},tmp_path)
    with pytest.raises(RuntimeError,match='manual process audit'):
        backend.locate({'key':'x','output':str(tmp_path/'result')},{},1)


@pytest.mark.skipif(process_identity(os.getpid()) is None,reason='Sandbox does not expose process /proc; run host admission separately')
def test_cancel_only_owned_process_group(tmp_path):
    backend=ProbeBackend({'runtime':{}},tmp_path)
    p=subprocess.Popen(['sleep','20'],start_new_session=True)
    handle=process_identity(p.pid)
    backend.processes[p.pid]=p
    try:
        assert backend.alive(handle)
        wrong=dict(handle,start='not-the-same-process')
        with pytest.raises(RuntimeError,match='PID identity'):backend.cancel(wrong)
        assert p.poll() is None
        backend.cancel(handle);p.wait(timeout=5)
        assert not backend.alive(handle)
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=5)


def test_monitoring_error_blocks_instead_of_treating_gpu_as_idle(tmp_path,monkeypatch):
    import AlphaBrain.research.dsol.execution.gpu_guard as guard
    monkeypatch.setattr(guard,'preflight',lambda *a,**k:dict(status='MONITOR_ERROR',error='stale'))
    backend=ProbeBackend(dict(runtime=dict(host='h',gpu_guard=dict(profile='h'))),tmp_path)
    with pytest.raises(RuntimeError,match='failed closed'):
        backend.gpu_available(dict(host='h',id='0'))


def test_start_rechecks_and_refuses_without_launch_gap(tmp_path,monkeypatch):
    backend=ProbeBackend(dict(runtime={}),tmp_path)
    backend.worker_command=lambda task:['unused']
    availability=iter([True,False])
    backend.gpu_available=lambda slot:next(availability)
    assert backend.gpu_available(dict(host='h',id='0'))  # Earlier preflight passed.
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('Must not spawn after changed ownership'))
    task=dict(key='trial',output=str(tmp_path/'result'))
    with pytest.raises(RuntimeError,match='changed before launch'):
        backend.start(task,dict(host='h',id='0'),1)
    assert backend.locate(task,{},1) is None
    assert not (tmp_path/'result').exists()


def test_missing_supervisor_contract_never_falls_back_to_process_names(tmp_path):
    backend=ProbeBackend(dict(runtime={}),tmp_path)
    with pytest.raises(RuntimeError,match='gpu_guard'):
        backend.gpu_available(dict(host='h',id='0'))
