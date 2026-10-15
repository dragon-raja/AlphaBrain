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


@pytest.mark.parametrize('owned,free,expected',[(True,18000,True),(False,18000,False),(True,12000,False)])
def test_dual_worker_requires_owned_peer_and_memory_headroom(tmp_path,monkeypatch,owned,free,expected):
    from AlphaBrain.research.dsol.data.artifacts import write
    identity=process_identity(os.getpid())
    if identity is None:pytest.skip('Process identity unavailable')
    handle=dict(identity,group=identity['group'] if owned else identity['group']+100000)
    write(tmp_path/'state.json',dict(tasks={'a':dict(status='RUNNING',slot=dict(host='h',id='0'),handle=handle)}))
    def output(command,**kwargs):
        if '--query-compute-apps=gpu_uuid,pid' in command:return f'GPU-0, {os.getpid()}\n'
        assert command[command.index('-i')+1]=='0'
        return str(free) if '--query-gpu=memory.free' in command else 'GPU-0'
    monkeypatch.setattr(subprocess,'check_output',output)
    backend=ProbeBackend(dict(runtime=dict(workers_per_gpu=2)),tmp_path)
    assert backend.gpu_available(dict(host='h',id='0:1',gpu_id='0'))==expected
