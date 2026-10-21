import numpy as np
import pytest

from benchmarks.LIBERO.search_capacity import run, wait_barrier
from benchmarks.LIBERO.search_runtime import exact_prediction
from benchmarks.LIBERO.search_queue import SearchBackend, verified_coexisting_process


def test_capacity_requires_explicit_execution(tmp_path):
    with pytest.raises(ValueError,match='Explicit'):
        run('missing',tmp_path/'result')
    assert not (tmp_path/'result').exists()


@pytest.mark.parametrize('key',['trace','actions','score','external','wrist'])
def test_capacity_checks_all_outputs_without_tolerance(key):
    expected={k:np.zeros(3) for k in ('trace','actions','score','external','wrist')}
    actual={k:v.copy() for k,v in expected.items()}
    actual[key][0]=1e-10
    with pytest.raises(ValueError,match=key):
        exact_prediction(actual,expected)


def test_barrier_does_not_silently_continue(tmp_path):
    with pytest.raises(TimeoutError):
        wait_barrier(tmp_path/'missing',timeout=-1)


def test_production_requires_real_capacity_receipt(tmp_path):
    plan=dict(runtime=dict(host='fresh',mode='evaluation',admission=dict(path=str(tmp_path/'missing'),sha256='x')))
    backend=SearchBackend(plan,tmp_path)
    with pytest.raises(ValueError,match='production'):
        backend.admit(dict(host='fresh',id='0'))


@pytest.mark.parametrize('field',['pid','start','boot','group','gpu_uuid','command_sha256','executable_sha256'])
def test_keepalive_identity_cannot_be_replaced_by_name(field):
    import hashlib
    current=dict(pid=5,start='1',boot='b',group=5,state='S')
    command=b'/existing/anchor\0--device\07'
    binding={**current,'gpu_uuid':'gpu','command_sha256':hashlib.sha256(command).hexdigest(),
             'executable_sha256':'binary'}
    assert verified_coexisting_process(binding,current,command,'binary','gpu')
    binding[field]='changed'
    assert not verified_coexisting_process(binding,current,command,'binary','gpu')


def test_original_fleet_stop_reason_is_preserved(tmp_path):
    from benchmarks.LIBERO.search_fleet import first_stop
    from AlphaBrain.research.dsol.data.artifacts import read
    path=tmp_path/'STOP.json'
    first_stop(path,dict(reason='unknown GPU process'))
    first_stop(path,dict(reason='cancelled by global stop'))
    assert read(path)==dict(reason='unknown GPU process')


def test_capacity_checks_last_card_before_starting_any_worker(tmp_path,monkeypatch):
    import benchmarks.LIBERO.search_fleet as module
    from AlphaBrain.research.dsol.data.artifacts import write, read
    shared=tmp_path/'phase'/'host';shared.mkdir(parents=True)
    manifest=tmp_path/'plan.json'
    write(manifest,dict(runtime=dict(host='host'),slots=[dict(id=str(i)) for i in range(8)]))
    checked=[]
    class Backend:
        def __init__(self,*args):pass
        def admit(self,slot):
            checked.append(slot['id'])
            return slot['id']!='7'
    monkeypatch.setattr(module,'SearchBackend',Backend)
    monkeypatch.setattr(module,'prepare',lambda *args:None)
    monkeypatch.setattr(module,'run',lambda *args,**kw:pytest.fail('A model must not launch before all cards pass'))
    monkeypatch.setattr(module.signal,'signal',lambda *args:None)
    with pytest.raises(RuntimeError,match='preflight refused'):
        module.local(manifest,tmp_path/'control',shared)
    assert checked==[str(i) for i in range(8)]
    assert 'preflight refused' in read(shared.parent/'STOP.json')['error']
