"""Real local subprocess / shared budget primitives, synthetic model only."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from benchmarks.Robocasa_tabletop.model_process import CPUModelProcess
from AlphaBrain.research.dsol.execution.phase_budget import PhaseBudget
from AlphaBrain.research.dsol.execution.manifest import file_digest

ROOT=Path(__file__).resolve().parents[3]


def inputs():
    return {'observation/state':np.zeros(29,np.float32)},np.zeros((16,32),np.float32)


def test_local_worker_requires_each_live_grant_and_rejects_duplicate(tmp_path):
    grants=[]
    with CPUModelProcess(sys.executable,ROOT,guard=lambda r:grants.append(r),log_path=tmp_path/'worker.log') as p:
        request=dict(query_id='cpu-1',epoch=2,member=0)
        value=p.infer(*inputs(),request=request)
        assert value['synthetic'] and value['feature'].shape==(4608,) and grants==[request]
        with pytest.raises(RuntimeError):p.infer(*inputs(),request=request)
        assert p.closed and len(grants)==1
        with pytest.raises(RuntimeError):p.infer(*inputs(),request=dict(query_id='cpu-2',epoch=2,member=0))


def test_disconnect_after_reservation_holds_shared_budget(tmp_path):
    auth=tmp_path/'authorization.json'
    auth.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL',total_model_call_limit=2,
                                   fixture='CPU_ONLY_NOT_SCIENTIFIC_AUTHORIZATION')))
    config=tmp_path/'call.json';config.write_text(json.dumps(dict(limits=dict(score=1,paired_execution=0,nominal_execution=0))))
    budget=PhaseBudget(tmp_path/'local',tmp_path/'persistent',auth,authorization_sha256=file_digest(auth))
    def guard(request):
        budget.reserve('cpu-call-1',config)
        raise ConnectionError('synthetic loss after durable reservation')
    with CPUModelProcess(sys.executable,ROOT,guard=guard,log_path=tmp_path/'worker.log') as p:
        with pytest.raises(ConnectionError):p.infer(*inputs(),request=dict(query_id='one',epoch=1,member=0))
        assert p.closed and budget.status()['held_unsettled']==1
    with pytest.raises(ValueError):budget.reserve('cpu-call-1',config)
    assert budget.status()['held_unsettled']==1


@pytest.mark.parametrize('field', ['prompt', 'state', 'image', 'noise', 'epoch'])
def test_grant_fingerprint_binds_actual_observation_and_noise(field):
    from benchmarks.Robocasa_tabletop.model_process import call_identity
    example, noise = inputs()
    example.update(prompt='pick cup', image=np.zeros((2, 2, 3), np.uint8))
    request = dict(query_id='q', epoch=0, member=0)
    before = call_identity(example, noise, request)
    if field == 'prompt': example['prompt'] = 'pick plate'
    elif field == 'state': example['observation/state'][0] = 1
    elif field == 'image': example['image'][0, 0, 0] = 1
    elif field == 'noise': noise[0, 0] = 1
    else: request['epoch'] = 1
    assert before != call_identity(example, noise, request)


def test_invalid_input_never_requests_grant(tmp_path):
    grants = []
    with CPUModelProcess(sys.executable, ROOT, guard=grants.append,
                         log_path=tmp_path/'child.log') as p:
        example, noise = inputs()
        noise[0, 0] = np.nan
        with pytest.raises(ValueError, match='finite'):
            p.infer(example, noise, request=dict(query_id='q', epoch=0, member=0))
        assert grants == [] and p.closed


def test_disconnect_after_result_is_not_accepted_or_replayed(tmp_path):
    checks = []
    grants = []
    def continuation():
        checks.append(1)
        if len(checks) == 3:
            raise ConnectionError('lost center after child returned')
    with CPUModelProcess(sys.executable, ROOT, guard=grants.append,
                         continuation_guard=continuation, log_path=tmp_path/'child.log') as p:
        with pytest.raises(ConnectionError):
            p.infer(*inputs(), request=dict(query_id='q', epoch=0, member=0))
        assert len(grants) == 1 and p.closed and p.process.poll() is not None
        with pytest.raises(RuntimeError, match='reconnect'):
            p.infer(*inputs(), request=dict(query_id='q', epoch=0, member=0))


@pytest.mark.parametrize('fault', [None, 'payload', 'grant', 'timeout', 'duplicate'])
def test_real_model_worker_protocol_with_injected_cpu_runtime(monkeypatch, fault):
    """Exercise the real child loop, replacing only the heavy model constructor.

    No JAX import, checkpoint load, render or inference occurs in this fixture.
    """
    from multiprocessing import Pipe
    from threading import Thread
    from types import SimpleNamespace
    from benchmarks.Robocasa_tabletop import model_process as ipc, model_runtime
    parent, child = Pipe()
    calls = []
    class Runtime:
        def __init__(self, **kwargs): self.guard = kwargs['call_guard']
        def infer(self, example, noise, *, request):
            self.guard(request)
            calls.append(request)
            return dict(synthetic=True)
    monkeypatch.setattr(model_runtime, 'GR1ModelRuntime', Runtime)
    monkeypatch.setattr(ipc, 'Client', lambda *a, **k: child)
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace())
    config = dict(score_contract=dict(denoise_steps=10, prefix=10, execute_actions=16, score_actions=16))
    thread = Thread(target=ipc._model_worker, args=('unused', config, 0.1), daemon=True)
    thread.start()
    def receive():
        assert parent.poll(3), 'child did not answer'
        return parent.recv()
    try:
        assert receive()['kind'] == 'MODEL_LOADED_CPU_ONLY'
        example, noise = inputs()
        request = dict(query_id='q', epoch=1, member=0)
        stamp = ipc.call_identity(example, noise, request)
        row = dict(kind='INFER', sequence=1, example=example, noise=noise,
                   request=request, call_sha256=stamp)
        if fault == 'payload': row['noise'][0, 0] = 1
        parent.send(row)
        intent = receive()
        if fault == 'payload':
            assert intent['kind'] == 'ERROR' and calls == []
        else:
            assert intent == dict(kind='AUTHORIZE', sequence=1, request=request, call_sha256=stamp)
            if fault != 'timeout':
                parent.send(dict(kind='GO', sequence=1, call_sha256='bad' if fault == 'grant' else stamp))
            result = receive()
            if fault in ('grant', 'timeout'):
                assert result['kind'] == 'ERROR' and calls == []
            else:
                assert result == dict(kind='RESULT', sequence=1, call_sha256=stamp, value=dict(synthetic=True))
                assert calls == [request]
                if fault == 'duplicate':
                    parent.send(dict(row, sequence=2))
                    assert receive()['kind'] == 'ERROR' and len(calls) == 1
    finally:
        parent.close()
        thread.join(3)
        assert not thread.is_alive()
