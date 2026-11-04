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
