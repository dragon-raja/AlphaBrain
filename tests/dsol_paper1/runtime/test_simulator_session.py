"""Real process/socket lifecycle with a CPU surrogate; no renderer/model import."""
import os
from pathlib import Path
import sys

import pytest

from benchmarks.LIBERO import simulator_session as module
from AlphaBrain.research.dsol.execution.dispatch_policy import _live_model_scope
from tests.dsol_paper1.paths import ROOT


@pytest.fixture
def case(tmp_path,monkeypatch):
    original=module.subprocess.Popen;children=[]
    def create(command,**kwargs):
        command=list(command)
        assert command[command.index('-m')+1]=='benchmarks.LIBERO.search_simulator'
        command[command.index('-m')+1]='tests.dsol_paper1.helpers.simulator_agent'
        child=original(command,**kwargs);children.append(child);return child
    monkeypatch.setattr(module.subprocess,'Popen',create)
    monkeypatch.setattr(module,'simulator_environment',lambda cfg:dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=''))
    config=dict(sim_python=sys.executable,source=str(ROOT))
    path=tmp_path/'config.json';path.write_text('{}')
    pool=module.SimulatorSessionPool(tmp_path/'logs')
    with _live_model_scope(dict(operation='LIBERO_BLOCK',permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1')),lambda:None):
        try:yield pool,path,config,children
        finally:pool.close()


def test_actual_ipc_reuses_process_but_fresh_initial_each_session(case,tmp_path):
    pool,path,config,children=case
    with pool.session(path,config,7,tmp_path/'first') as query:
        first=query(dict(kind='query_pose'))
        assert query(dict(kind='step'))['state']==8
    with pool.session(path,config,7,tmp_path/'second') as query:
        second=query(dict(kind='query_pose'))
    assert first==second and first['state']==7 and len(children)==1
    assert next(iter(pool.servers.values())).sequence==2
    pool.close()
    assert children[0].poll()==0
    with pytest.raises(RuntimeError,match='recovery'):
        with pool.session(path,config,7,tmp_path/'third'):pass


@pytest.mark.parametrize('fault',['caller','process'])
def test_partial_simulator_session_never_restarts_or_reuses(case,tmp_path,fault):
    pool,path,config,children=case
    with pytest.raises((RuntimeError,OSError,EOFError)):
        with pool.session(path,config,2,tmp_path/'first') as query:
            if fault=='caller':raise RuntimeError('partial trajectory')
            children[0].terminate();children[0].wait(timeout=5)
            query(dict(kind='query_pose'))
    assert pool.failed and len(children)==1 and children[0].poll() is not None
    with pytest.raises(RuntimeError,match='recovery'):
        with pool.session(path,config,2,tmp_path/'retry'):pass
