from pathlib import Path
import pytest

from AlphaBrain.research.dsol.data.artifacts import write
from benchmarks.Robocasa365.render_queue import RenderBackend


def test_renderer_uses_frozen_plan_and_explicit_initial(tmp_path):
    runtime={'python':'/python','plan':'/frozen/plan.json'}
    backend=RenderBackend({'runtime':runtime},tmp_path)
    command=backend.worker_command({'payload':{'initial':'abc'},'output':'/results/abc'})
    assert command==['/python','-u','-m','benchmarks.Robocasa365.confirmation_render',
                     '--plan','/frozen/plan.json','--initial','abc','--output','/results/abc']


def test_failed_pilot_never_admits_gpu(tmp_path):
    write(tmp_path/'admission.json',{'status':'FAILED'})
    backend=RenderBackend({'runtime':{'host':'fresh','admission':str(tmp_path/'admission.json')}},tmp_path)
    with pytest.raises(ValueError,match='admission missing'):
        backend.admit({'host':'fresh','id':'0'})


def test_wrong_host_rejected_without_resource_access(tmp_path):
    backend=RenderBackend({'runtime':{'host':'fresh'}},tmp_path)
    with pytest.raises(ValueError,match='Wrong host'):
        backend.admit({'host':'other','id':'0'})
