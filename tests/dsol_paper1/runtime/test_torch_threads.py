from types import SimpleNamespace
import subprocess
import sys
import pytest
from AlphaBrain.common.torch_runtime import configure_threads


def test_matching_thread_settings_never_repeat_one_shot_setter():
    def forbidden(*args):pytest.fail('Already configured; cannot repeat one-shot setter')
    fake=SimpleNamespace(get_num_threads=lambda:2,get_num_interop_threads=lambda:1,
                         set_num_threads=forbidden,set_num_interop_threads=forbidden)
    configure_threads(fake);configure_threads(fake)


def test_conflicting_late_configuration_does_not_swallow_failure():
    def rejected(*args):raise RuntimeError('parallel work already started')
    fake=SimpleNamespace(get_num_interop_threads=lambda:4,set_num_interop_threads=rejected)
    with pytest.raises(RuntimeError,match='already started'):configure_threads(fake)


def test_actual_torch_configuration_can_repeat_after_parallel_work():
    pytest.importorskip('torch')
    code='''import torch
from AlphaBrain.common.torch_runtime import configure_threads
configure_threads(torch)
x=torch.ones((128,128)); torch.mm(x,x)
configure_threads(torch)
assert torch.get_num_threads()==2 and torch.get_num_interop_threads()==1
'''
    subprocess.run([sys.executable,'-c',code],check=True,timeout=30)
