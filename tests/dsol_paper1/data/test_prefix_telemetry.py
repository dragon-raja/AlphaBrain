from types import SimpleNamespace
import numpy as np
from AlphaBrain.research.dsol.data.prefix_telemetry import frame,exact,save_branch
from AlphaBrain.research.dsol.data.artifacts import read,sha


def test_read_only_copy(tmp_path):
    data=SimpleNamespace(time=1.,qpos=np.array([1.,2.]),ncon=0,contact=[])
    image=np.ones((2,2,3),np.uint8);sim=SimpleNamespace(data=data)
    f=frame(sim,{'image':image,'language':'not numeric'});f['sim/qpos'][0]=99;f['obs/image'][0]=0
    assert data.qpos[0]==1 and image.min()==1
    assert f['sim/contacts'].shape==(0,3)
    save_branch(tmp_path/'branch',f,[],dict(repeat=0),False)
    receipt=read(tmp_path/'branch/receipt.json');assert not receipt['complete']
    assert receipt['arrays_sha256']==sha(tmp_path/'branch/telemetry.npz')


def test_strict_gate():
    assert exact([1.,2.],[1.,2.])['exact']
    assert not exact([1.,2.],[1.,2.00000001])['exact']
    assert not exact([1.],[1.,2.])['exact']
