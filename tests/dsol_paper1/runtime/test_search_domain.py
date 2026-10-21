"""CPU-only reset and hook-control contract regressions."""
from contextlib import contextmanager
from copy import deepcopy
import json

import numpy as np
import pytest

from benchmarks.LIBERO import search_domain as module


def example():
    return dict(lang='frozen task',image=[np.zeros((4,4,3),np.uint8)]*2,
        state=np.zeros(7),camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4))


@pytest.mark.parametrize('field',['state','camera_intrinsics','camera_to_world_opencv','image','lang'])
def test_reset_difference_is_not_tolerated(field):
    a=example();b=deepcopy(a)
    if field=='lang':b[field]='other'
    elif field=='image':b[field][0][0,0,0]=1
    else:b[field].flat[0]+=1e-9
    with pytest.raises(ValueError):module.exact_input(a,b)


@pytest.mark.parametrize('perturb',[False,True])
def test_four_calls_per_initial_and_fail_stop(tmp_path,monkeypatch,perturb):
    noise=tmp_path/'noise.npz';np.savez(noise,score=np.zeros((16,2,10,7),np.float32))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(
        domain_initials=list(range(16)),domain_call_cap=64,input_hashes={},domain_noise=str(noise))))
    calls=[];commands=[]
    class Runtime:
        def __init__(self,c,o,limit):self.calls=0;assert limit==64
        def infer(self,item,noise,identity,capture):
            self.calls+=1;calls.append((identity,capture))
            return dict(trace=np.ones(3)*(perturb and capture),actions=np.zeros(7),
                score=1.,external=np.zeros(4608),wrist=np.zeros(4608))
    @contextmanager
    def simulator(*args):
        def request(command):
            commands.append(command['kind']);assert command['kind'] in ('render','reset')
            return dict(example=example())
        yield request
    monkeypatch.setattr(module,'PolicyRuntime',Runtime)
    monkeypatch.setattr(module,'simulator',simulator)
    monkeypatch.setattr(module,'check_simulator_dependencies',lambda _:None)
    out=tmp_path/'result'
    if perturb:
        with pytest.raises(ValueError,match='hook changed'):module.run(config,out,execute=True)
        assert len(calls)==2 and not (out/'status.json').exists()
    else:
        module.run(config,out,execute=True)
        assert len(calls)==64 and len(commands)==48
        assert [c for _,c in calls]==[False,True]*32
        assert json.loads((out/'status.json').read_text())['calls']==64
