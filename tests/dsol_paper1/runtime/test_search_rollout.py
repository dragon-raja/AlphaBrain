"""Paired execution contract; no simulator, model, training or CUDA."""
from copy import deepcopy
import numpy as np
import pytest

from benchmarks.LIBERO.search_rollout import rollout_episode,validate_block


def test_stop_delivery_targets_only_explicit_local_fleet_marker(tmp_path):
    import sys
    from benchmarks.LIBERO.search_fleet import stop_hosts
    root=tmp_path/'own-control'
    errors=stop_hosts(dict(hosts=[dict(name='local',python=sys.executable,ssh=[],control=str(root))]))
    assert errors==[] and (tmp_path/'own-control.stop').read_text()=='fleet stopped\n'
    assert not root.exists()


def test_full_task_adapter_requires_both_admission_receipts(tmp_path):
    import json
    from AlphaBrain.research.dsol.data.artifacts import sha
    from benchmarks.LIBERO.search_queue import SearchBackend
    capacity=tmp_path/'capacity.json';capacity.write_text(json.dumps(dict(status='PASS')))
    domain=tmp_path/'domain.json';domain.write_text(json.dumps(dict(status='FAILED_STOPPED')))
    plan=dict(runtime=dict(host='fresh',mode='full_tasks',
        capacity_admission=dict(path=str(capacity),sha256=sha(capacity)),
        domain_admission=dict(path=str(domain),sha256=sha(domain))))
    backend=SearchBackend(plan,tmp_path)
    with pytest.raises(ValueError,match='domain_admission'):
        backend.admit(dict(host='fresh',id='0'))


@pytest.mark.parametrize('change',[{'repeats':[0]},{'cohort':'C'},{'methods':['ppo']},{'start':66,'cohort':'B'}])
def test_block_cannot_expand_or_select_favorable_trials(change):
    block=dict(cohort='A',initial=2,start=65,methods=['stay','local_accel','ppo'],repeats=list(range(8)))
    block.update(change)
    with pytest.raises(ValueError):validate_block(block)


@pytest.mark.parametrize('success',[False,True])
def test_execution_pairing_clipping_step_boundary_and_negative_retention(success):
    example=dict(image=[np.zeros((2,2,3),np.uint8)]*2,state=np.zeros(7),
        camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4))
    calls=[];chunks=[]
    class Bank:
        def get(self,key,repeat,replan):
            calls.append((key,repeat,replan));return dict(noise=np.zeros((10,7)),noise_sha256=str(replan))
    def infer(item,noise,identity,capture):
        assert capture is False and identity['repeat']==3
        return dict(actions=np.full((10,7),2.))
    def request(command):
        chunk=command['actions'];chunks.append(chunk)
        return dict(example=example,executed=2 if success else len(chunk),success=success)
    result,arrays=rollout_episode(request,infer,example,Bank(),'same-physical-state',3,dict(stage='execution'))
    expected=1 if success else 60
    assert calls==[('same-physical-state',3,r) for r in range(expected)]
    assert result['success'] is success and result['steps']==(2 if success else 300)
    assert all(c.shape==(5,7) and np.all(c==1) for c in chunks)
    assert arrays['actions'].shape==(result['steps'],7)


def test_camera_drift_fails_instead_of_returning_success():
    item=dict(image=[np.zeros((2,2,3))]*2,state=np.zeros(7),camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4))
    changed=deepcopy(item);changed['camera_to_world_opencv'][0,3]=1e-9
    class Bank:
        def get(self,*args):return dict(noise=np.zeros((10,7)),noise_sha256='x')
    with pytest.raises(ValueError,match='camera moved'):
        rollout_episode(lambda _:dict(example=changed,executed=5,success=True),
            lambda *args,**kw:dict(actions=np.zeros((10,7))),item,Bank(),'state',0,{})
