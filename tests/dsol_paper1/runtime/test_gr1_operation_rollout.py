from types import SimpleNamespace
import numpy as np
import pytest
from benchmarks.Robocasa_tabletop.operation_rollout import run_selected_operation


class Model:
    contract=SimpleNamespace(execute_actions=2)
    def __init__(self): self.calls=[]
    def infer(self,example,noise,*,request):
        self.calls.append(request)
        return {'actions':np.zeros((2,29))}


class Backend:
    def __init__(self,fail=False): self.steps=0;self.fail=fail
    def begin_selected_view(self,epoch): self.epoch=epoch
    def capture_input(self,epoch): return {'fixture':True},self.epoch
    def step_absolute(self,action):
        self.steps+=1
        if self.fail: raise ConnectionError('lost action response')
        return {'success':self.steps==3,'terminated':False}


def run(model,backend,receipts,guard=lambda:None):
    return run_selected_operation(model,backend,episode_id='cpu-operation',epoch=1,
        noises=np.zeros((3,16,32),np.float32),max_actions=5,max_seconds=10,
        guard=guard,clock=lambda:0.,persist=lambda r:receipts.append(r))


def test_actual_operation_entry_stops_at_success_and_persists_counts():
    model=Model();backend=Backend();receipts=[]
    result=run(model,backend,receipts)
    assert result['success'] and result['actions']==3 and len(model.calls)==2
    assert receipts[-1]['status']=='COMPLETE'


def test_unknown_action_retains_model_consumption_no_retry():
    model=Model();backend=Backend(fail=True);receipts=[]
    with pytest.raises(ConnectionError):run(model,backend,receipts)
    assert backend.steps==1 and len(model.calls)==1
    assert receipts[-1]['status']=='ACTION_UNKNOWN' and receipts[-1]['completed_model_calls']==1
    assert receipts[-1]['outcome']=='UNKNOWN_REQUIRES_AUDIT'


def test_absent_authority_stops_before_model_and_action():
    model=Model();backend=Backend();receipts=[]
    def deny(): raise PermissionError('no GR1 permit')
    with pytest.raises(PermissionError):run(model,backend,receipts,deny)
    assert backend.steps==0 and model.calls==[]
