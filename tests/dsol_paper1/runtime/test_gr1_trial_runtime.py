"""Actual shared policy -> search -> selected operation, synthetic ports only."""
from types import SimpleNamespace
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewConfig, PoseReceipt
from AlphaBrain.research.dsol.data.search_core import core_spec
from benchmarks.Robocasa_tabletop.model_runtime import ScoreContract
from benchmarks.Robocasa_tabletop.policy_binding import make_binding,build_policy
from benchmarks.Robocasa_tabletop.search_runtime import ArticulatedHeadMount,GR1SearchSource
from benchmarks.Robocasa_tabletop.trial_runtime import run_search_and_operation
from benchmarks.Robocasa_tabletop.operation_backend import SelectedTaskBackend,NativeEgoCapture
from benchmarks.Robocasa_tabletop.task_backend import EgoFrame


def fixture():
    torch.set_num_threads(1)
    def fk(angles):
        p=np.eye(4);p[:2,3]=np.asarray(angles)/100;return p
    class Backend:
        angles=np.zeros(2);epoch=0
        def pose_receipt(self):return PoseReceipt(fk(self.angles),'fixed')
        def move_head_degrees(self,a):self.angles=a.copy();return self.pose_receipt()
        def reset_observation_cache(self,e):self.epoch=e
        def capture_input(self,e):
            return dict(prompt='CPU fixture',**{'observation/image':np.zeros((2,2,3),np.uint8),
                        'observation/state':np.zeros(29,np.float32)}),self.epoch
    class Model:
        contract=ScoreContract(10,10,2,2);calls=0
        def infer(self,*args,request):
            self.calls+=1
            return dict(feature=np.ones(4608,np.float32),score=1.,actions=np.zeros((2,29)))
    class Operation:
        steps=0;fail=False
        def accept_selected_view(self,pose,*,instruction,epoch):
            assert pose.physical_state_id=='fixed' and instruction=='CPU fixture'
            self.epoch=epoch
        def begin_selected_view(self,epoch):assert epoch==self.epoch
        def capture_input(self,epoch):return {'fixture':True},epoch
        def step_absolute(self,action):
            self.steps+=1
            if self.fail:raise ConnectionError('unknown physical action')
            return dict(success=self.steps==3,terminated=False)
    model=Model();backend=Backend()
    source=GR1SearchSource(ArticulatedHeadMount(LocalViewConfig(),fk,joint_limits_deg=[[-40,40],[-25,25]]),
        backend,model,noise_bank=np.zeros((2,16,32),np.float32),episode_id='cpu-trial',
        model_sha256='a'*64,producer_sha256='a'*64,history_sha256='a'*64,guard=lambda:None)
    provenance={k:'a'*64 for k in ('model_sha256','producer_sha256','history_sha256',
                                  'processor_sha256','normalization_sha256','source_sha256')}
    binding=make_binding(core_spec('U12',width=16),provenance,model.contract)
    policy=build_policy(binding)
    with torch.no_grad():policy.stop_head.bias.fill_(100.)
    return source,policy,binding,Operation()


def run(parts,records):
    return run_search_and_operation(*parts,instruction='CPU fixture',
        session_options=dict(initial_angles=[0,0],physical_state_id='fixed',score_scale=1.,seed=4),
        search_decisions=8,search_seconds=10.,operation_noises=np.zeros((2,16,32),np.float32),
        operation_actions=4,operation_seconds=10.,guard=lambda:None,clock=lambda:0.,persist=records.append)


def test_full_trial_counts_both_stages_and_refuses_replay():
    parts=fixture();records=[];r=run(parts,records)
    assert r['status']=='COMPLETE' and r['completed_model_calls']==4
    assert r['operation']['success'] and r['operation']['actions']==3
    assert parts[0].status=='COMPLETE'
    with pytest.raises(ValueError):run(parts,records)
    assert parts[0].runtime.calls==4


def test_operation_failure_holds_whole_trial_and_preserves_search_cost():
    parts=fixture();parts[-1].fail=True;records=[]
    with pytest.raises(ConnectionError):run(parts,records)
    assert parts[0].status=='HOLD' and parts[0].runtime.calls==3
    assert records[-1]['stage']=='OPERATION' and records[-1]['detail']['status']=='ACTION_UNKNOWN'
    assert records[-1]['outcome']=='UNKNOWN_REQUIRES_AUDIT'
    with pytest.raises(ValueError):run(parts,records)


def backend_fixture():
    state=dict(q=np.zeros(3),time=0.,scene='initial')
    class IO:
        def read_head_state(self):return dict(positions=tuple(state['q']),velocities=(0.,0.,0.),observed_at=state['time'])
        def non_head_state_identity(self):return state['scene']
    class Geometry:
        def measured_pose(self,angles,roll):return np.eye(4)
    class Env:
        steps=0
        def _check_success(self):return self.steps==2
        def step(self,action):
            self.steps+=1;state['time']+=.05;state['scene']='moved';return {},0.,False,{}
    env=Env();io=IO();geometry=Geometry()
    def frame(epoch,seq):return EgoFrame(np.zeros((2,2,3),np.uint8),epoch,seq,state['time'],np.eye(4))
    backend=SelectedTaskBackend(env,io,geometry,frame,guard=lambda:None,head_check=lambda q:None,
        step_controlled=env.step,position_tolerance_rad=.01,
        input_builder=lambda e,p:{'prompt':'task'},action_builder=lambda e,a:a)
    return backend,state,env,io,geometry


def test_native_operation_allows_scene_evolution_but_not_head_drift():
    b,s,_,_,_=backend_fixture()
    b.accept_selected_view(PoseReceipt(np.eye(4),'initial'),instruction='task',epoch=2)
    b.begin_selected_view(2);b.capture_input(2)
    assert b.step_absolute(np.zeros(29))==dict(success=False,terminated=False)
    b.capture_input(2)
    s['q'][0]=.03
    with pytest.raises(ValueError,match='drift'):b.capture_input(2)
    assert b.status=='HOLD'


def test_handoff_mismatch_and_stale_frame_rejected():
    b,s,_,_,_=backend_fixture()
    with pytest.raises(ValueError,match='handoff'):
        b.accept_selected_view(PoseReceipt(np.eye(4),'another'),instruction='task',epoch=0)
    b,s,_,_,_=backend_fixture()
    b.accept_selected_view(PoseReceipt(np.eye(4),'initial'),instruction='task',epoch=0);b.begin_selected_view(0)
    b.capture=lambda e,q:EgoFrame(np.zeros((2,2,3),np.uint8),e,q,-1.,np.eye(4))
    with pytest.raises(ValueError,match='Stale'):b.capture_input(0)


def test_renderer_denied_before_any_context_or_frame():
    _,_,env,io,geometry=backend_fixture()
    def denied():raise PermissionError('no rendering admission')
    capture=NativeEgoCapture(env,io,geometry,guard=lambda:None,render_guard=denied)
    with pytest.raises(PermissionError):capture(0,1)
