import numpy as np
import pytest
from benchmarks.LIBERO import continuous_evaluation as evaluator
from benchmarks.LIBERO.local_view_runtime import RenderedViewSource
from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount


@pytest.mark.parametrize('offset',[-15.,0.,15.])
@pytest.mark.parametrize('method',['stay','fixed_home','continuous_ppo','local_near_best_accel'])
def test_full_methods_share_first_input_and_exact_execution_pose(tmp_path,monkeypatch,offset,method):
    if method=='local_near_best_accel':pytest.importorskip('gymnasium')
    current=PanTiltMount(np.eye(4)).pose([0.,0.]);initial_pose=None
    def request(msg):
        nonlocal current
        if msg['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=dict(physical_state_id='s'))
        if msg['kind']=='move_pose':current=np.asarray(msg['camera_to_world']).copy()
        return dict(receipt=dict(camera_to_world=current.copy(),physical_state_id='s',settled=True),
            example=dict(lang='synthetic test',image=[current.copy(),np.ones((2,2))],state=np.zeros(7),camera_intrinsics=np.eye(3),camera_to_world_opencv=current.copy()))
    class Calls:
        def infer(self,example,noise,identity,capture):
            nonlocal initial_pose
            if initial_pose is None:initial_pose=example['image'][0].copy()
            return dict(score=1.+float(np.linalg.norm(example['image'][0]-initial_pose)),
                external=example['image'][0].ravel(),wrist=np.ones(16),actions=np.zeros((10,7)),trace=np.zeros((10,10,7)))
    decisions=[]
    def actor_action(actor,obs):
        decisions.append(obs)
        return dict(stop=len(decisions)>1,delta_deg=np.zeros(2) if len(decisions)>1 else np.array([-2.,-2.]))
    monkeypatch.setattr(evaluator,'actor_action',actor_action)
    row=dict(method=method,coordinate_family='heldout_installation_v1',mount_yaw_offset_deg=offset,
        search_yaw_limit_deg=15.,angles_deg=[offset+10.,10.],proposal_seed=10,query_cap=8)
    info,_=evaluator.search(row,request,Calls(),np.zeros((2,10,7)),tmp_path,None,1.)
    physical=current.copy()
    current=np.eye(4)
    source=RenderedViewSource(request,Calls(),np.zeros((2,10,7)),tmp_path)
    evaluator.move_to_search_selection(source,info)
    np.testing.assert_array_equal(current,physical)
    if method=='fixed_home':
        assert info['selected_angles']==[offset,0.] and info['queries']==2
        assert info['selection_rule']=='mechanical_home_ignores_scores'
    else:
        np.testing.assert_array_equal(info['selected_angles'],row['angles_deg'])
    shifted=PanTiltMount(PanTiltMount(np.eye(4)).pose([offset,0.]))
    np.testing.assert_array_equal(initial_pose,shifted.pose([10.,10.]))
