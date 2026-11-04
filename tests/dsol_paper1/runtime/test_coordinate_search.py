import numpy as np
import pytest
from benchmarks.LIBERO import continuous_evaluation as evaluator
from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount


@pytest.mark.parametrize('offset',[-5.,0.,5.])
def test_actual_search_uses_same_camera_model_and_preserves_physical_output(tmp_path,monkeypatch,offset):
    pose=PanTiltMount(np.eye(4)).pose([10.,0.]);poses=[]
    def request(message):
        nonlocal pose
        if message['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=dict(physical_state_id='same'))
        if message['kind']=='move_pose':pose=np.asarray(message['camera_to_world']);poses.append(pose.copy())
        return dict(receipt=dict(camera_to_world=pose.copy(),physical_state_id='same',settled=True),
                    example=dict(image=[pose.copy(),np.ones((2,2))],state=np.zeros(7),camera_intrinsics=np.eye(3),camera_to_world_opencv=pose.copy()))
    class Calls:
        def infer(self,example,noise,identity,capture):
            return dict(score=float(np.linalg.norm(example['image'][0]-np.eye(4))),external=np.ones(4),wrist=np.ones(4),actions=np.zeros((10,7)),trace=np.zeros((10,10,7)))
    decisions=[]
    def action(actor,obs):
        decisions.append(obs)
        return dict(stop=len(decisions)>1,delta_deg=np.zeros(2) if len(decisions)>1 else np.array([-5.,0.]))
    monkeypatch.setattr(evaluator,'actor_action',action)
    info,_=evaluator.search(dict(method='continuous_ppo',angles_deg=[10.,0.],proposal_seed=0,query_cap=8,
        mount_yaw_offset_deg=offset,search_yaw_limit_deg=25.),request,Calls(),np.zeros((2,10,7)),tmp_path,None,1.)
    np.testing.assert_allclose(info['selected_angles'],[5.,0.],atol=1e-14)
    np.testing.assert_allclose(info['mechanical_selected_angles'],[5.-offset,0.],atol=1e-14)
    np.testing.assert_allclose(pose,PanTiltMount(np.eye(4)).pose([5.,0.]),atol=1e-14)
    assert info['queries']==2 and info['event_angle_frame']=='shifted_mount_mechanical'
    # Mechanical state follows installation; physical camera start stays fixed.
    np.testing.assert_allclose(decisions[0]['joint_state'][0],(10.-offset)/25,atol=1e-7)
    # Actual execution must replay identical matrix arithmetic, not reconstruct
    # a mathematically equivalent pose through another Euler representation.
    scored_pose=pose.copy()
    replay=local_view_source(request,Calls(),tmp_path)
    evaluator.move_to_search_selection(replay,info)
    np.testing.assert_array_equal(pose,scored_pose)


def local_view_source(request,calls,tmp_path):
    from benchmarks.LIBERO.local_view_runtime import RenderedViewSource
    return RenderedViewSource(request,calls,np.zeros((2,10,7)),tmp_path)


def test_frame_replay_rejects_mismatched_physical_angle(tmp_path):
    class Source:
        mount=PanTiltMount(np.eye(4))
        current=np.zeros(2)
        def move(self,pose):raise AssertionError('Must reject before movement')
    with pytest.raises(ValueError,match='disagree'):
        evaluator.move_to_search_selection(Source(),dict(mount_yaw_offset_deg=5.,
            search_yaw_limit_deg=25.,mechanical_selected_angles=[-2.,1.],selected_angles=[4.,1.]))


def test_unapproved_mounting_parameters_rejected_before_search(tmp_path,monkeypatch):
    class Source:
        def __init__(self,*args):
            self.mount=PanTiltMount(np.eye(4)); self.physical='s';self.last_values=[dict(wrist=np.ones(4))]
        def move_angles(self,*args):pass
        def query(self):return None
    from benchmarks.LIBERO import local_view_runtime
    monkeypatch.setattr(local_view_runtime,'RenderedViewSource',Source)
    with pytest.raises(ValueError,match='Unsupported'):
        evaluator.search(dict(method='continuous_ppo',angles_deg=[10.,0.],proposal_seed=0,
            mount_yaw_offset_deg=12.,search_yaw_limit_deg=25.),None,None,None,tmp_path,None,1.)
