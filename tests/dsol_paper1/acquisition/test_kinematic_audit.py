import numpy as np
import pytest
from AlphaBrain.research.dsol.acquisition.kinematic_audit import camera_ancestry,inside_primitive,bounded_joint_path


def test_camera_chain_distinguishes_parent_and_head_joints():
    xml='<mujoco><worldbody><body name="base"><joint name="base_yaw"/><body name="head"><joint name="pan"/><camera name="eye"/></body></body></worldbody><actuator><motor name="motor" joint="pan"/></actuator></mujoco>'
    chain=camera_ancestry(xml,'eye')['chain']
    assert [x['body'] for x in chain]==['head','base']
    assert chain[0]['joints'][0]['actuators'][0]['name']=='motor'
    assert chain[1]['joints'][0]['actuators']==[]


def test_point_in_primitive_is_strict_and_unknown_is_not_free():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    np.testing.assert_array_equal(inside_primitive(p,'box',[1,1,1]),[True,False,False])
    np.testing.assert_array_equal(inside_primitive(p,'sphere',[1,0,0]),[True,False,False])
    assert inside_primitive(p,'mesh',[1,1,1]) is None


def test_bounded_path_respects_limits_step_and_endpoint():
    p=bounded_joint_path([0,0],[.3,-.2],[[-1,1],[-.5,.5]],.05)
    np.testing.assert_allclose(p[-1],[.3,-.2]);assert np.abs(np.diff(p,axis=0)).max() <= .05+1e-12
    with pytest.raises(ValueError):bounded_joint_path([0,0],[2,0],[[-1,1],[-.5,.5]],.05)
    with pytest.raises(ValueError):bounded_joint_path([0,0],[0,0],[[-1,1],[-.5,.5]],0)
