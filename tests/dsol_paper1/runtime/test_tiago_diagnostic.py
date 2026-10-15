"""Pure tests: importing the diagnostic must not start MuJoCo or a renderer."""
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import pytest
from benchmarks.Robocasa365.tiago_integration import contact_summary, prescreen_robot_base, allowed_support_contact


def test_support_exception_does_not_allow_cabinet_or_deep_floor_penetration():
    wheel='robot0_wheel_left_collision';floor='floor_1_room_g0'
    assert allowed_support_contact(wheel,floor,-1e-13,1.)
    assert allowed_support_contact(floor,wheel,0.,-1.)
    assert not allowed_support_contact(wheel,floor,-.001,1.)
    assert not allowed_support_contact(wheel,floor,0.,.5)
    assert not allowed_support_contact(wheel,'cabinet_g0',0.,1.)
    assert not allowed_support_contact('robot0_base_collision',floor,0.,1.)
    assert not allowed_support_contact(wheel,floor,float('nan'),1.)


def test_contact_summary_orders_penetration_and_keeps_names():
    model = SimpleNamespace(geom_bodyid=[0, 1])
    data = SimpleNamespace(ncon=2, contact=[
        SimpleNamespace(geom1=0, geom2=1, dist=-0.002),
        SimpleNamespace(geom1=0, geom2=1, dist=-0.12),
    ])
    api = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_GEOM='geom', mjOBJ_BODY='body'),
                          mj_id2name=lambda model, kind, index: f'{kind}_{index}')
    result = contact_summary(model, data, api)
    assert result['count'] == 2
    assert result['penetrating_1cm'] == 1
    assert result['deepest'][0]['distance'] == -0.12
    assert result['deepest'][0]['bodies'] == ['body_0', 'body_1']


def test_empty_contact_summary():
    assert contact_summary(None, SimpleNamespace(ncon=0, contact=[]), None) == {
        'count': 0, 'deepest': [], 'penetrating_1cm': 0}


def test_prescreen_is_bounded_and_does_not_solve_rejected_contacts():
    forwards=[]
    sim=SimpleNamespace(model=SimpleNamespace(_model=object(),get_joint_qpos_addr=lambda _:0),
                        data=SimpleNamespace(qpos=np.zeros(1),_data=SimpleNamespace(ncon=2)),
                        forward=lambda: forwards.append(True),get_state=lambda:None,set_state=lambda _:None)
    env=SimpleNamespace(sim=sim,rng=np.random.default_rng(0))
    utils=SimpleNamespace(no_collision=lambda _:nullcontext(),
                          generate_random_robot_pos=lambda *args:np.zeros(3),
                          set_robot_to_position=lambda *args:None,detect_robot_collision=lambda _:True)
    api=SimpleNamespace(mj_kinematics=lambda *args:None,mj_comPos=lambda *args:None,mj_collision=lambda *args:None)
    receipt=[]
    with pytest.raises(RuntimeError,match='3 native proposals'):
        prescreen_robot_base(env,np.zeros(3),np.zeros(3),.1,.1,.1,utils=utils,mujoco=api,
                             receipt=receipt,max_attempts=3)
    assert len(receipt)==3
    assert len(forwards)==1  # initial yaw only, no solve of rejected placements
