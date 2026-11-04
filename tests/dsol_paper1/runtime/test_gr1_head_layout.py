import pytest
import xml.etree.ElementTree as ET
from benchmarks.Robocasa_tabletop.head_layout import read_head_layout,extend_initial_joint_state
from benchmarks.Robocasa_tabletop.head_layout import extend_generalized_joint_state, restore_head_actuation


def model(tmp_path):
    p=tmp_path/'robot.xml'
    p.write_text('<mujoco><compiler angle="radian"/><worldbody><body>'+''.join(
        f'<joint name="head_{n}" axis="{a}" range="-1 1"/>' for n,a in [('yaw','0 0 1'),('roll','1 0 0'),('pitch','0 1 0')])+'</body></worldbody><actuator>'+''.join(f'<motor name="m_{n}" joint="head_{n}"/>' for n in ['yaw','roll','pitch'])+'</actuator></mujoco>')
    return p


def test_targets_are_named_and_roll_is_explicit_not_raw_motor_ctrl(tmp_path):
    layout=read_head_layout(model(tmp_path))
    assert layout.actuator_kind=='motor'
    assert layout.named_targets([.2,.3],roll_hold=0)=={'head_yaw':.2,'head_roll':0.,'head_pitch':.3}
    with pytest.raises(ValueError):layout.named_targets([.2,2.],roll_hold=0)
    with pytest.raises(TypeError):layout.named_targets([.2,.3])


def test_changed_axis_or_units_rejected(tmp_path):
    p=model(tmp_path);p.write_text(p.read_text().replace('radian','degree'))
    with pytest.raises(ValueError):read_head_layout(p)
    p=model(tmp_path);p.write_text(p.read_text().replace('0 0 1','0 0 2'))
    with pytest.raises(ValueError):read_head_layout(p)


def test_initial_state_mapping_preserves_old_joints_despite_index_shift():
    old={'waist':.1,'left_arm':.4,'right_arm':-.3}
    assert extend_initial_joint_state(old,['head_yaw','right_arm','waist','head_pitch','left_arm'],added_head_positions={'head_yaw':0,'head_pitch':0})==(0.,-.3,.1,0.,.4)
    with pytest.raises(ValueError):extend_initial_joint_state(old,['waist','left_arm'],added_head_positions={})
    with pytest.raises(ValueError):extend_initial_joint_state(old,[*old,'new_arm'],added_head_positions={'new_arm':0})


def test_generalized_replay_preserves_free_joint_and_velocity_blocks():
    original = {'object': {'qpos': (1, 2, 3, 1, 0, 0, 0),
                           'qvel': (.1, .2, .3, .4, .5, .6)},
                'arm': {'qpos': (.7,), 'qvel': (.8,)}}
    result = extend_generalized_joint_state(original,
        [('head_yaw', 1, 1), ('object', 7, 6), ('head_pitch', 1, 1), ('arm', 1, 1)],
        added_head_positions={'head_yaw': .2, 'head_pitch': -.1})
    assert result['qpos'] == (.2, 1, 2, 3, 1, 0, 0, 0, -.1, .7)
    assert result['qvel'] == (0, .1, .2, .3, .4, .5, .6, 0, .8)
    assert original['arm']['qvel'] == (.8,)


def test_generalized_replay_rejects_topology_change_or_lost_velocity():
    old = {'object': {'qpos': (1, 2, 3, 1, 0, 0, 0), 'qvel': (0,) * 6}}
    with pytest.raises(ValueError):
        extend_generalized_joint_state(old, [('object', 7, 7)], added_head_positions={})
    with pytest.raises(ValueError):
        extend_generalized_joint_state({'arm': {'qpos': (0,)}}, [('arm', 1, 1)],
                                       added_head_positions={})
    with pytest.raises(ValueError):
        extend_generalized_joint_state({}, [('head_yaw', 7, 6)],
                                       added_head_positions={'head_yaw': 0})


def head_extension_sources(prefix=''):
    root=ET.Element('mujoco');ET.SubElement(root,'compiler',angle='radian')
    world=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator')
    arm=ET.SubElement(world,'body',name=prefix+'arm')
    ET.SubElement(arm,'joint',name=prefix+'arm_joint')
    ET.SubElement(act,'motor',name=prefix+'arm_motor',joint=prefix+'arm_joint')
    sensor=ET.SubElement(root,'sensor')
    for n in ('yaw','roll','pitch'):
        name=prefix+'head_'+n;body=ET.SubElement(world,'body',name=name,pos='0 0 1')
        ET.SubElement(body,'joint',name=name,axis='0 0 1',range='-1 1')
        ET.SubElement(act,'motor',name=name+'_motor',joint=name)
        ET.SubElement(sensor,'jointpos',name=name+'_sensor',joint=name)
    reference=ET.tostring(root,encoding='unicode')
    for parent in root.iter():
        for child in list(parent):
            if child.tag=='joint' and 'head_' in child.get('name','') or 'head_' in child.get('joint',''):
                parent.remove(child)
    return ET.tostring(root,encoding='unicode'),reference


def test_restore_only_head_preserves_arm_and_source_and_rejects_second_application():
    task,ref=head_extension_sources('robot0_')
    result,added=restore_head_actuation(task,ref,naming_prefix='robot0_')
    assert len(added)==3
    actual=ET.fromstring(result)
    assert actual.find('./actuator')[0].attrib=={'name':'robot0_arm_motor','joint':'robot0_arm_joint'}
    assert len(actual.findall('./sensor/*'))==3
    assert len(ET.fromstring(task).findall('.//joint'))==1
    with pytest.raises(ValueError):restore_head_actuation(result,ref,naming_prefix='robot0_')


@pytest.mark.parametrize('failure',['mount','prefix','coupling','actuator'])
def test_restore_rejects_ambiguous_source_contract(failure):
    task,ref=head_extension_sources()
    if failure=='mount':task=task.replace('pos="0 0 1"','pos="0 0 2"')
    elif failure=='coupling':ref=ref.replace('</mujoco>','<equality><joint joint1="head_yaw" joint2="arm_joint"/></equality></mujoco>')
    elif failure=='actuator':task=task.replace('</actuator>','<motor name="head_yaw_motor" joint="arm_joint"/></actuator>')
    with pytest.raises(ValueError):restore_head_actuation(task,ref,naming_prefix='robot0_' if failure=='prefix' else '')
