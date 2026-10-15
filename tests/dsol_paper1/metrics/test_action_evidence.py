import numpy as np
import pytest

from AlphaBrain.research.dsol.metrics.action_evidence import flow_components, libero_commands


def test_constant_flow_can_have_nonzero_actions_and_zero_accel():
    raw,_=flow_components(np.ones((8,5,10,10,7)),'libero')
    assert np.all(raw['score']==0)
    commands=libero_commands(np.ones((8,5,10,7)))
    assert np.allclose(commands['translation_command_norm'],np.sqrt(3))
    assert commands['overlapping_replan_translation_change']==[None,0,0,0,0]


def test_flow_ratio_is_not_absolute_motion():
    v=np.random.default_rng(0).normal(size=(8,5,10,10,7))
    a,_=flow_components(v,'libero');b,_=flow_components(10*v,'libero')
    assert np.allclose(a['score'],b['score'])
    assert np.allclose(10*a['denominator'],b['denominator'])


def test_unexecuted_tail_does_not_enter_command_magnitude():
    a=np.zeros((8,5,10,7));a[:,:,5:,:3]=20
    r=libero_commands(a)
    assert r['translation_command_norm']==[0]*5
    assert np.allclose(r['overlapping_replan_translation_change'][1:],np.sqrt(3))


def test_invalid_commands_rejected():
    with pytest.raises(ValueError):libero_commands(np.zeros((8,5,25,12)))
