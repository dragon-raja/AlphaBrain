import numpy as np
import pytest
from AlphaBrain.research.dsol.metrics.action_structure import action_structure,ordinal_concordance,outcome_label


def test_original_clipping_and_zero_hold_are_preserved():
    a=np.zeros((5,7));a[:,6]=[-2,-1,0,1,-1]
    f=action_structure(a,5)
    assert f['close_fraction']==.2
    assert f['gripper_switch']==.25
    assert f['gripper_variation']==1.
    assert np.isnan(f['rotation_reversal'])


def test_far_prediction_cannot_change_near_structure():
    a=np.zeros((10,7));b=a.copy();b[5:]=1
    assert action_structure(a,5)['close_fraction']==action_structure(b,5)['close_fraction']
    assert action_structure(b,10)['close_fraction']==.5


def test_rotation_command_variation_and_valid_direction():
    a=np.zeros((3,7));a[:,3]=[1,-1,0]
    f=action_structure(a,3)
    assert f['rotation_magnitude']==pytest.approx(2/3)
    assert f['rotation_variation']==1.5
    assert f['rotation_reversal']==1.


def test_concordance_ties_and_missing_never_become_failures():
    assert ordinal_concordance([0,1],[0,1])==.5
    assert ordinal_concordance([2],[0,1])==1.
    assert ordinal_concordance([], [1]) is None
    assert outcome_label(None)=='unknown'
    assert outcome_label(float('nan'))=='unknown'


def test_invalid_input_rejected():
    with pytest.raises(ValueError):action_structure(np.zeros((5,12)),5)
