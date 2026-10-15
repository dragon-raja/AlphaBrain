import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.action_sensitivity import action_difference


def test_action_comparison_detects_discrete_flips():
    a = np.zeros((50, 12)); b = a.copy()
    a[:, 6] = .499; b[:, 6] = .501
    r = action_difference(a, b)
    assert r['gripper_threshold_flips'] == 50
    assert r['control_mode_threshold_flips'] == 0
    assert action_difference(a, a)['max_abs'] == 0


def test_action_comparison_rejects_wrong_shape():
    with pytest.raises(ValueError):
        action_difference(np.zeros((50, 12)), np.zeros((49, 12)))
