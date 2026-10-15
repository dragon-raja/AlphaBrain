import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.task_score_attribution import crossover


def test_score_average_precedes_choice_and_execution_is_fixed():
    a=np.array([[[[0.,2.],[1.2,1.2]]],[[[2.,2.],[0.,0.]]]])
    y=np.array([[[[1.,0.],[0.,0.]]],[[[1.,1.],[0.,1.]]]])
    choices,q=crossover(y,a)
    np.testing.assert_array_equal(choices,[[0],[1]])
    np.testing.assert_allclose(q.mean((2,3)),[[.5,1.],[0.,.5]])
    # Changing outcomes must not change scorer choices.
    np.testing.assert_array_equal(crossover(1-y,a)[0],choices)


def test_reject_unmatched_matrices():
    with pytest.raises(ValueError):crossover(np.zeros((2,3,4,4)),np.zeros((2,4,4,8)))
