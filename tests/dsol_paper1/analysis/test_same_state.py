import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.same_state import factorial,summarize


def test_state_view_and_interaction_are_separated():
    # y = state + view + state*view, using +/-1 coding.
    y=np.array([[9.,3.],[-1.,1.]])
    r=factorial(y)
    assert r['state']==6 and r['view']==2 and r['interaction']==8
    assert r['diagonal']==r['state']+r['view']
    assert r['view_at_good_state']==6 and r['view_at_bad_state']==-2


def test_missing_cells_are_not_assumed_failures():
    with pytest.raises(ValueError):factorial([[1.,np.nan],[2.,3.]])


def test_sign_consistency_respects_control_noise():
    y=np.tile([[2.,1.],[2.,1.]],(8,1,1))
    assert summarize(y,0.)['view']['consistent_7_of_8']
    assert not summarize(y,.3)['view']['consistent_7_of_8']
