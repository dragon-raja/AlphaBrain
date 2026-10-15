import numpy as np
from AlphaBrain.research.dsol.analysis.window_decision_audit import report


def test_execution_half_selection_never_uses_test_half_labels():
    s=np.ones((3,8,5));s[0]=.5;s[1,:,0]=.8;s[1,:,1:]=0
    y=np.array([[1.,1.,0.,0.],[0.,0.,1.,1.],[0.,0.,0.,0.]])
    r=report([s],[y],[dict(task='x',split='all')],False,dict(horizon=3,required_noise_votes=7))
    d=r['two_expert_noise_diagnostic']
    assert d['w1']==d['w3']==.5
    assert d['cross_half']==0.
