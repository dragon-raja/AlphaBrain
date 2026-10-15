import numpy as np

from AlphaBrain.research.dsol.analysis.window_transitions import transition, aggregate


def test_later_scores_can_reliably_select_worse_endpoint():
    s=np.ones((2,8,5));s[0,:,0]=0;s[0,:,1:]=4
    y=np.array([[1,1,1,1],[0,0,0,0]])
    r=transition(s,y,['a','b'],1,3)
    assert r['outcome']=='degraded' and r['delta']==-1
    assert r['scoring_noise_support']==8 and r['scoring_halves_same_winner']
    assert r['added_stages_outweigh_initial_disadvantage']


def test_missing_is_not_failure_or_complete_population():
    s=np.ones((3,8,5));s[1,:,0]=0;s[1,:,1:]=4
    y=np.array([[1,1],[1,1],[np.nan,np.nan]])
    r=transition(s,y,['a','b','c'],1,3,True)
    a=aggregate([dict(task='t',**r)])
    assert r['outcome']=='missing' and a['delta'] is None and a['coverage']==0


def test_same_view_has_exactly_zero_effect():
    s=np.ones((2,8,5));s[0]=0
    r=transition(s,np.array([[1,0,1,0],[0,0,0,0]]),['a','b'],3,5)
    assert not r['switched'] and r['delta']==0


def test_paired_execution_halves_opposite_not_stable():
    s=np.ones((2,8,5));s[0,:,0]=0;s[0,:,1:]=4
    r=transition(s,np.array([[0,0,1,1],[1,1,0,0]]),['a','b'],1,3)
    assert r['execution_half_deltas']==[1,-1]
    assert not r['both_execution_halves_same_direction']
