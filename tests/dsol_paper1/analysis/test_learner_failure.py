import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.learner_failure import select_visited, gap_decomposition, split_signs


def test_ties_retain_first_visit_not_smallest_candidate():
    assert select_visited([0,0,1],np.array([[1,0,2]])).tolist()==[1]


def test_cross_evaluation_keeps_negative_gap_instead_of_clipping():
    labels=np.array([[1,1,0,0],[0,0,1,1]])
    d=gap_decomposition(labels,np.array([[0,1]]),np.array([0]))
    assert d['empirical_ranking_gap'][0]==0
    assert d['cross_ranking_gap'][0]==-.5
    assert d['cross_search_gap'][0]==0


def test_search_miss_and_wrong_ranking_are_separate():
    labels=np.array([[0,0,0,0],[1,0,1,0],[1,1,1,1]])
    d=gap_decomposition(labels,np.array([[0,1]]),np.array([0]))
    assert d['empirical_search_gap'][0]==.5
    assert d['empirical_ranking_gap'][0]==.5
    assert d['cross_search_gap'][0]==.5
    assert d['cross_ranking_gap'][0]==.5


def test_missing_labels_fail_instead_of_becoming_failures():
    with pytest.raises(ValueError):gap_decomposition([[1,np.nan]],[[0]],[0])


def test_ties_and_reversals_remain_distinct():
    labels=np.array([[0,0,1,1],[1,1,0,0],[0,0,0,0]])
    x=split_signs(labels,np.array([1,2,0]),np.array([0,2,2]))
    assert x['strict_reversal'].tolist()==[True,False,False]
    assert x['equal_both'].tolist()==[False,True,False]
    assert x['positive_both'].tolist()==[False,False,False]
