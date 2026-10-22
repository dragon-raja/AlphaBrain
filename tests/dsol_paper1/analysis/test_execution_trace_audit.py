import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.execution_trace_audit import first_row_difference, compare_recorded_noise


def test_matching_prefix_is_not_complete_agreement():
    r = first_row_difference(np.zeros((2, 7)), np.zeros((3, 7)))
    assert r['first_different_row'] is None
    assert not r['complete_equal'] and not r['lengths_equal']


def test_first_recorded_difference_and_invalid_schema():
    a = np.zeros((8, 7)); b = a.copy(); b[4, 6] = 1e-8
    assert first_row_difference(a, b)['first_different_row'] == 4
    with pytest.raises(ValueError): first_row_difference(a, np.zeros((8, 8)))
    b[0, 0] = np.nan
    with pytest.raises(ValueError): first_row_difference(a, b)


def test_noise_addresses_and_unequal_episode_lengths():
    old = [dict(replan_index=i, noise_sha256=str(i)) for i in range(3)]
    assert compare_recorded_noise(old, ['0', '1'])['first_mismatch'] is None
    assert compare_recorded_noise(old, ['0', 'x'])['first_mismatch'] == 1
    with pytest.raises(ValueError): compare_recorded_noise(old[1:], ['1'])
