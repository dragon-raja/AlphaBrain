import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.temporal_view_diagnostic import cross_noise_choices


def test_each_split_uses_only_its_score_repeats():
    a = np.array([[[0, 0, 8, 8], [8, 8, 0, 0]]], dtype=float)
    y = np.array([[[1, 1, 0, 0], [0, 0, 1, 1]]])
    r = cross_noise_choices(a, y)
    assert len(r['splits']) == 6
    for k,s in enumerate(r['splits']):
        assert set(s['score_repeats']).isdisjoint(s['evaluate_repeats'])
        winner = a[...,s['score_repeats']].mean(-1).argmin(1)
        np.testing.assert_array_equal(r['choices'][k], winner)
        np.testing.assert_equal(r['values'][k,0], y[0,winner[0],s['evaluate_repeats']].mean())
    # The two strongly noise-dependent winners fail on the opposite repeats.
    assert r['values'][0,0] == r['values'][-1,0] == 0
    np.testing.assert_allclose(r['per_initial'], r['values'].mean(0))


def test_outcome_labels_do_not_change_choices():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3,7,4)); y = rng.integers(2,size=a.shape)
    first,second = cross_noise_choices(a,y),cross_noise_choices(a,1-y)
    np.testing.assert_array_equal(first['choices'],second['choices'])
    np.testing.assert_allclose(first['per_initial']+second['per_initial'],1)


@pytest.mark.parametrize('bad', ['shape', 'nan', 'nonbinary'])
def test_invalid_arrays_fail(bad):
    a=np.zeros((2,3,4));y=np.zeros_like(a)
    if bad=='shape': a=a[...,:3]
    if bad=='nan': a[0,0,0]=np.nan
    if bad=='nonbinary': y[0,0,0]=.5
    with pytest.raises(ValueError): cross_noise_choices(a,y)
