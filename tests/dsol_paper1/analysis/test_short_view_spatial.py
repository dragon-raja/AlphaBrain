import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.short_view_spatial import spatial_statistics


def test_spatial_excludes_canonical_but_selector_includes_it():
    a=np.broadcast_to(np.arange(97)[None,:,None],(2,97,5)).copy().astype(float)
    q=np.broadcast_to(np.linspace(1,0,97),(2,97)).copy();q[:,0]=.25
    rows=spatial_statistics(a,q)
    assert rows[0]['selected_success']==.25
    assert rows[0]['association']['spatial_mean_spearman']==pytest.approx(1)
    assert rows[0]['association']['pooled_within_initial_rank_correlation']==pytest.approx(1)
    assert all(x>y for x,y in zip(rows[0]['quartile_success'],rows[0]['quartile_success'][1:]))


def test_within_initial_rank_is_invariant_to_positive_affine_rescaling():
    rng=np.random.default_rng(1);a=rng.random((4,97,5));q=rng.random((4,97))
    b=a*np.array([1,7,.3,12])[:,None,None]+np.array([5,0,2,8])[:,None,None]
    first,second=spatial_statistics(a,q),spatial_statistics(b,q)
    for x,y in zip(first,second):
        assert x['association']['pooled_within_initial_rank_correlation']==pytest.approx(y['association']['pooled_within_initial_rank_correlation'])
        assert x['quartile_success']==y['quartile_success']


def test_missing_values_rejected():
    a=np.ones((2,97,5));a[0,1,0]=np.nan
    with pytest.raises(ValueError):spatial_statistics(a,np.ones((2,97)))
