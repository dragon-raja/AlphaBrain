import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.recovery_geometry import rays, distance, summarize


def record():
    return dict(state_key='state',task='task',split='unseen_test',method='B',view_id=1,
                start_angles=[30,0],selected_angles=[10.25,2.5],success=[True,False],
                stay_success=[False,False],nominal_success=[True,True])


def test_optical_directions_match_pan_tilt_signs():
    v=rays([[0,0],[30,0],[0,20]])
    np.testing.assert_allclose(v[0],[0,1,0])
    np.testing.assert_allclose(np.linalg.norm(v,axis=1),1)
    assert v[1,0]>0 and v[2,2]>0
    np.testing.assert_allclose(distance([[30,0],[0,20]]),[30,20])


def test_endpoint_and_nominal_counterfactual_are_distinct():
    s=summarize([record()],[[0,0],[30,0]]);g=s['groups']['unseen_test/B']
    assert s['nominal_start_searches']==0 and g['off_panel_fraction']==1
    assert g['closer_fraction']==1 and g['rescue_vs_stay']==1
    assert g['loss_vs_nominal_pairs']==1 and g['gain_vs_nominal_pairs']==0


@pytest.mark.parametrize('change',[{'selected_angles':[31,0]},{'selected_angles':[float('nan'),0]},
                                  {'success':[True]},{'success':[1,False]},{'start_angles':[20,0]}])
def test_invalid_evidence_rejected(change):
    r=record();r.update(change)
    with pytest.raises(ValueError):summarize([r],[[0,0],[30,0]])


def test_duplicate_search_rejected():
    with pytest.raises(ValueError):summarize([record(),record()],[[0,0],[30,0]])
