from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.reward_calibration import fixed_model_scale


def records():
    return [dict(address=str(i),split='train',model_sha256='weights',query_kind='initial',
                 score_members=[x,x+.002]) for i,x in enumerate([.01,.02,.03])]


def test_scale_is_median_of_train_initial_means():
    value=fixed_model_scale(records(),['0','1','2'],model_sha256='weights')
    assert value['scale']==pytest.approx(.021)
    assert value['query_cost']==.01 and not value['observation_scale_changed']


@pytest.mark.parametrize('fault',['heldout','duplicate','missing','model','noninitial','nan'])
def test_refuses_leakage_missing_or_changed_calibration(fault):
    rows=deepcopy(records())
    if fault=='heldout':rows[0]['split']='test'
    if fault=='duplicate':rows.append(rows[0])
    if fault=='missing':rows.pop()
    if fault=='model':rows[0]['model_sha256']='other'
    if fault=='noninitial':rows[0]['query_kind']='selected'
    if fault=='nan':rows[0]['score_members'][0]=float('nan')
    with pytest.raises(ValueError):fixed_model_scale(rows,['0','1','2'],model_sha256='weights')
