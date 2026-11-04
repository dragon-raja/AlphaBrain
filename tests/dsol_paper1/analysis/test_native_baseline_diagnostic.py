"""Pairing and outcome-blind selection on a complete synthetic view surface."""
from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.analysis.native_baseline_diagnostic import analyze


def panel():
    return [dict(state_key='s',task='task',suite='suite',split='validation',view_id=i,
                 yaw_deg=float(i),pitch_deg=0.,success=[i==0,i==0],
                 accel=0. if i==1 else float(i+1),independent_accel=float(i+1)) for i in range(27)]


def test_complete_pairing_preserves_harm_and_does_not_select_success():
    value=analyze(panel())
    assert value['nominal_successes']==2 and value['nominal_episodes']==2
    assert value['perturbed_successes']==0 and value['perturbed_episodes']==52
    assert value['perturbation_harm_pairs']==52 and value['perturbation_improvement_pairs']==0
    assert value['dense']['task_macro']['perturbation_loss']==1.
    for selected in value['offline_selectors'].values():
        assert selected['task_macro']['success']==0. and selected['harm_count']==2
    assert value['per_state'][0]['selected_views']==dict(min_accel_27=1,min_accel_26=1)


def test_changed_outcomes_never_change_score_based_choice():
    rows=panel();before=analyze(rows)
    for row in rows:row['success']=[not x for x in row['success']]
    after=analyze(rows)
    assert before['per_state'][0]['selected_views']==after['per_state'][0]['selected_views']
    assert after['perturbation_improvement_pairs']==52


@pytest.mark.parametrize('change',['missing','duplicate','nonfinite','missing_outcome','integer_success'])
def test_incomplete_or_invalid_panels_fail(change):
    rows=deepcopy(panel())
    if change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]=rows[0]
    elif change=='nonfinite':rows[0]['accel']=float('nan')
    elif change=='missing_outcome':rows[0]['success']=[True]
    else:rows[0]['success']=[1,0]
    with pytest.raises(ValueError):analyze(rows)


def test_constant_outcomes_do_not_invent_correlation():
    rows=panel()
    for row in rows:row['success']=[True,True]
    assert analyze(rows)['correlation']['states_defined']==0
