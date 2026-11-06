"""Presentation must not silently omit conditions or change paired results."""
from copy import deepcopy

import pytest

from AlphaBrain.research.dsol.analysis.core_review import verify_presentation


def evidence():
    rows = [dict(panel=p, seed=s, suite='suite', task='task', state_key=i,
                 view_id=0, repeat=0, split='validation', success=bool(i),
                 stay_success=False, nominal_success=True, queries=2, query_cap=8,
                 search_initial_accel=1., search_best_accel=.4)
            for p in ['a', 'b'] for s in range(2) for i in range(2)]
    values = dict(operations=4, successes=2, success_rate=.5, seed_rates=[.5, .5],
                  seed_sd=0., mean_queries=2., initial_only_fraction=0., cap_fraction=0.,
                  mean_initial_accel=1., mean_best_accel=.4, rescues=2, harms=0,
                  stay_rate=0., reference_rate=1.)
    summary = dict(summaries={p: deepcopy(values) for p in ['a', 'b']}, comparisons={
        'a minus b': dict(mean_difference=0., query_difference=0., rescues=0, harms=0)})
    return rows, summary


def test_complete_verified_presentation():
    rows, summary = evidence()
    result = verify_presentation(list(reversed(rows)), summary)
    assert result['panels'] == 2
    assert result['outcome_rows'] == 8
    assert result['new_model_calls'] == 0


@pytest.mark.parametrize('error', ['omission', 'success', 'duplicate', 'identity', 'baseline', 'contrast'])
def test_reject_changed_or_incomplete_evidence(error):
    rows, summary = evidence()
    if error == 'omission':
        rows = rows[:4]
    elif error == 'success':
        summary['summaries']['a']['success_rate'] = .75
    elif error == 'duplicate':
        rows[1] = deepcopy(rows[0])
    elif error == 'identity':
        rows[0]['state_key'] = 'wrong-state'
    elif error == 'baseline':
        rows[0]['stay_success'] = True
    else:
        summary['comparisons']['a minus b']['rescues'] = 1
    with pytest.raises(ValueError):
        verify_presentation(rows, summary)
