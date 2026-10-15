from copy import deepcopy

import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.recovery_confirmation import selected_conditions
from benchmarks.Robocasa365.confirmation_execution import validate_request


def fixture():
    views = ['canonical']+[f'train_{i:03d}' for i in range(64)]+[f'extra_{i:03d}' for i in range(32)]
    selected = selected_conditions(views, views[-32:], np.ones((97, 8, 5)))
    selected.update(model='official', initial='i', all_97_candidates_verified=True, window1_exact=True)
    return dict(initial_ids=['i']), selected


def test_original_scope_and_independent_eight_repeats():
    plan, selected = fixture()
    validate_request(plan, selected, 'official', 'i', selected['evaluation_views'], list(range(8)))


@pytest.mark.parametrize('views,repeats', [(['train_003'], [0]), (['canonical'], [8]),
                                          (['canonical', 'canonical'], [0]), (['canonical'], [0, 0]),
                                          ([], [0]), (['canonical'], [])])
def test_unplanned_duplicate_or_empty_execution_rejected(views, repeats):
    plan, selected = fixture()
    with pytest.raises(ValueError): validate_request(plan, selected, 'official', 'i', views, repeats)


def test_model_or_unverified_selection_rejected():
    plan, selected = fixture()
    with pytest.raises(ValueError): validate_request(plan, selected, 'xr1', 'i', ['canonical'], [0])
    changed = deepcopy(selected); changed['window1_exact'] = False
    with pytest.raises(ValueError): validate_request(plan, changed, 'official', 'i', ['canonical'], [0])
