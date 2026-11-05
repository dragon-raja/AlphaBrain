from copy import deepcopy

import pytest

from AlphaBrain.research.dsol.analysis.core_decision import contrast, summarize


def panel():
    return [dict(seed=s, suite='suite', task=str(t), state_key=f'{t}-{i}', view_id=v,
                 repeat=0, split='validation', success=bool((t+i+v) % 2),
                 queries=3, query_cap=8, horizon=100, replan_steps=5, settle_steps=10,
                 angles_deg=[v, 0], execution_noise_sha256=['same'],
                 search_initial_accel=1., search_best_accel=.5)
            for s in range(3) for t in range(2) for i in range(2) for v in range(2)]


def test_paired_seed_order_and_identity():
    a = panel()
    result = contrast(a, list(reversed(a)), draws=100)
    assert result['mean_difference'] == 0
    assert result['crossed_seed_task_initial_ci95_sensitivity'] == [0, 0]
    assert result['rescues'] == result['harms'] == 0


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'noise', 'horizon', 'test'])
def test_reject_unpaired_or_test_evidence(mutation):
    a = panel()
    b = deepcopy(a)
    if mutation == 'missing':
        b.pop()
    elif mutation == 'duplicate':
        b.append(b[0])
    elif mutation == 'noise':
        b[0]['execution_noise_sha256'] = ['different']
    elif mutation == 'horizon':
        b[0]['horizon'] += 1
    else:
        b[0]['split'] = 'test'
    with pytest.raises(ValueError):
        contrast(a, b, draws=100)


def test_query_cap_is_not_hardcoded_to_eight():
    rows = panel()
    for r in rows:
        r.update(query_cap=16, queries=16)
    assert summarize(rows)['cap_fraction'] == 1
    assert summarize(rows)['mean_queries'] == 16
