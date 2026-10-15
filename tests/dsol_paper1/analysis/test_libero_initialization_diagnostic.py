from copy import deepcopy

import pytest

from AlphaBrain.research.dsol.analysis.libero_initialization_diagnostic import PAIR_KEYS, validate_pair


def example():
    row = {key: 'fixed' for key in PAIR_KEYS}
    row.update(status='complete', explicit_flow_noise=True, success=False,
               max_steps=10, completion_steps=10, inference_calls=2,
               policy_calls=[dict(noise_sha256='noise0', replan_index=0),
                             dict(noise_sha256='noise1', replan_index=1)])
    return row


def test_success_can_end_earlier_without_requiring_same_replan_count():
    old = example()
    new = deepcopy(old)
    new.update(success=True, completion_steps=5, inference_calls=1)
    new['policy_calls'] = new['policy_calls'][:1]
    validate_pair(old, new)


@pytest.mark.parametrize('change', ['initialization_receipt', 'noise', 'truncated', 'missing_calls'])
def test_diagnostic_rejects_unmatched_or_incomplete_evidence(change):
    old = example()
    new = deepcopy(old)
    if change == 'initialization_receipt':
        new[change] = 'different'
    elif change == 'noise':
        new['policy_calls'][0]['noise_sha256'] = 'different'
    elif change == 'truncated':
        new['completion_steps'] = 9
    else:
        new['policy_calls'] = []
    with pytest.raises(ValueError):
        validate_pair(old, new)
