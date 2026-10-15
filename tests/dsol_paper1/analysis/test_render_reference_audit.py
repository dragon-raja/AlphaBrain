import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.render_reference_audit import summarize_pairs


def test_current_repeatability_does_not_imply_historical_agreement():
    historical = np.zeros((256, 256, 3), dtype=np.uint8)
    current = historical.copy()
    current[0, :8] = 2
    assert summarize_pairs([historical], [current])['failures'] == 1
    assert summarize_pairs([current], [current.copy()])['failures'] == 0


def test_every_pair_is_checked_not_best_retry():
    a = np.zeros((256, 256, 3), dtype=np.uint8)
    bad = a.copy()
    bad[0, 0] = 3
    result = summarize_pairs([a, bad], [a, a])
    assert result['comparisons'] == 4
    assert result['failures'] == 2


def test_empty_group_rejected():
    with pytest.raises(ValueError):
        summarize_pairs([], [np.zeros((2, 2, 3), dtype=np.uint8)])
