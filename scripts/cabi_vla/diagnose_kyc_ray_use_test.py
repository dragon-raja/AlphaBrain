from __future__ import annotations

import numpy as np

from diagnose_kyc_ray_use import action_difference


def test_identical_actions_have_zero_difference() -> None:
    actions = np.ones((20, 7), dtype=np.float32)
    metrics = action_difference(actions, actions.copy())
    assert metrics["chunk_rms"] == 0.0
    assert metrics["first_action_rms"] == 0.0
    assert metrics["max_abs"] == 0.0
    assert np.isclose(metrics["cosine_similarity"], 1.0)


def test_action_difference_reports_first_and_full_chunk() -> None:
    reference = np.zeros((2, 2), dtype=np.float32)
    candidate = np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    metrics = action_difference(reference, candidate)
    assert np.isclose(metrics["chunk_rms"], 1.0)
    assert np.isclose(metrics["first_action_rms"], np.sqrt(2.0))
    assert metrics["max_abs"] == 2.0

