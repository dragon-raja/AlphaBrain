import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.search_design_audit import (
    noise_subset_audit, relative_camera_extent,
)


def test_noise_agreement_known_stable_order():
    members = np.broadcast_to(np.array([1., 2., 3.])[None, :, None], (2, 3, 8))
    result = noise_subset_audit(members, np.ones((2, 3)))
    assert [row["subset_count"] for row in result["subsets"]] == [8, 28, 70]
    assert all(row["same_choice_as_disjoint_complement"] == 1 for row in result["subsets"])
    assert all(row["normalized_score_regret_p50_p95"] == [0., 0.] for row in result["subsets"])


def test_complement_can_disagree_and_labels_do_not_choose():
    scores = np.array([[[1.] * 4 + [3.] * 4, [3.] * 4 + [1.] * 4]])
    a = noise_subset_audit(scores, np.array([[0., 1.]]))
    b = noise_subset_audit(scores, np.array([[1., 0.]]))
    assert a["subsets"][0]["same_choice_as_disjoint_complement"] == 0
    assert a["subsets"][0]["same_choice_as_8"] == b["subsets"][0]["same_choice_as_8"]


def test_camera_distance_and_rotation():
    poses = np.tile(np.eye(4), (1, 2, 1, 1))
    poses[0, 1, 0, 3] = 2.
    poses[0, 1, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    result = relative_camera_extent(poses)
    np.testing.assert_allclose(result["translation_m"], [2.] * 4)
    np.testing.assert_allclose(result["rotation_deg"], [90.] * 4)
    poses[0, 1, 0, 0] = 3
    with pytest.raises(ValueError, match="Rigid"):
        relative_camera_extent(poses)
