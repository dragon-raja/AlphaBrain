from __future__ import annotations

import numpy as np
import pytest

from validate_libero_plus_ray_alignment import policy_centroid


def test_policy_centroid_applies_raw_rot180() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 2] = True

    np.testing.assert_allclose(policy_centroid(mask), [3.0, 2.0])


def test_policy_centroid_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        policy_centroid(np.zeros((3, 3), dtype=bool))
