import numpy as np
import pytest

from AlphaBrain.research.dsol.data.recorded_mocap import mocap_from_body_poses


def recover(mapping=(-1, 1, -1, 0), names=('world', 'a', 'b', 'c'), positions=None):
    positions = np.arange(12.).reshape(4, 3) if positions is None else positions
    return mocap_from_body_poses(np.array(mapping), 2, ('world', 'a', 'b', 'c'), names,
                                positions, np.tile([1., 0., 0., 0.], (4, 1)))


def test_noncontiguous_bodies_and_permuted_slots_follow_current_recorded_pose():
    pos, quat, bodies = recover()
    np.testing.assert_array_equal(bodies, [3, 1])
    np.testing.assert_array_equal(pos, [[9, 10, 11], [3, 4, 5]])
    moved = np.arange(12.).reshape(4, 3) + 7
    np.testing.assert_array_equal(recover(positions=moved)[0], pos+7)
    np.testing.assert_array_equal(quat[:, 0], [1, 1])


@pytest.mark.parametrize('mapping', [(-1, 0, -1, 0), (-1, 2, -1, 0), (-1, -1, -1, 0), (-2, 1, -1, 0)])
def test_incomplete_duplicate_or_invalid_mapping_fails(mapping):
    with pytest.raises(ValueError, match='mapping'):
        recover(mapping=mapping)


def test_identity_shape_and_nonfinite_records_fail():
    with pytest.raises(ValueError, match='identity/order'):
        recover(names=('world', 'c', 'b', 'a'))
    with pytest.raises(ValueError, match='shape'):
        recover(positions=np.zeros((3, 3)))
    with pytest.raises(ValueError, match='Nonfinite'):
        recover(positions=np.full((4, 3), np.nan))
