import json
import pytest

h5py = pytest.importorskip('h5py')
np = pytest.importorskip('numpy')
from benchmarks.RoboTwin.data_admission import inspect, screen_joint_limits


def make_demo(root):
    (root / 'data').mkdir()
    (root / '_traj_data').mkdir()
    (root / 'seed.txt').write_text('41\n')
    (root / 'scene_info.json').write_text(json.dumps({'episode_0': {}}))
    (root / '_traj_data/episode0.pkl').write_bytes(b'not unpickled during audit')
    with h5py.File(root / 'data/episode0.hdf5', 'w') as h:
        h.create_dataset('joint_action/vector', data=np.zeros((2, 14)))
        for cam in ('head_camera', 'left_camera', 'right_camera'):
            h.create_dataset(f'observation/{cam}/rgb', data=np.array([b'\xff\xd8x'] * 2, dtype='S3'))


def test_structure_does_not_imply_replay(tmp_path):
    make_demo(tmp_path)
    result = inspect(tmp_path)
    assert result['status'] == 'STRUCTURE_CHECKED'
    assert not result['runtime_replay_verified']
    assert not result['trajectory_pickle_loaded']


def test_missing_trajectory_fails(tmp_path):
    make_demo(tmp_path)
    (tmp_path / '_traj_data/episode0.pkl').unlink()
    result = inspect(tmp_path)
    assert result['status'] == 'FAILED'
    assert result['errors'] == ['Missing trajectory: 0']


def test_joint_limit_screen(tmp_path):
    path = tmp_path / 'robot.urdf'
    path.write_text('<robot><joint name="arm"><limit lower="-1" upper="1"/></joint></robot>')
    result = screen_joint_limits({'episodes': [{'episode': 0, 'joint_min': [-2], 'joint_max': [0]}]}, path, ['arm'])
    assert result['status'] == 'FLAGGED'
    assert not result['realized_state_or_collision_validated']
