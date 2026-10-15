import pickle
import pytest
import numpy as np
from benchmarks.RoboTwin.replay import load_numpy_pickle


def test_native_arrays_only(tmp_path):
    data = {'left_joint_path': [{'position': np.zeros((4, 6)), 'status': 'Success'}],
            'right_joint_path': []}
    path = tmp_path / 'trajectory.pkl'
    path.write_bytes(pickle.dumps(data, protocol=4))
    restored = load_numpy_pickle(path)
    np.testing.assert_array_equal(restored['left_joint_path'][0]['position'], np.zeros((4, 6)))


def test_arbitrary_pickle_global_rejected(tmp_path):
    path = tmp_path / 'bad.pkl'
    path.write_bytes(pickle.dumps(PathForTest))
    with pytest.raises(ValueError, match='Unapproved'):
        load_numpy_pickle(path)


class PathForTest:
    pass


def test_renderer_errors_not_masked_by_process_success():
    from benchmarks.RoboTwin.replay_admission import render_errors
    log = '[svulkan2] [error] OIDN Error: invalid handle\nexit_code 0\n'
    assert render_errors(log) == ['[svulkan2] [error] OIDN Error: invalid handle']
    assert render_errors('missing pytorch3d\nWarp DeprecationWarning: test') == []


@pytest.mark.parametrize('renderer_log,admitted', [('', True), ('[svulkan2] [error] OIDN Error', False)])
def test_postflight_separates_joint_and_render_admission(tmp_path, renderer_log, admitted):
    import json
    h5py = pytest.importorskip('h5py')
    cv2 = pytest.importorskip('cv2')
    Image = pytest.importorskip('PIL.Image')
    from AlphaBrain.research.dsol.data.resource_import import digest
    from benchmarks.RoboTwin.replay_admission import audit_attempt
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    raw = tmp_path / 'episode.hdf5'
    pixels = np.full((3, 4, 3), 90, dtype=np.uint8)
    joints = np.zeros((1, 14))
    cameras = ('head_camera', 'left_camera', 'right_camera')
    with h5py.File(raw, 'w') as file:
        file['joint_action/vector'] = joints
        blob = cv2.imencode('.png', pixels)[1].tobytes()
        for name in cameras:
            file.create_dataset(f'observation/{name}/rgb', data=[blob], dtype=f'S{len(blob)}')
            Image.fromarray(pixels).save(attempt / f'canonical-{name}.png')
    np.save(attempt / 'recorded_joint_states.npy', joints)
    views = [{'label': label, 'wrist_extrinsics_unchanged': True,
              'pixel_mae_from_first': {name: int(label == 'translate_x_3cm' and name == 'head_camera')
                                       for name in cameras}}
             for label in ('canonical', 'repeat', 'translate_x_3cm', 'restored')]
    state = {'status': 'COMPLETE', 'denoiser': 'oidn', 'task': 'test', 'task_success': True,
             'raw_inputs': {str(raw): digest(raw)}, 'trajectory_consumption': {'left': [1, 1]},
             'camera_probe': {'observable_physics_unchanged': True, 'views': views}}
    (attempt / 'status.json').write_text(json.dumps(state))
    log = tmp_path / 'run.log'
    log.write_text(renderer_log)
    report = audit_attempt(attempt, log)
    assert report['physics_replay_verified']
    assert report['admitted'] is admitted


def test_replay_cli_does_not_start_gpu(tmp_path, monkeypatch, capsys):
    from scripts.dsol_paper1.runtime.robotwin_replay_probe import main
    monkeypatch.setattr('sys.argv', ['probe', '--config', 'absent.json', '--task', 'adjust_bottle',
                                    '--output', str(tmp_path / 'out')])
    main()
    assert not (tmp_path / 'out').exists()
    assert '"execute": false' in capsys.readouterr().out
