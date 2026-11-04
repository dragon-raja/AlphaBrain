import json

import pytest

from AlphaBrain.research.dsol.execution.batch_boundary import pause_requested


def request(path, **changes):
    row = dict(schema='PAUSE_AT_COMPLETE_BATCH_V1', experiment_id='run-a',
               config_sha256='a' * 64, after_version=21)
    row.update(changes)
    path.write_text(json.dumps(row))


def check(path, version=21):
    return pause_requested(path, experiment_id='run-a', config_sha256='a' * 64,
                           completed_version=version)


def test_absent_request_does_not_pause(tmp_path):
    assert not check(tmp_path / 'pause.json')


def test_waits_for_whole_requested_batch(tmp_path):
    path = tmp_path / 'pause.json'
    request(path)
    assert not check(path, 20)
    assert check(path, 21)
    assert check(path, 22)
    assert path.exists()  # Requests cannot vanish and silently authorize resume.


@pytest.mark.parametrize('changes', [dict(experiment_id='run-b'),
    dict(config_sha256='b' * 64), dict(after_version=-1),
    dict(after_version=True), dict(schema='GO'), dict(dispatch_enabled=True)])
def test_invalid_or_other_run_request_fails_closed(tmp_path, changes):
    path = tmp_path / 'pause.json'
    request(path, **changes)
    with pytest.raises(ValueError):
        check(path)
