import pytest

from AlphaBrain.research.dsol.analysis.runtime_capacity import host_capacity, population_work


def test_capacity_separates_cold_start_and_resident_model():
    row = dict(status='PASS', seconds=20., rollout=dict(seconds=10., steps=100))
    supervisor = dict(status='COMPLETE', returncode=0, started_unix=0., finished_unix=20.)
    host = host_capacity([row, row], [supervisor, supervisor], 2)
    assert host['modeled_resident_steps_per_second'] == 20.
    assert host['observed_cold_batch_episodes_per_hour'] == 360.
    result = population_work([host, host], [100] * 4, 3, 2, 8)
    assert result['episodes'] == 192
    assert result['modeled_resident_days'] == pytest.approx(480 / 86400)
    assert result['cold_probe_pattern_days'] == pytest.approx(192 / 720 / 24)


def test_failed_probe_is_not_silently_averaged():
    with pytest.raises(ValueError):
        host_capacity([dict(status='ERROR')], [dict(status='COMPLETE')], 1)
