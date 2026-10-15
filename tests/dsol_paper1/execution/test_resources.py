import pytest

from AlphaBrain.research.dsol.execution.resources import choose_capacity, training_allocation


def test_capacity_requires_quality_headroom_and_material_speed():
    rows = [dict(workers=n, status='PASS', quality_pass=True, throughput=speed,
                 headroom_mib=free, required_headroom_mib=2048)
            for n, speed, free in [(1, 10, 20000), (2, 19, 10000), (3, 20, 800)]]
    assert choose_capacity(rows) == 2
    rows[2]['headroom_mib'] = 3000
    assert choose_capacity(rows) == 3
    rows[2]['quality_pass'] = False
    assert choose_capacity(rows) == 2
    rows[1]['throughput'] = 10.1
    assert choose_capacity(rows) == 1
    assert choose_capacity([]) is None


def test_training_is_exclusive_gang_not_multiworker():
    devices = [dict(host='fresh', gpu='0'), dict(host='gnu', gpu='0')]
    assert training_allocation(devices, world_size=2)['ranks'][1]['rank'] == 1
    with pytest.raises(ValueError):
        training_allocation(devices + devices, world_size=4)
    with pytest.raises(ValueError):
        training_allocation(devices, world_size=3)
