import numpy as np

from AlphaBrain.research.dsol.data.evaluation_matrix import assign_workers, flow_noise


def test_noise_is_addressed_not_consumed_and_streams_are_disjoint():
    a = flow_noise('initial', 'execution', 0, 3)
    flow_noise('another', 'execution', 3, 17)
    np.testing.assert_array_equal(a, flow_noise('initial', 'execution', 0, 3))
    assert a.shape == (50, 32) and a.dtype == np.float32
    assert not np.array_equal(a, flow_noise('initial', 'score', 0, 3))
    assert not np.array_equal(a, flow_noise('initial', 'execution', 1, 3))


def test_full_population_is_balanced_and_models_are_paired():
    jobs = [dict(initial_id=f'{split}-{task}-{seed}', split=split, task=str(task), scene_seed=seed)
            for split in ['seen', 'target'] for task in range(10) for seed in range(20)]
    workers = assign_workers(jobs)
    assert len(workers) == 32
    assert all(len(w['initials']) == 25 for w in workers)
    for job in jobs:
        assigned = [w for w in workers if job['initial_id'] in w['initials']]
        assert len(assigned) == 2
        assert len({(w['host'], w['gpu']) for w in assigned}) == 1
