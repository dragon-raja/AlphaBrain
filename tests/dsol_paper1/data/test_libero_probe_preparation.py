from copy import deepcopy

import numpy as np
import pytest

from AlphaBrain.research.dsol.data.libero_probe_preparation import array_signature, score_anchor, plus_request


def ranking():
    return dict(status='PASS', ensemble_size=8, ensemble_seeds=list(range(8)),
                candidate_count=2, checkpoint_action_horizon=10,
                ranking=[dict(candidate_id='b', member_accel_3=[1.]*8, mean_accel_3=1.),
                         dict(candidate_id='a', member_accel_3=[2.]*8, mean_accel_3=2.)])


def test_reconstruct_catalog_order_and_original_shared_noise():
    scores, noise = score_anchor(ranking(), ['a', 'b'])
    assert scores[:, 0].tolist() == [2., 1.]
    assert noise.shape == (8, 10, 7)
    assert noise.dtype == np.float32
    assert np.array_equal(noise[3], np.random.default_rng(3).standard_normal((1,10,7), dtype=np.float32)[0])
    assert array_signature(noise)['shape'] == [8, 10, 7]


@pytest.mark.parametrize('change', [
    lambda x: x['ranking'].pop(),
    lambda x: x['ranking'][0].update(mean_accel_3=0.),
    lambda x: x['ranking'][0].update(member_accel_3=[float('nan')]*8),
    lambda x: x.update(ensemble_seeds=[0]*8),
    lambda x: x['ranking'][0].update(candidate_id='a')])
def test_invalid_anchors_fail_closed(change):
    value=ranking(); change(value)
    with pytest.raises(ValueError): score_anchor(value, ['a','b'])


def test_plus_virtual_bddl_distinct_from_actual_base_file():
    camera=dict(suite='libero_goal', camera_task='task_view_10_20_90_0_0_initstate_0',
                base_task='task', camera_parameters=dict(horizon=10,vertical=20,distance_scale=.9,
                                                       endpoint_yaw=0,endpoint_pitch=0))
    initial=dict(task_id='task',array_sha256='digest',array_path='/initial.npy',index=3)
    result=plus_request(camera,initial,'/runtime')
    assert result['resolved_base_bddl'].endswith('/task.bddl')
    assert result['wrapper_bddl_argument'].endswith('_initstate_0.bddl')
    assert result['physical_initial']['index']==3
    assert not result['policy_results_reusable']
    changed=deepcopy(camera); changed['camera_task']=changed['camera_task'][:-1]+'1'
    with pytest.raises(ValueError): plus_request(changed,initial,'/runtime')
    changed=deepcopy(camera); changed['camera_parameters']['distance_scale']=1.
    with pytest.raises(ValueError): plus_request(changed,initial,'/runtime')
