import numpy as np
import pytest
from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
from AlphaBrain.research.dsol.analysis.coordinate_mechanism import mounting_intervention, aggregate, endpoint_response


def test_yaw_mount_changes_preserve_common_physical_path():
    old = PanTiltMount(np.eye(4))
    rng = np.random.default_rng(9)
    for offset in [-5., 5.]:
        shifted = PanTiltMount(old.pose([offset, 0.]))
        for _ in range(300):
            q = rng.uniform([-20, -20], [20, 20])
            np.testing.assert_allclose(shifted.pose(q-[offset, 0.]), old.pose(q), atol=1e-14, rtol=0)


def test_only_mechanical_features_change_and_edge_cases_rejected():
    q, b = np.array([5., -3.]), np.array([30., 20.])
    obs = dict(joint_state=np.concatenate((q/b, (q+b)/(2*b), (b-q)/(2*b))).astype(np.float32),
               delta_low=np.array([-10., -10.]), delta_high=np.array([10., 10.]),
               visual=np.arange(10), scores=np.array([.01, .002]), geometry=np.eye(3))
    zero = mounting_intervention(obs, q, 0.)
    changed = mounting_intervention(obs, q, 5.)
    for key in obs:
        np.testing.assert_array_equal(obs[key], zero[key])
        if key != 'joint_state': np.testing.assert_array_equal(obs[key], changed[key])
    assert not np.array_equal(obs['joint_state'], changed['joint_state'])
    with pytest.raises(ValueError): mounting_intervention(obs, [25., 0.], 5.)
    with pytest.raises(ValueError): mounting_intervention(obs, q, 26.)


def test_response_summary_keeps_conditional_action_separate_from_stop():
    rows=[dict(seed=0,task='t',source_episode=0,mount_yaw_deg=b,
               conditional_delta_deg=[b,0],stop=False,stop_probability=.1) for b in [-5.,0.,5.]]
    s=aggregate(rows)['0']
    assert s['median_yaw_response_per_mount_degree']==1
    assert s['deterministic_stop_flip_percent']==0
    with pytest.raises(ValueError): aggregate(rows[:2])
    with pytest.raises(ValueError): aggregate(rows+rows[:1])


@pytest.mark.parametrize('follow_mount',[False,True])
def test_endpoint_response_distinguishes_physical_invariance_from_mount_following(follow_mount):
    rows=[]
    for offset in (-5.,0.,5.):
        yaw=offset if follow_mount else 0.
        rows.append(dict(state_key='s',view_id=1,method=f'G-seed-0-mount-{offset:+g}',
            mount_yaw_offset_deg=offset,selected_angles=[yaw,0.],mechanical_selected_angles=[yaw-offset,0.]))
    result=endpoint_response(rows)['G-seed-0']
    assert result['median_yaw_response_per_mount_degree']==int(follow_mount)
    assert result['median_endpoint_shift_joint_degrees']==5*int(follow_mount)
    with pytest.raises(ValueError):endpoint_response(rows[:2])
    with pytest.raises(ValueError):endpoint_response(rows+rows[:1])
