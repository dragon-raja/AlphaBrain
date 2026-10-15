import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.local_view_derivative import analyze_probes
from AlphaBrain.research.dsol.acquisition.image_geometry import image_motion_jacobian
from AlphaBrain.research.dsol.analysis.local_view_derivative import (
    descending_axis, local_noise_controls, paired_score_matrix,
)


def test_mirror_scale_chain_and_downhill_selection():
    y, x = np.indices((20, 24))
    rgb = np.stack((x+2*y, x*x/10, y*y/12), axis=-1).astype(float)
    depth = np.full((20, 24), 1.2)
    k = np.array([[30,0,11.5],[0,30,9.5],[0,0,1]])
    jac, _ = image_motion_jacobian(rgb, depth, k)
    steps = np.array([.001]*3+[.002]*3)
    twists = np.zeros((13,6))
    images = [rgb[:, ::-1]]
    for axis in range(6):
        for sign in (-1, 1):
            twists[len(images), axis] = sign*steps[axis]
            images.append((rgb+sign*steps[axis]*jac[...,axis])[:, ::-1])
    images = np.array(images)
    gradients = np.ones((2,20,24,3))/rgb.size
    scores = np.repeat(images.mean(axis=(1,2,3))[None], 2, axis=0)
    result = analyze_probes(images, depth, k, twists, scores, gradients)
    baseline = result["geometric_predictions"]["all_valid"]
    assert baseline["scaled_direction_cosine"] == pytest.approx(1)
    assert result["image_chain_cosine"] == pytest.approx(1)
    assert baseline["selected_score_change"] < 0
    np.testing.assert_allclose(result["measured_half_difference"], baseline["predicted_half_difference"], atol=1e-12)
    twists[1, 5] = 1
    with pytest.raises(ValueError, match="axis-isolated"):
        analyze_probes(images, depth, k, twists, scores, gradients)


def test_score_matrix_uses_explicit_seed_order_and_rejects_duplicate_or_missing():
    rows=[dict(state_index=0,candidate_index=1,seed=s,probe_index=i,score=s+i/100)
          for s in [41,42] for i in range(13)]
    value=paired_score_matrix(rows,state_index=0,candidate_index=1,seeds=[42,41])
    assert value[0,0]==42 and value[1,12]==pytest.approx(41.12)
    with pytest.raises(ValueError,match="duplicate"):
        paired_score_matrix(rows+[rows[0]],state_index=0,candidate_index=1,seeds=[41,42])
    with pytest.raises(ValueError,match="missing"):
        paired_score_matrix(rows[:-1],state_index=0,candidate_index=1,seeds=[41,42])


def test_control_selection_never_uses_test_outcomes():
    scores=np.ones((8,13))
    scores[:4,2]=.8
    prediction={"geometric_predictions":{"all_valid":{"predicted_half_difference":[-1,0,0,0,0,0]}},
                "image_chain_half_difference":[-1,0,0,0,0,0]}
    first=local_noise_controls(scores,prediction)
    changed=scores.copy()
    changed[4:,2]=10
    changed[4:,7]=-100
    second=local_noise_controls(changed,prediction)
    for name in first:
        assert first[name]["selected_probe_index"]==second[name]["selected_probe_index"]
    assert second["geometry_chain"]["mean_test_change"]==9
    assert second["local_best_with_stay"]["selected_probe_index"]==2
    assert first["stay"]["paired_test_deltas"]==[0]*4
    assert not first["geometry_chain"]["requires_real_neighbor_images"]
    assert first["finite_difference"]["requires_real_neighbor_images"]


def test_zero_gradient_stays_and_local_best_can_reject_all_movements():
    assert descending_axis(np.zeros(6))==0
    scores=np.full((8,13),2.)
    scores[:,0]=1
    prediction={"geometric_predictions":{"all_valid":{"predicted_half_difference":[0]*6}},
                "image_chain_half_difference":[0]*6}
    results=local_noise_controls(scores,prediction)
    assert all(v["selected_probe_index"]==0 for v in results.values())
