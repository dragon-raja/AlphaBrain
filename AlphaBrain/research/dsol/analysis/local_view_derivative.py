"""Pure analysis of paired local camera probes; no simulation/model imports."""
from __future__ import annotations

import numpy as np

from AlphaBrain.research.dsol.acquisition.image_geometry import (
    contract_image_gradient, image_motion_jacobian,
)


def cosine(left, right):
    denominator = np.linalg.norm(left)*np.linalg.norm(right)
    return float(np.dot(left, right)/denominator) if denominator > 0 else None


def paired_score_matrix(rows, *, state_index, candidate_index, seeds):
    """Restore all 13 paired probes in explicit seed order, rejecting duplicates."""
    lookup = {}
    for row in rows:
        if (row["state_index"], row["candidate_index"]) != (state_index, candidate_index):
            continue
        key = (row["seed"], row["probe_index"])
        if key in lookup:
            raise ValueError("duplicate score cell")
        lookup[key] = row["score"]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("nonempty unique seeds required")
    try:
        values = np.array([[lookup[(seed, i)] for i in range(13)] for seed in seeds], dtype=float)
    except KeyError as exc:
        raise ValueError("missing score cell") from exc
    if not np.isfinite(values).all():
        raise ValueError("nonfinite scores")
    return values


def descending_axis(half_difference):
    """Scale-normalized coordinate descent; return base if every component is zero."""
    value = np.asarray(half_difference, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("six finite directions required")
    if not np.any(value):
        return 0
    axis = int(np.argmax(np.abs(value)))
    return 1+2*axis if value[axis] > 0 else 2+2*axis


def local_noise_controls(scores, train_prediction, *, split=4):
    """Exploratory diagnostic controls, NOT deployable no-query algorithms.

    Direction/or best-view selection only sees the first noise subset.
    Finite difference and local-best require real neighbouring images.
    Test deltas are paired within the same held-out noise draw.
    """
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 2 or scores.shape[1] != 13
            or not 1 <= split < len(scores) or not np.isfinite(scores).all()):
        raise ValueError("finite [noise,13] scores with nonempty split required")
    train, test = scores[:split].mean(0), scores[split:]
    choices = {
        "geometry_chain": descending_axis(train_prediction["geometric_predictions"]["all_valid"]["predicted_half_difference"]),
        "actual_image_chain": descending_axis(train_prediction["image_chain_half_difference"]),
        "finite_difference": descending_axis((train[2::2]-train[1::2])/2),
        "local_best_with_stay": int(np.argmin(train)),
        "stay": 0,
    }
    output = {}
    for name, selected in choices.items():
        delta = test[:, selected]-test[:, 0]
        output[name] = dict(selected_probe_index=selected,
            paired_test_deltas=delta.tolist(), mean_test_change=float(delta.mean()),
            training_change=float(train[selected]-train[0]),
            requires_real_neighbor_images=name not in {"geometry_chain", "stay"})
    return output


def analyze_probes(policy_images, optical_depth, intrinsics, twists, scores, image_gradients):
    """13 probes: base, then [-,+] for each body axis; scores [noise,13].

    Policy RGB/gradients are horizontal mirrors of optical-upright RGB.
    Scale derivatives by actual probe steps before comparing m/rad axes.
    """
    images = np.asarray(policy_images, dtype=np.float64)
    depth = np.asarray(optical_depth, dtype=np.float64)
    twists = np.asarray(twists, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    gradients = np.asarray(image_gradients, dtype=np.float64)
    if images.ndim != 4 or images.shape[0] != 13 or images.shape[-1] != 3:
        raise ValueError("13 HWC RGB probes required")
    if scores.ndim != 2 or scores.shape[1] != 13 or gradients.shape != (scores.shape[0], *images.shape[1:]):
        raise ValueError("score/gradient shape mismatch")
    if depth.shape != images.shape[1:3] or twists.shape != (13, 6):
        raise ValueError("depth/twist shape mismatch")
    if not all(np.isfinite(a).all() for a in (images, depth, twists, scores, gradients)):
        raise ValueError("nonfinite diagnostic data")
    steps = np.asarray([twists[2+2*i, i] for i in range(6)])
    expected = np.zeros((13, 6))
    for i in range(6):
        expected[1+2*i, i], expected[2+2*i, i] = -steps[i], steps[i]
    if (steps <= 0).any() or not np.allclose(twists, expected, atol=1e-12, rtol=0):
        raise ValueError("probes must be paired axis-isolated perturbations")
    mean_scores = scores.mean(axis=0)
    gradient = gradients.mean(axis=0)[:, ::-1]
    upright = images[0, :, ::-1]
    valid = depth > 0
    valid[:3] = False
    valid[-3:] = False
    valid[:, :3] = False
    valid[:, -3:] = False
    # Conservative 5x5 depth-edge exclusion, fixed before seeing score changes.
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            neighbour = np.roll(depth, (dy, dx), (0, 1))
            valid &= np.abs(depth-neighbour) < .02*depth
    actual = (mean_scores[2::2]-mean_scores[1::2])/2
    actual_pixel_delta = (images[2::2]-images[1::2])/2
    direct_chain = np.einsum("hwc,khwc->k", gradients.mean(axis=0), actual_pixel_delta)
    results = {}
    for name, mask in (("all_valid", None), ("depth_edge_mask", valid)):
        jac, _ = image_motion_jacobian(upright, depth, intrinsics, valid_mask=mask)
        prediction = contract_image_gradient(gradient, jac)*steps
        axis = int(np.argmax(np.abs(prediction)))
        chosen = 1+2*axis if prediction[axis] > 0 else 2+2*axis
        expected_pixel_delta = np.moveaxis(jac*steps, -1, 0)[:, :, ::-1]
        flat_expected = expected_pixel_delta.reshape(6, -1)
        flat_actual = actual_pixel_delta.reshape(6, -1)
        results[name] = dict(predicted_half_difference=prediction.tolist(),
            scaled_direction_cosine=cosine(prediction, actual),
            sign_agreement_count=int(np.sum(prediction*actual > 0)),
            nonzero_actual_axes=int(np.count_nonzero(actual)),
            selected_probe_index=chosen, selected_score=float(mean_scores[chosen]),
            selected_score_change=float(mean_scores[chosen]-mean_scores[0]),
            image_direction_cosines=[cosine(a,b) for a,b in zip(flat_expected,flat_actual)])
    return dict(base_score=float(mean_scores[0]), mean_probe_scores=mean_scores.tolist(),
        measured_half_difference=actual.tolist(), image_chain_half_difference=direct_chain.tolist(),
        image_chain_cosine=cosine(direct_chain, actual),
        depth_edge_retained_fraction=float(valid.mean()), geometric_predictions=results,
        empirical_best_local_probe_score=float(mean_scores[1:].min()),
        note="Paired initial scoring only. No task execution, global minimum or success-rate claim.")
