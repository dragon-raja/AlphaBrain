"""Descriptive noise/geometry audit of verified INITIAL32 inputs, without fitting."""

import itertools
import json
from pathlib import Path

import numpy as np

from ..data.search_replay import load_initial32


def noise_subset_audit(members, outcomes):
    """Enumerate all k-of-8 subsets, not select a favorable random seed.

    The full-eight reference overlaps each subset; neither it nor the disjoint
    complement is ground truth. Subsets are correlated and are not independent
    physical trials. Historical outcome maxima are never used for fitting.
    """
    members, outcomes = np.asarray(members), np.asarray(outcomes)
    if (members.ndim != 3 or members.shape[-1] != 8
            or outcomes.shape != members.shape[:2] or members.shape[0] < 1
            or members.shape[1] < 2 or not np.isfinite(members).all()
            or (members < 0).any() or not np.isfinite(outcomes).all()
            or ((outcomes < 0) | (outcomes > 1)).any()):
        raise ValueError("Finite [initial, candidate, 8] scores and outcome means required")
    reference = members.mean(-1)
    normalizer = np.median(reference, axis=1)
    if (normalizer <= 0).any():
        raise ValueError("Positive within-initial score median required")
    best = reference.argmin(1)
    indices = np.arange(len(best))
    result = {"reference_8_success": float(outcomes[indices, best].mean()), "subsets": []}
    for count in (1, 2, 4):
        agree, complement_agree, regret, successes = [], [], [], []
        for subset in itertools.combinations(range(8), count):
            selected = members[..., list(subset)].mean(-1).argmin(1)
            complement = [j for j in range(8) if j not in subset]
            other = members[..., complement].mean(-1).argmin(1)
            agree.extend(selected == best)
            complement_agree.extend(selected == other)
            regret.extend((reference[indices, selected] - reference.min(1)) / normalizer)
            successes.append(float(outcomes[indices, selected].mean()))
        result["subsets"].append({
            "noise_count": count, "subset_count": len(successes),
            "same_choice_as_8": float(np.mean(agree)),
            "same_choice_as_disjoint_complement": float(np.mean(complement_agree)),
            "normalized_score_regret_p50_p95": np.percentile(regret, [50, 95]).tolist(),
            "historical_success_mean_min_max": [
                float(np.mean(successes)), min(successes), max(successes)],
        })
    return result


def relative_camera_extent(poses):
    """Endpoint offset from index-0 reference, not an executed travel path."""
    poses = np.asarray(poses)
    if (poses.ndim != 4 or poses.shape[-2:] != (4, 4) or poses.shape[1] < 2
            or not np.isfinite(poses).all()):
        raise ValueError("Finite [initial, candidate, 4, 4] poses required")
    rotation = poses[..., :3, :3]
    if (not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation,
                               np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
        raise ValueError("Rigid camera transforms required")
    relative_rotation = rotation[:, 0].transpose(0, 2, 1)[:, None] @ rotation[:, 1:]
    degrees = np.degrees(np.arccos(np.clip(
        (np.trace(relative_rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    translation = np.linalg.norm(poses[:, 1:, :3, 3] - poses[:, :1, :3, 3], axis=-1)
    return {"quantiles": [0, 50, 95, 100],
            "translation_m": np.percentile(translation, [0, 50, 95, 100]).tolist(),
            "rotation_deg": np.percentile(degrees, [0, 50, 95, 100]).tolist()}


def run(source, output):
    """Write one new audit artifact; never fit or invoke the operation model."""
    data, _ = load_initial32(source)
    with np.load(Path(source) / "matrix.npz", allow_pickle=False) as matrix:
        outcomes = matrix["success"][0].mean(-1)
    result = {
        "status": "DESCRIPTIVE_ONLY_NO_MODEL_CALLS",
        "model": "historical_canonical_PaliGemmaPi05",
        "source_hashes": data.source_hashes,
        "caveats": [
            "Subsets overlap the eight-noise reference; this is not independent confirmation.",
            "Correlated noise subsets are not independent initial states.",
            "Historical outcome means are descriptive, not used to choose a noise seed.",
            "Camera offsets are endpoint geometry, not physical motion or head reachability.",
        ],
        "geometry": relative_camera_extent(data.poses),
        "noise": {
            name: noise_subset_audit(data.accel_members[0, list(indices)], outcomes[list(indices)])
            for name, indices in (("development", data.development), ("evaluation", data.evaluation))
        },
    }
    output = Path(output)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result
