"""Negative provenance/sensor cases, with no simulator or model imports."""
from dataclasses import replace

import pytest

from AlphaBrain.research.dsol.data.native_observation import (
    CameraFrame, FeatureSpec, ObservationBundle, validate_feature_metadata,
)


def bundle(count=1):
    return ObservationBundle(
        "query-a", "normalized-device-clock", 10., 11., 1., .2,
        "a" * 64, "b" * 64,
        tuple(CameraFrame(f"camera-{i}", "movable" if i == 0 else "context",
                          True, 10.9, "c" * 64) for i in range(count)))


def check(obs, **changes):
    spec = FeatureSpec("target_native", "d" * 64, "prepare.output.tokens", "joint",
                       tuple(c.camera_id for c in obs.cameras if c.valid),
                       (16, 128), "float32", obs.preprocessing_sha256)
    validate_feature_metadata(replace(spec, **changes), obs, query_id=obs.query_id,
                              history_sha256=obs.history_sha256,
                              observed_shape=(16, 128), observed_dtype="float32")


@pytest.mark.parametrize("count", [1, 2, 3])
def test_real_view_counts_and_joint_records(count):
    check(bundle(count))
    check(bundle(count), fusion_scope="per_view", camera_ids=("camera-0",))


def test_missing_context_is_not_a_fake_frame():
    obs = bundle()
    missing = CameraFrame("missing-wrist", "context", False, None, None)
    check(replace(obs, cameras=obs.cameras + (missing,)))
    with pytest.raises(ValueError, match="fabricated"):
        replace(obs, cameras=obs.cameras + (replace(missing, image_sha256="c" * 64),)).validate()
    with pytest.raises(ValueError, match="available"):
        check(replace(obs, cameras=obs.cameras + (missing,)),
              fusion_scope="per_view", camera_ids=("missing-wrist",))


@pytest.mark.parametrize("stamp", [9.9, 11.1, float("nan")])
def test_pre_motion_future_and_invalid_frames_rejected(stamp):
    obs = bundle()
    with pytest.raises(ValueError):
        replace(obs, cameras=(replace(obs.cameras[0], captured_at=stamp),)).validate()


def test_age_skew_and_duplicate_identity_rejected():
    obs = bundle(2)
    for bad in (replace(obs, max_age=.01),
                replace(obs, cameras=(obs.cameras[0], replace(obs.cameras[1], captured_at=10.1))),
                replace(obs, cameras=(obs.cameras[0], obs.cameras[0]))):
        with pytest.raises(ValueError):
            bad.validate()


@pytest.mark.parametrize("changes", [
    {"camera_ids": ("camera-1", "camera-0")},
    {"camera_ids": ("camera-0",)},
    {"fusion_scope": "guessed-split"},
    {"preprocessing_sha256": "e" * 64},
    {"model_sha256": "unknown"},
    {"shape": (32, 128)},
    {"dtype": "float16"},
])
def test_mapping_and_feature_identity_rejected(changes):
    with pytest.raises(ValueError):
        check(bundle(2), **changes)


@pytest.mark.parametrize("query,history", [("query-b", "b" * 64), ("query-a", "e" * 64)])
def test_other_candidate_or_history_cache_rejected(query, history):
    obs = bundle()
    spec = FeatureSpec("target_native", "d" * 64, "tokens", "joint",
                       ("camera-0",), (16, 128), "float32", "a" * 64)
    with pytest.raises(ValueError, match="another query"):
        validate_feature_metadata(spec, obs, query_id=query, history_sha256=history,
                                  observed_shape=(16, 128), observed_dtype="float32")
