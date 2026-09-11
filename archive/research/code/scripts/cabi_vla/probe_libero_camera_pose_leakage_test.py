from __future__ import annotations

import numpy as np

from probe_libero_camera_pose_leakage import (
    downsample_rgb,
    probe_metrics,
    ridge_fit_predict,
)


def test_downsample_rgb_preserves_constant_color() -> None:
    image = np.full((8, 8, 3), [32, 64, 128], dtype=np.uint8)
    feature = downsample_rgb(image, size=2).reshape(2, 2, 3)
    assert np.allclose(feature, np.asarray([32, 64, 128]) / 255.0)


def test_ridge_probe_recovers_linearly_separable_pose() -> None:
    labels = np.tile(np.arange(3), 8)
    features = np.eye(3)[labels]
    parameters = np.stack(
        [labels.astype(float), np.zeros_like(labels), np.ones_like(labels)],
        axis=1,
    )
    train = np.zeros(len(labels), dtype=bool)
    train[:12] = True
    test = ~train
    metrics = probe_metrics(
        features,
        labels,
        parameters,
        train,
        test,
        ridge=1e-3,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["r2"]["azimuth_deg"] > 0.99


def test_ridge_fit_predict_validates_regularization() -> None:
    values = np.ones((2, 1))
    try:
        ridge_fit_predict(values, values, values, ridge=0.0)
    except ValueError as error:
        assert "ridge" in str(error)
    else:
        raise AssertionError("expected invalid ridge to fail")

