import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.accel_field import (
    camera_folds, dense_field, field_metrics, normalize_coordinates,
    predict_field, quadratic_curvature, structure_diagnostics, validate_interpolation,
)


def geometry():
    return np.random.default_rng(3).uniform(size=(96, 3))


def test_normalization_and_input_validation():
    x, lo, span = normalize_coordinates(geometry())
    np.testing.assert_allclose(x.min(0), 0)
    np.testing.assert_allclose(x.max(0), 1)
    np.testing.assert_allclose(x * span + lo, geometry())
    with pytest.raises(ValueError):
        normalize_coordinates(np.zeros((10, 3)))


def test_quadratic_reproduces_bowl_and_curvature():
    x = geometry()
    y = np.sum((x - .3) ** 2, axis=1)
    q = np.array([[.3, .3, .3], [.8, .4, .9]])
    np.testing.assert_allclose(predict_field(x, y, q, "quadratic"), np.sum((q - .3) ** 2, axis=1), atol=1e-12)
    c = quadratic_curvature(x, y[None])[0]
    assert c["positive_definite"] and c["stationary_in_box"]
    np.testing.assert_allclose(c["stationary"], .3, atol=1e-12)


def test_location_folds_are_disjoint_and_complete():
    for mode in ("random", "spatial"):
        counts = np.zeros(96)
        for train, test in camera_folds(geometry(), mode):
            assert not (train & test).any()
            assert (train | test).all()
            counts += test
        np.testing.assert_array_equal(counts, 1)


def test_oof_does_not_read_heldout_targets():
    x = geometry()
    y = np.arange(96, dtype=float)[None]
    p, _ = validate_interpolation(x, y, ["rbf_smooth"], "random")
    _, test = camera_folds(x, "random")[0]
    changed = y.copy()
    changed[:, test] += 10000
    p2, _ = validate_interpolation(x, changed, ["rbf_smooth"], "random")
    np.testing.assert_allclose(p["rbf_smooth"][:, test], p2["rbf_smooth"][:, test])


def test_metrics_report_missing_support_and_low_region():
    m = field_metrics(np.arange(8), np.arange(8, dtype=float))
    assert m["r2"] == 1 and m["low_quartile_recall"] == 1
    m = field_metrics(np.arange(8), [0, 1, 2, 3, np.nan, np.nan, np.nan, np.nan])
    assert m["coverage"] == .5 and m["n"] == 4


def test_dense_estimates_mask_outside_hull():
    x = geometry() * .5 + .25
    d = dense_field(x, np.sum(x, axis=1)[None], 8)
    assert (~d["inside_hull"]).any()
    assert np.isnan(d["estimated_accel"][:, ~d["inside_hull"]]).all()
    assert np.isfinite(d["estimated_accel"][:, d["inside_hull"]]).all()


def test_transfer_template_excludes_target_task():
    a = np.random.default_rng(0).uniform(size=(2, 8, 96, 8))
    t = np.repeat([0, 1], 4)
    d = structure_diagnostics(a, t, np.tile(np.arange(4), 2))
    assert len(d["noise"]) == 16 and len(d["cross_model"]) == 8
    with pytest.raises(ValueError):
        structure_diagnostics(a[:, :, :, :3], t, t)


def test_noise_target_cannot_affect_fitted_predictions():
    x = geometry()
    y = np.sum(x*x, axis=1)[None]
    p, rows = validate_interpolation(x, y, ["quadratic"], "spatial", target_fields=y+7)
    p2, _ = validate_interpolation(x, y, ["quadratic"], "spatial", target_fields=y)
    np.testing.assert_allclose(p["quadratic"], p2["quadratic"])
    assert rows[0]["rmse"] == pytest.approx(7)


def test_common_noise_offset_does_not_imply_unstable_shape():
    spatial = np.linspace(.01, .02, 96)
    offsets = np.arange(8) / 10
    a = np.broadcast_to(spatial[:, None] + offsets, (2, 8, 96, 8)).copy()
    d = structure_diagnostics(a, np.repeat([0, 1], 4), np.tile(range(4), 2))
    n = d["noise"][0]
    assert n["mean_noise_variance_to_spatial_variance"] > 1
    assert n["offset_removed_noise_variance_to_spatial_variance"] < 1e-20
    assert n["low_quartile_recall"] == 1
