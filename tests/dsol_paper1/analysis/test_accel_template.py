import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.accel_template import (
    calibrated_prediction, coverage_curve, template_for,
)


def test_coverage_definition_and_monotonicity():
    y = np.arange(96, dtype=float)
    c = coverage_curve(y, y, 24)
    assert c["capture"][23] == 1
    assert c["min_sites_capture80"] == 20
    assert c["min_sites_capture90"] == 22
    assert c["min_sites_capture100"] == 24
    assert np.all(np.diff(c["capture"]) >= 0)
    reverse = coverage_curve(y, -y, 24)
    assert reverse["capture"][71] == 0
    assert reverse["capture"][-1] == 1


def test_template_excludes_target_and_target_task():
    rng = np.random.default_rng(9)
    fields = rng.normal(size=(8, 96))
    tasks = np.repeat([0, 1], 4)
    for kind in ["same_task", "other_tasks"]:
        before = template_for(fields, tasks, 0, kind)
        changed = fields.copy()
        changed[0] += rng.normal(size=96)*100
        if kind == "other_tasks":
            changed[:4] += rng.normal(size=(4, 96))*100
        np.testing.assert_allclose(before, template_for(changed, tasks, 0, kind))


def test_positive_affine_recovers_without_changing_order():
    x = np.random.default_rng(3).uniform(size=(96, 3))
    t = np.sin(x[:, 0]*6) + x[:, 1]**2
    y = 2 + 3*t
    train = np.arange(96) % 2 == 0
    p, fit = calibrated_prediction(t, x, y, train, "affine")
    np.testing.assert_allclose(p, y, atol=1e-12)
    assert fit["scale"] == pytest.approx(3)
    np.testing.assert_array_equal(np.argsort(p), np.argsort(t))


def test_spatial_residual_captures_shape_change():
    x = np.random.default_rng(3).uniform(size=(96, 3))
    t = np.sin(x[:, 0]*6) + x[:, 1]**2
    y = 2 + 3*t + x @ np.array([.1, 2, -.8])
    train = np.arange(96) % 2 == 0
    p, _ = calibrated_prediction(t, x, y, train, "affine_linear")
    np.testing.assert_allclose(p, y, atol=1e-12)


def test_calibration_never_uses_heldout_values():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=(96, 3))
    t, y = rng.normal(size=(2, 96))
    train = np.arange(96) % 2 == 0
    for method in ["constant", "affine", "affine_linear", "affine_quadratic"]:
        p, _ = calibrated_prediction(t, x, y, train, method)
        changed = y.copy()
        changed[~train] += 100
        p2, _ = calibrated_prediction(t, x, changed, train, method)
        np.testing.assert_allclose(p, p2)


def test_negative_scale_is_not_allowed_to_flip_low_region():
    x = np.random.default_rng(3).uniform(size=(96, 3))
    t = np.arange(96, dtype=float)
    p, fit = calibrated_prediction(t, x, -t, np.ones(96, bool), "affine")
    assert fit["scale"] == 0
    assert np.ptp(p) == 0


def test_invalid_region_size_rejected():
    with pytest.raises(ValueError):
        coverage_curve(np.arange(5), np.arange(5), 6)
