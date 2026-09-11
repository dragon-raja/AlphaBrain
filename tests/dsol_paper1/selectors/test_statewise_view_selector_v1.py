from __future__ import annotations

import numpy as np

from scripts.dsol_paper1.training.train_statewise_view_selector_v1 import (
    apply_pca,
    fit_pca,
    fit_ridge,
    geometry_vector,
    select_predictions,
)


def test_ridge_recovers_linear_order() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    targets = np.asarray([0.0, 1.0, 2.0, 3.0])
    model = fit_ridge(features, targets, alpha=0.1)
    predictions = model.predict(features)
    assert np.all(np.diff(predictions) > 0)


def test_pca_transform_has_requested_dimension() -> None:
    values = np.arange(60, dtype=np.float64).reshape(10, 6)
    pca = fit_pca(values, 3)
    assert apply_pca(values, pca).shape == (10, 3)


def test_prediction_ties_use_lexicographic_candidate_id() -> None:
    selected = select_predictions(
        np.asarray([1.0, 1.0]),
        [(0, 0), (0, 1)],
        ["view-z", "view-a"],
    )
    assert selected == {0: 1}


def test_geometry_vector_does_not_use_visibility() -> None:
    candidate = {
        "selected_candidate_id": "broad_train_001",
        "pose": {"azimuth_deg": 30.0, "elevation_deg": 15.0, "radius_scale": 1.1},
        "candidate_features": {"visibility_score": 999.0},
    }
    first = geometry_vector(candidate)
    candidate["candidate_features"]["visibility_score"] = -999.0
    second = geometry_vector(candidate)
    assert np.array_equal(first, second)
