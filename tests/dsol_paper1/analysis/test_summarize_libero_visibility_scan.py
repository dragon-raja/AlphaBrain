from __future__ import annotations

import math

from scripts.dsol_paper1.analysis.summarize_libero_visibility_scan import diagnostic_plot_data


def test_diagnostic_plot_data_uses_protocol_split_names() -> None:
    flat_rows = [
        {"split": "calibration", "group": "broad_training_64", "delta_visibility": 0.1},
        {"split": "heldout_test", "group": "broad_training_64", "delta_visibility": 0.2},
    ]
    state_rows = [
        {
            "split": "calibration",
            "best_delta_visibility": 0.1,
            "worst_delta_visibility": -0.1,
        },
        {
            "split": "heldout_test",
            "best_delta_visibility": 0.2,
            "worst_delta_visibility": -0.2,
        },
    ]

    groups, labels, values = diagnostic_plot_data(flat_rows, state_rows)

    assert groups == {"broad_training_64": [0.1]}
    assert labels == [
        "calibration best",
        "calibration worst",
        "held-out best",
        "held-out worst",
    ]
    assert values == [0.1, -0.1, 0.2, -0.2]
    assert all(math.isfinite(value) for value in values)
