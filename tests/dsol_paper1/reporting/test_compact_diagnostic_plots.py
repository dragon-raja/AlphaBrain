"""Compact report plots retain input geometry and audited diagnostic ratios."""
import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AlphaBrain.research.dsol.plotting.rig_geometry import compact_candidate_figure
from AlphaBrain.research.dsol.plotting.observation_evidence import compact_response_figure


def test_candidate_plot_retains_catalogue_and_does_not_mutate_inputs():
    parameters = np.arange(291, dtype=float).reshape(97, 3)
    positions = np.arange(582, dtype=float).reshape(97, 2, 3) / 100
    forwards = np.zeros_like(positions)
    forwards[..., 0] = 1
    originals = [a.copy() for a in (parameters, positions, forwards)]
    fig = compact_candidate_figure(parameters, positions, forwards)
    try:
        ax = fig.axes[0]
        for collection, expected in zip(ax.collections[:3], [parameters[1:65], parameters[65:], parameters[:1]]):
            np.testing.assert_allclose(np.array(collection._offsets3d).T, expected)
        for a, original in zip((parameters, positions, forwards), originals):
            np.testing.assert_array_equal(a, original)
    finally:
        plt.close(fig)


def test_candidate_plot_rejects_incompatible_geometry():
    with pytest.raises(ValueError):
        compact_candidate_figure(np.zeros((96, 3)), np.zeros((97, 2, 3)), np.zeros((97, 2, 3)))


def test_response_plot_uses_broad_over_canonical_without_success_interpretation():
    keys = ['view_response_rms', 'canonical_state_response_rms', 'perturbed_demonstration_mse']
    canonical = dict(zip(keys, [2., 4., 5.]))
    broad = dict(zip(keys, [1., 4., 6.]))
    summary = {'benchmarks': {
        'libero': {'models': {'canonical': canonical, 'broad': broad}},
        'robocasa': {'models': {'canonical_8250': canonical, 'broad_8250': broad}},
    }}
    fig = compact_response_figure(summary)
    try:
        np.testing.assert_allclose([p.get_height() for p in fig.axes[0].patches], [.5, 1, .5, 1])
        np.testing.assert_allclose([p.get_height() for p in fig.axes[1].patches], [1.2, 1.2])
    finally:
        plt.close(fig)
