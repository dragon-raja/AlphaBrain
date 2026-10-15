import numpy as np
import pytest

matplotlib=pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AlphaBrain.research.dsol.plotting.local_view_derivative import (
    direction_figure,noise_figure,axis_curves_figure,
)


def test_plot_values_and_adaptive_symmetric_scales_are_not_reestimated():
    records=[]
    cases=[]
    for model in ("canonical","broad"):
        for state in (0,4):
            for candidate in (0,1):
                record=dict(model=model,state_index=state,candidate_index=candidate,
                    geometric_predictions={"all_valid":dict(predicted_half_difference=[.001]*6,scaled_direction_cosine=.5)},
                    measured_half_difference=[-.002]*6)
                records.append(record)
                scores=np.tile(np.arange(13),(8,1))*.001
                cases.append(dict(model=model,state_index=state,candidate_index=candidate,scores=scores,record=record))
    fig=direction_figure(records)
    assert len(fig.axes)==8
    np.testing.assert_allclose([p.get_height() for p in fig.axes[0].patches], [1]*6+[-2]*6)
    lo,hi=fig.axes[0].get_ylim()
    assert lo==-hi
    plt.close(fig)
    fig=noise_figure(cases,list(range(41,49)))
    np.testing.assert_allclose(np.asarray(fig.axes[0].images[0].get_array()),.5)
    plt.close(fig)
    fig=axis_curves_figure(cases)
    # First line: measured -x, base, +x, relative to base; no interpolated observations.
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(),[1,0,2])
    plt.close(fig)
