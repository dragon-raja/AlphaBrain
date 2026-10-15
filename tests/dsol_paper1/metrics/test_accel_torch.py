import numpy as np
import pytest

torch = pytest.importorskip("torch")

from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from AlphaBrain.research.dsol.metrics.accel_torch import compute_accel_score_tensor


def test_registered_formula_matches_numpy_all_prefixes():
    trace = np.random.default_rng(17).normal(size=(4, 10, 5, 3))
    reference = compute_accel_scores(trace)
    tensor = torch.tensor(trace, dtype=torch.float64, requires_grad=True)
    for prefix in range(2, 11):
        actual = compute_accel_score_tensor(tensor, prefix=prefix)
        for name in ("score", "numerator", "denominator", "degenerate"):
            np.testing.assert_allclose(actual[name].detach().numpy(), reference[prefix][name],
                                       atol=1e-12, rtol=1e-12)
        assert actual["score"].requires_grad


def test_accel_gradient_matches_finite_differences():
    trace = torch.tensor(np.random.default_rng(3).normal(size=(2, 3, 2, 2)),
                         dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: compute_accel_score_tensor(x)["score"], (trace,))


def test_degenerate_is_not_a_valid_low_score_and_backward_is_finite():
    trace = torch.zeros((1, 3, 2), dtype=torch.float64, requires_grad=True)
    result = compute_accel_score_tensor(trace)
    assert result["degenerate"].all()
    result["score"].sum().backward()
    assert torch.isfinite(trace.grad).all()


@pytest.mark.parametrize("prefix", [1, 4, True, 2.5])
def test_invalid_prefix_rejected(prefix):
    with pytest.raises(ValueError):
        compute_accel_score_tensor(torch.ones((1, 3, 2)), prefix=prefix)
