"""Eager capture lifecycle/identity tests using only a small CPU stand-in."""
import pytest
torch = pytest.importorskip("torch")

from AlphaBrain.model.fastwam_features import FastWAMFeatureCapture


class Expert:
    def __init__(self):
        self.calls = 0

    def prepare(self, **kwargs):
        self.calls += 1
        self.output = (torch.rand(1, 4, 8),) + (None,) * 9
        return self.output


def invoke(expert, **changes):
    inputs = dict(x=torch.zeros(1, 2, 1, 2, 2), timestep=torch.zeros(1), action=None)
    inputs.update(changes)
    return expert.prepare(**inputs)


def capture(expert, query="q"):
    return FastWAMFeatureCapture(expert, query_id=query, token_shape=(1, 4, 8))


def test_no_extra_forward_or_rng_and_original_output_identity():
    expert = Expert()
    torch.manual_seed(123)
    expected = invoke(expert)
    rng_after = torch.get_rng_state().clone()
    torch.manual_seed(123)
    with capture(expert) as tap:
        result = invoke(expert)
        assert result is expert.output
        with pytest.raises(RuntimeError):
            tap.features(query_id="q")
    assert expert.calls == 2
    assert torch.equal(result[0], expected[0])
    assert torch.equal(torch.get_rng_state(), rng_after)
    assert "prepare" not in vars(expert)
    result[0].zero_()
    assert torch.equal(tap.features(query_id="q"), expected[0])
    with pytest.raises(ValueError):
        tap.features(query_id="other-query")


def test_nested_and_reused_context_rejected():
    expert = Expert()
    with capture(expert) as tap:
        with pytest.raises(RuntimeError):
            with capture(expert):
                pass
        invoke(expert)
    with pytest.raises(RuntimeError):
        with tap:
            pass
    with capture(expert, "new") as new:
        invoke(expert)
    assert new.features(query_id="new").shape == (1, 4, 8)


def test_zero_or_duplicate_calls_cannot_return_stale_features():
    expert = Expert()
    with capture(expert) as empty:
        pass
    with pytest.raises(ValueError):
        empty.features(query_id="q")
    with capture(expert) as duplicate:
        invoke(expert)
        with pytest.raises(ValueError):
            invoke(expert)
    with pytest.raises(ValueError):
        duplicate.features(query_id="q")


@pytest.mark.parametrize("changes", [
    {"x": torch.zeros(1, 2, 2, 2, 2)},
    {"timestep": torch.ones(1)},
    {"timestep": torch.tensor([float("nan")])},
    {"action": torch.zeros(1)},
])
def test_non_observation_calls_fail_and_restore(changes):
    expert = Expert()
    with pytest.raises(ValueError):
        with capture(expert) as tap:
            invoke(expert, **changes)
    assert "prepare" not in vars(expert)
    assert expert.calls == 0
    with pytest.raises(ValueError):
        tap.features(query_id="q")


def test_existing_override_and_exception_restored():
    expert = Expert()
    def fail(**kwargs):
        raise RuntimeError("simulated model error")
    expert.prepare = fail
    with pytest.raises(RuntimeError, match="model error"):
        with capture(expert) as tap:
            invoke(expert)
    assert expert.prepare is fail
    with pytest.raises(ValueError):
        tap.features(query_id="q")


@pytest.mark.parametrize("bad", [torch.zeros(1, 3, 8), torch.full((1, 4, 8), float("nan"))])
def test_invalid_output_rejected(bad):
    expert = Expert()
    expert.prepare = lambda **kwargs: (bad,) + (None,) * 9
    with pytest.raises(ValueError):
        with capture(expert):
            invoke(expert)
