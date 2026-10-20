"""Capture must reuse, not rerun or alter, the frozen operation forward."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from AlphaBrain.model.pi05_features import SiglipFeatureCapture


class FakeTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, tokens):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=tokens * 2)


def test_capture_no_extra_forward_no_output_change():
    tower = FakeTower().eval()
    x = torch.arange(32.).reshape(1, 16, 2).requires_grad_()
    expected = tower(x).last_hidden_state
    with SiglipFeatureCapture(tower, patch_grid=(4, 4), view_names=("external", "wrist")) as tap:
        actual = tower(x).last_hidden_state
        tower(x + 1)
    assert tower.calls == 3
    assert not tower.training
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    features = tap.features()
    assert set(features) == {"external", "wrist"}
    assert features["external"].shape == (1, 4, 2)
    assert not features["external"].requires_grad
    torch.testing.assert_close(features["external"][0, 0], expected[0, [0, 1, 4, 5]].mean(0))
    assert len(tower._forward_hooks) == 0


def test_missing_extra_or_wrong_shape_fails_and_removes_hook():
    tower = FakeTower()
    with SiglipFeatureCapture(tower, patch_grid=(4, 4), view_names=("external",)) as tap:
        pass
    with pytest.raises(ValueError, match="Missing"):
        tap.features()
    with pytest.raises(ValueError, match="extra"):
        with tap:
            tower(torch.ones(1, 16, 2))
            tower(torch.ones(1, 16, 2))
    assert len(tower._forward_hooks) == 0
    with pytest.raises(ValueError, match="patch grid"):
        with tap:
            tower(torch.ones(1, 17, 2))
    assert len(tower._forward_hooks) == 0
    with pytest.raises(ValueError, match="Missing"):
        tap.features()


def test_nested_use_rejected_and_cleaned():
    tower = FakeTower()
    tap = SiglipFeatureCapture(tower, patch_grid=(4, 4), view_names=("external",))
    with pytest.raises(RuntimeError, match="already"):
        with tap:
            with tap:
                pass
    assert len(tower._forward_hooks) == 0
