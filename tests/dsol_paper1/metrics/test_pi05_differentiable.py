from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from AlphaBrain.research.dsol.metrics.pi05_differentiable import velocity_prefix


class PrefixLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(past_key_values=inputs_embeds.mean())


class Expert(PrefixLM):
    def forward(self, inputs_embeds, past_key_values, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds+past_key_values)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.num_inference_steps = 10
        self.action_horizon = 2
        self.action_dim = 1
        self.action_expert = SimpleNamespace(model=SimpleNamespace(model=Expert()))
        self.action_out_proj = torch.nn.Identity()

    def embed_suffix(self, state, x, time):
        assert state is None
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        return x+time[:, None, None], mask, mask, None


class Model(torch.nn.Module):
    pi05 = True

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(framework=SimpleNamespace())
        self.flow_matching_head = Head()
        self.lm = PrefixLM()
        self.eval()

    def _detect_vlm_type(self, config):
        return "paligemma"

    def _prepare_prefix_paligemma(self, examples):
        emb = examples[0]["image"].reshape(1, 2, 1)
        mask = torch.ones((1, 2), dtype=torch.bool)
        return emb, mask, torch.zeros_like(mask)

    def _apply_cabi_prefix(self, emb, pad, **kwargs):
        return emb, None

    def _get_vlm_language_model(self):
        return self.lm


def test_preserves_original_grid_recurrence_and_image_graph():
    model = Model()
    image = torch.tensor([2., 4.], requires_grad=True)
    noise = torch.ones((1, 2, 1))
    trace, times = velocity_prefix(model, [{"image": image}], noise)
    x, time, dt = noise, torch.tensor(1.), torch.tensor(-.1)
    reference = []
    for _ in range(3):
        v = x+time+image.mean()
        reference.append(v)
        x, time = x+dt*v, time+dt
    torch.testing.assert_close(trace, torch.stack(reference, dim=1), rtol=0, atol=0)
    np.testing.assert_allclose(times.numpy(), [1., .9, .8], atol=1e-7)
    actual, = torch.autograd.grad(trace.sum(), image, retain_graph=True)
    expected, = torch.autograd.grad(torch.stack(reference).sum(), image)
    torch.testing.assert_close(actual, expected)
    assert torch.all(actual != 0)


@pytest.mark.parametrize("prefix", [1, 11, True, 2.5])
def test_rejects_invalid_prefix(prefix):
    with pytest.raises(ValueError):
        velocity_prefix(Model(), [], torch.ones((1, 2, 1)), prefix=prefix)


def test_rejects_training_or_unfrozen_model():
    model = Model().train()
    with pytest.raises(ValueError, match="frozen"):
        velocity_prefix(model, [], torch.ones((1, 2, 1)))
    model.eval().flow_matching_head.anchor.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        velocity_prefix(model, [], torch.ones((1, 2, 1)))
