"""CPU tests only; no FluxVLA import, checkpoint, CUDA graph or simulator."""
from types import SimpleNamespace
import types
import sys
import hashlib

import numpy as np
import pytest

from benchmarks.LIBERO.groot_flow_capture import instrument_source, read_graph_trace
from benchmarks.LIBERO import groot_flow_capture


def test_observer_binding_is_per_instance_and_rejects_active_graph(tmp_path, monkeypatch):
    source = '''class FlowMatchingInferenceHead:
    def __init__(self):
        self.loaded_weights = False
    def _load_weights_and_buffer(self):
        self.graph = torch.cuda.CUDAGraph()
    def record_run(self):
        actions = self.buffers['actions']
        dt = 0.25
        for t in range(4):
            pred_velocity = actions + 1
            actions = actions + dt * pred_velocity
        self.buffers['actions'].copy_(actions)
'''
    path = tmp_path/'native.py'; path.write_text(source)
    module = types.ModuleType('_groot_observer_cpu_fixture')
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(source, module.__dict__)
    monkeypatch.setattr(groot_flow_capture, 'UPSTREAM_SHA256',
                        hashlib.sha256(source.encode()).hexdigest())
    native = module.FlowMatchingInferenceHead()
    observed = module.FlowMatchingInferenceHead()
    original = native.record_run.__func__
    stamp = groot_flow_capture.install_graph_observer(observed)
    assert len(stamp) == 64
    assert observed.record_run.__func__ is not original
    assert observed.record_run.__self__ is observed
    assert native.record_run.__func__ is original
    assert module.FlowMatchingInferenceHead.record_run is original
    assert path.read_text() == source
    with pytest.raises(ValueError): groot_flow_capture.install_graph_observer(observed)
    native.loaded_weights = True
    with pytest.raises(ValueError): groot_flow_capture.install_graph_observer(native)
    native.loaded_weights = False
    path.write_text(source+'# changed\n')
    with pytest.raises(ValueError, match='source identity'):
        groot_flow_capture.install_graph_observer(native)


def test_unreviewed_source_is_rejected():
    with pytest.raises(ValueError, match='source identity'):
        instrument_source('class NotTheReviewedSampler: pass\n')


def test_observation_copies_preserve_cpu_sampler_arithmetic_and_rng(monkeypatch):
    torch = pytest.importorskip('torch')
    # Exercise the insertion code on a tiny trusted CPU sampler. This is not
    # a CUDA replay test; the actual upstream overlay is also AST checked.
    source = '''class Sampler:
    def __init__(self):
        self.num_inference_timesteps, self.num_steps, self.action_dim = 4, 10, 32
        self.buffers = {'actions': torch.zeros(1, 10, 32)}
        self.graph = torch.cuda.CUDAGraph()
    def run(self):
        actions = self.buffers['actions']
        dt = 1.0 / self.num_inference_timesteps
        for t in range(self.num_inference_timesteps):
            pred_velocity = actions * 0.25 + 1
            actions = actions + dt * pred_velocity
        self.buffers['actions'].copy_(actions)
'''
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', lambda: None)
    monkeypatch.setattr(groot_flow_capture, 'UPSTREAM_SHA256',
                        hashlib.sha256(source.encode()).hexdigest())
    original = {'torch': torch}
    observed = {'torch': torch}
    exec(source, original)
    exec(instrument_source(source), observed)
    a, b = original['Sampler'](), observed['Sampler']()
    rng = torch.get_rng_state().clone()
    for generation in (1, 2):
        a.run(); b.run()
        assert torch.equal(a.buffers['actions'], b.buffers['actions'])
        assert b.buffers['dsol_generation'].item() == generation
        assert torch.equal(torch.get_rng_state(), rng)
        for i in range(3):
            assert torch.equal(b.buffers['dsol_states'][i + 1],
                               b.buffers['dsol_states'][i] +
                               .25 * b.buffers['dsol_velocities'][i])
    with pytest.raises(ValueError): instrument_source(instrument_source(source))


def make_head():
    torch = pytest.importorskip('torch')
    return SimpleNamespace(loaded_weights=True, num_inference_timesteps=4,
                           num_steps=10, action_dim=32, ori_action_dim=7,
                           buffers=dict(dsol_generation=torch.tensor(5),
                                        dsol_states=torch.zeros(4, 1, 10, 32),
                                        dsol_velocities=torch.ones(4, 1, 10, 32),
                                        prefill_mask=torch.zeros(1, 10, 1),
                                        prefix_ts_mask=torch.zeros(1, 10)))


def test_replay_trace_is_detached_and_selects_only_effective_dimensions():
    head = make_head()
    result = read_graph_trace(head, previous_generation=4)
    assert result['effective_velocities'].shape == (4, 1, 10, 7)
    np.testing.assert_array_equal(result['times'], [0, .25, .5, .75])
    head.buffers['dsol_velocities'].fill_(99)
    assert np.all(result['velocities'] == 1)


@pytest.mark.parametrize('failure', ['stale', 'multiple', 'uninitialized', 'rtc',
                                    'shape', 'nonfinite', 'padding', 'sampler'])
def test_unusable_graph_trace_fails_closed(failure):
    head = make_head()
    if failure == 'stale': head.buffers['dsol_generation'].fill_(4)
    if failure == 'multiple': head.buffers['dsol_generation'].fill_(6)
    if failure == 'uninitialized': head.loaded_weights = False
    if failure == 'rtc': head.buffers['prefill_mask'].fill_(1)
    if failure == 'shape': head.buffers['dsol_states'] = head.buffers['dsol_states'][:3]
    if failure == 'nonfinite': head.buffers['dsol_velocities'].fill_(float('nan'))
    if failure == 'padding': head.buffers['dsol_states'][..., 7:].fill_(1)
    if failure == 'sampler': head.num_inference_timesteps = 10
    with pytest.raises(ValueError): read_graph_trace(head, previous_generation=4)
