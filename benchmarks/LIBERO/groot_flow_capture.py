"""Pinned FluxVLA graph observation overlay, not a model loader or GPU permit.

Graph replay does not execute Python forward hooks. Preallocated copy operations
must be captured alongside the native graph. Keep the upstream snapshot intact;
write the returned source to a separate, hash-bound candidate release. CPU tests
cannot establish numerical equivalence of the real CUDA graphs.
"""
import ast
import hashlib
from pathlib import Path
import sys
import types

import numpy as np


UPSTREAM_SHA256 = 'c33b76649811e987cbc7b5ce9c540ebcf03781a76c1265299424e57fd1602b79'
ALLOCATE = '''        # DSOL observation buffers: no alias to native actions or RNG use.
        self.buffers['dsol_states'] = self.buffers['actions'].new_empty(
            (self.num_inference_timesteps,) + tuple(self.buffers['actions'].shape))
        self.buffers['dsol_velocities'] = self.buffers['dsol_states'].new_empty(
            self.buffers['dsol_states'].shape)
        self.buffers['dsol_generation'] = torch.zeros(
            (), dtype=torch.int64, device=self.buffers['actions'].device)
'''
OBSERVE = '''            self.buffers['dsol_states'][t].copy_(actions)
            self.buffers['dsol_velocities'][t].copy_(pred_velocity)
'''
GENERATION = "        self.buffers['dsol_generation'].add_(1)\n"


def instrument_source(source):
    """Fail closed on any upstream revision change; preserve native arithmetic."""
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError('Unreviewed FluxVLA source identity')
    changes = (
        ('        self.graph = torch.cuda.CUDAGraph()\n', ALLOCATE, False),
        ('            actions = actions + dt * pred_velocity\n', OBSERVE, False),
        ("        self.buffers['actions'].copy_(actions)\n", GENERATION, True),
    )
    result = source
    for anchor, addition, after in changes:
        if result.count(anchor) != 1:
            raise ValueError('Ambiguous native graph capture seam')
        result = result.replace(anchor, anchor + addition if after else addition + anchor)
    ast.parse(result)
    restored = result
    for _, addition, _ in changes:
        restored = restored.replace(addition, '', 1)
    if restored != source:
        raise ValueError('Instrumentation changed native source')
    return result


def install_graph_observer(head):
    """Bind only the two reviewed methods before native graph initialization.

    The owning runtime must already hold its model permit. This function does
    not create a model, initialize CUDA, rebuild an active graph or patch the
    upstream module/class globally. A separate native instance remains usable
    as the numerical reference.
    """
    cls = type(head)
    module = sys.modules.get(cls.__module__)
    if (module is None or getattr(module, 'FlowMatchingInferenceHead', None) is not cls
            or getattr(head, 'loaded_weights', True)
            or hasattr(head, '_dsol_observer_sha256')):
        raise ValueError('Uninitialized pinned native head required')
    path = Path(module.__file__)
    source = path.read_text()
    observed = instrument_source(source)
    tree = ast.parse(observed)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'FlowMatchingInferenceHead']
    if len(classes) != 1:
        raise ValueError('Ambiguous native head class')
    names = {'_load_weights_and_buffer', 'record_run'}
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef)
               and n.name in names]
    if {n.name for n in methods} != names or any(n in head.__dict__ for n in names):
        raise ValueError('Native methods already overridden or missing')
    namespace = dict(vars(module))
    executable = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    exec(compile(executable, str(path)+'#dsol-observer', 'exec'), namespace)
    for name in names:
        setattr(head, name, types.MethodType(namespace[name], head))
    head._dsol_observer_sha256 = hashlib.sha256(observed.encode()).hexdigest()
    return head._dsol_observer_sha256


def graph_generation(head):
    """Caller synchronizes through scalar read; does not initialize a model."""
    if not getattr(head, 'loaded_weights', False):
        raise ValueError('Initialized graph required; warmup is not a scored call')
    value = head.buffers['dsol_generation']
    if tuple(value.shape) != ():
        raise ValueError('Invalid graph generation')
    return int(value.detach().cpu().item())


def read_graph_trace(head, *, previous_generation):
    """Copy a single completed replay, rejecting stale, RTC or nonnative traces.

    The caller must serialize native and instrumented model calls and record
    generation *after* graph initialization but *before* the scored replay.
    This returns raw velocities; it neither changes nor selects a score metric.
    """
    if type(previous_generation) is not int or previous_generation < 0:
        raise ValueError('Explicit previous graph generation required')
    if graph_generation(head) != previous_generation + 1:
        raise ValueError('Stale or multiple graph replays')
    if (head.num_inference_timesteps, head.num_steps, head.action_dim,
            head.ori_action_dim) != (4, 10, 32, 7):
        raise ValueError('Unreviewed GR00T native sampler')
    array = lambda tensor: tensor.detach().float().cpu().numpy().copy()
    for key in ('prefill_mask', 'prefix_ts_mask'):
        if np.any(array(head.buffers[key]) != 0):
            raise ValueError('RTC prefix conditioning is outside the score contract')
    states = array(head.buffers['dsol_states'])
    velocities = array(head.buffers['dsol_velocities'])
    if (states.shape != (4, 1, 10, 32) or velocities.shape != states.shape
            or not np.isfinite(states).all() or not np.isfinite(velocities).all()):
        raise ValueError('Invalid native graph trace')
    if np.any(states[..., 7:] != 0):
        raise ValueError('Native padded state dimensions changed')
    return dict(states=states, velocities=velocities,
                effective_velocities=velocities[..., :7].copy(),
                initial_noise=states[0].copy(),
                times=np.arange(4, dtype=np.float64) / 4,
                generation=previous_generation + 1)
