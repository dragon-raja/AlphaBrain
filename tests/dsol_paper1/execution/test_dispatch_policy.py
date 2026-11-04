"""Actual closed entrypoints: no models, workers, subprocesses or queue mutations."""
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest

from AlphaBrain.research.dsol.execution import dispatch_policy as policy, queue
from AlphaBrain.research.dsol.execution.controlled_queue import LedgerRecoveryBackend
from AlphaBrain.research.dsol.execution.simulation_backend import SimulationBackend

# Explicit contract inventory, independent of AST guard discovery.
BOUNDARIES = [
    ('AlphaBrain.research.dsol.execution.process_backend', 'LocalProcessBackend.start'),
    ('benchmarks.LIBERO.baseline_dispatch', 'run'),
    ('benchmarks.Robocasa365.probe_queue', 'ProbeBackend.start'),
    ('benchmarks.LIBERO.continuous_evaluation_fleet', 'worker'),
    ('benchmarks.LIBERO.continuous_evaluation_fleet', 'run_phase'),
    ('benchmarks.LIBERO.continuous_evaluation_fleet', 'BlockBackend.start'),
    ('benchmarks.LIBERO.continuous_evaluation', 'run'),
    ('benchmarks.LIBERO.continuous_nominal', 'run'),
    ('benchmarks.LIBERO.continuous_training', 'run'),
    ('benchmarks.LIBERO.continuous_training', 'LiveEpisodes.__init__'),
    ('benchmarks.LIBERO.continuous_parallel', 'worker'),
    ('benchmarks.LIBERO.continuous_parallel', 'collect_batch'),
    ('benchmarks.LIBERO.continuous_parallel', 'admission'),
    ('benchmarks.LIBERO.continuous_parallel', 'train'),
    ('benchmarks.LIBERO.continuous_parallel', 'capacity_admission'),
    ('benchmarks.LIBERO.continuous_resident', 'serve'),
    ('benchmarks.LIBERO.continuous_resident', 'ResidentPool.__init__'),
    ('benchmarks.LIBERO.continuous_resident', 'ResidentPool.start'),
    ('benchmarks.LIBERO.continuous_resident', 'ResidentPool.collect'),
    ('benchmarks.LIBERO.search_runtime', 'PolicyRuntime.__init__'),
    ('benchmarks.LIBERO.search_runtime', 'PolicyRuntime.infer'),
    ('benchmarks.LIBERO.native_search_runtime', 'NativePolicyRuntime.__init__'),
    ('benchmarks.LIBERO.native_search_runtime', 'NativePolicyRuntime.infer'),
    ('benchmarks.LIBERO.search_live', 'simulator'),
    ('benchmarks.LIBERO.search_live', 'run'),
    ('benchmarks.LIBERO.search_fleet', 'capacity'),
    ('benchmarks.LIBERO.search_fleet', 'evaluation'),
    ('benchmarks.LIBERO.search_capacity', 'run'),
    ('benchmarks.LIBERO.search_rollout', 'run'),
    ('benchmarks.LIBERO.continuous_action_admission', 'run'),
]


class Poison:
    def __getattr__(self, name):
        raise AssertionError('Touched execution inputs before refusing dispatch: '+name)
    def __fspath__(self):
        raise AssertionError('Touched execution paths before refusing dispatch')
    def __getitem__(self, key):
        raise AssertionError('Read execution inputs before refusing dispatch')


@pytest.mark.parametrize('module_name,qualified', BOUNDARIES)
def test_real_boundary_refuses_before_inputs_or_effects(module_name, qualified, tmp_path, monkeypatch):
    function = importlib.import_module(module_name)
    for part in qualified.split('.'):
        function = getattr(function, part)
    def forbidden(*args, **kwargs):
        raise AssertionError('Process start or signalling reached')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(os, 'kill', forbidden)
    kwargs = {}
    for name, param in inspect.signature(function).parameters.items():
        if name == 'execute':
            kwargs[name] = True
        elif param.default is inspect.Parameter.empty:
            kwargs[name] = Poison()
    before = set(sys.modules)
    with pytest.raises(policy.ProductionDispatchDisabled):
        if qualified == 'simulator':
            with function(**kwargs):
                pytest.fail('Simulator entered')
        else:
            function(**kwargs)
    assert not {'torch', 'mujoco', 'wandb'}.intersection(set(sys.modules)-before)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('operation', ['run', 'recover'])
def test_raw_queue_refused_before_files_or_backend(operation, tmp_path):
    root = tmp_path/'must-not-be-created'
    with pytest.raises(policy.ProductionDispatchDisabled):
        getattr(queue, operation)(root, Poison())
    assert not root.exists()


@pytest.mark.parametrize('operation', ['run', 'recover'])
def test_raw_simulator_cannot_bypass_controller(operation, tmp_path):
    backend = SimulationBackend(tmp_path/'fake', {})
    before = backend.path.read_bytes()
    with pytest.raises(policy.ProductionDispatchDisabled):
        getattr(queue, operation)(tmp_path/'queue', backend)
    assert backend.path.read_bytes() == before
    assert not (tmp_path/'queue').exists()


@pytest.mark.parametrize('operation', ['run', 'recover'])
def test_private_scope_rejects_foreign_or_subclass_backend(operation, tmp_path):
    class Foreign(LedgerRecoveryBackend):
        pass
    backend = SimulationBackend(tmp_path/'fake', {})
    for wrapped in (Poison(), backend, Foreign(backend)):
        with pytest.raises(policy.ProductionDispatchDisabled):
            with policy._controlled_scope(tmp_path/'queue', wrapped, operation):
                pytest.fail('Unregistered backend admitted')


def test_scope_is_bound_to_root_backend_operation_thread_and_process(tmp_path, monkeypatch):
    backend = LedgerRecoveryBackend(SimulationBackend(tmp_path/'fake', {}))
    root = tmp_path/'queue'
    with policy._controlled_scope(root, backend, 'recover'):
        policy.require_queue_scope(root, backend, 'recover')
        for path, adapter, action in [(root/'other', backend, 'recover'), (root, Poison(), 'recover'), (root, backend, 'run')]:
            with pytest.raises(policy.ProductionDispatchDisabled):
                policy.require_queue_scope(path, adapter, action)
        context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(context.run, policy.require_queue_scope, root, backend, 'recover')
            with pytest.raises(policy.ProductionDispatchDisabled):
                future.result()
        with monkeypatch.context() as patch:
            original = os.getpid()
            patch.setattr(os, 'getpid', lambda: original+1)
            with pytest.raises(policy.ProductionDispatchDisabled):
                policy.require_queue_scope(root, backend, 'recover')
    with pytest.raises(policy.ProductionDispatchDisabled):
        policy.require_queue_scope(root, backend, 'recover')


@pytest.mark.parametrize('action', ['run', 'recover', 'audit-results'])
def test_old_cli_subprocess_cannot_import_poison_backend(action, tmp_path):
    repo = Path(__file__).resolve().parents[3]
    # Deliberately absent queue: rejection must precede reading its manifest.
    root = tmp_path/'absent'
    result = subprocess.run([sys.executable, str(repo/'scripts/dsol_paper1/operations/manage_work_queue.py'),
                             action, '--root', str(root), '--execute'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and 'disabled' in result.stderr
    assert not root.exists()
