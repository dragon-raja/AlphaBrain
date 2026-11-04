"""Current-release dispatch closure, not an OS or hostile-Python sandbox.

There is no environment override or legacy permit fallback. Only registered
controller adapters may enter the shared queue; model scope needs a live grant. Historical
frozen checkouts are outside this policy and must not be launched.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import threading


class ProductionDispatchDisabled(RuntimeError):
    pass


def reject_production(entry):
    allowed = _model_scope.get()
    if allowed is not None and entry in _MODEL_ENTRIES:
        pid, thread, guard = allowed
        if pid == os.getpid() and thread == threading.get_ident():
            guard()
            return
    raise ProductionDispatchDisabled(
        f'Production dispatch disabled in this release: {entry}; '
        'only the ledger-controlled CPU simulation entry is available')


_scope = ContextVar('controlled_cpu_queue_scope', default=None)
_model_scope = ContextVar('live_model_worker_scope', default=None)
_MODEL_ENTRIES = frozenset({
    'benchmarks/LIBERO/prefix_pilot.py:run',
    'benchmarks/LIBERO/score_prefix_runtime.py:PrefixRuntime.infer',
    'benchmarks/LIBERO/rldx_search_runtime.py:RLDXSearchRuntime.__init__',
    'benchmarks/LIBERO/rldx_search_runtime.py:RLDXSearchRuntime.infer',
    'benchmarks/LIBERO/rldx_spatial_panel.py:run',
    'benchmarks/LIBERO/rldx_runtime.py:RLDXRuntime.bind_task',
    'benchmarks/LIBERO/controlled_rldx_operation.py:run',
    'benchmarks/LIBERO/controlled_rldx.py:run',
    'benchmarks/LIBERO/rldx_runtime.py:RLDXRuntime.__init__',
    'benchmarks/LIBERO/rldx_runtime.py:RLDXRuntime.infer',
    'benchmarks/LIBERO/controlled_groot_admission.py:run',
    'benchmarks/LIBERO/groot_runtime.py:GrootRuntime.__init__',
    'benchmarks/LIBERO/groot_runtime.py:GrootRuntime.infer',
    'benchmarks/LIBERO/groot_runtime.py:NativeProcessor.__init__',
    'benchmarks/LIBERO/groot_runtime.py:NativeProcessor.inputs',
    'benchmarks/LIBERO/controlled_admission.py:run',
    'benchmarks/LIBERO/controlled_target_evaluation.py:run',
    'benchmarks/LIBERO/controlled_target_recovery.py:run',
    'benchmarks/LIBERO/controlled_target_panel.py:run',
    'benchmarks/LIBERO/legacy/fastwam_runtime.py:load_components',
    'benchmarks/LIBERO/legacy/fastwam_runtime.py:FastWAMRuntime.__init__',
    'benchmarks/LIBERO/legacy/fastwam_runtime.py:FastWAMRuntime.infer',
    'benchmarks/LIBERO/controlled_recovery.py:run',
    'benchmarks/LIBERO/controlled_training.py:run',
    'benchmarks/LIBERO/continuous_training.py:LiveEpisodes.__init__',
    'benchmarks/LIBERO/native_view_panel.py:run',
    'benchmarks/LIBERO/continuous_evaluation.py:run',
    'benchmarks/LIBERO/continuous_nominal.py:run',
    'benchmarks/LIBERO/search_runtime.py:PolicyRuntime.__init__',
    'benchmarks/LIBERO/search_runtime.py:PolicyRuntime.infer',
    'benchmarks/LIBERO/native_search_runtime.py:NativePolicyRuntime.__init__',
    'benchmarks/LIBERO/native_search_runtime.py:NativePolicyRuntime.infer',
    'benchmarks/LIBERO/search_live.py:simulator',
})


@contextmanager
def _live_model_scope(launch, guard):
    """Entered only by the bound remote agent after its live GO handshake.

    This is a cooperative process/thread boundary, not a hostile-code sandbox.
    No old fleet, training controller or direct CLI receives this scope.
    """
    if (launch.get('operation') != 'LIBERO_BLOCK'
            or launch.get('permit', {}).get('authorization') != 'EXPLICIT_MODEL_STAGE_V1'):
        reject_production('missing live model grant')
    guard()
    token = _model_scope.set((os.getpid(), threading.get_ident(), guard))
    try:
        yield
    finally:
        _model_scope.reset(token)


@contextmanager
def _controlled_scope(root, backend, operation):
    # Exact classes: a subclass must not substitute a real start or recovery.
    from .controlled_queue import LedgerBackend, LedgerRecoveryBackend
    from .simulation_backend import SimulationBackend
    from .authority_backend import AuthoritySimulationBackend
    from .model_authority import ModelAuthorityBackend
    expected = {'run': LedgerBackend, 'recover': LedgerRecoveryBackend}
    if (type(backend) is not expected.get(operation)
            or type(backend.backend) not in (SimulationBackend, AuthoritySimulationBackend, ModelAuthorityBackend)):
        reject_production('unregistered queue backend')
    value = (Path(root).resolve(), backend, operation, os.getpid(), threading.get_ident())
    token = _scope.set(value)
    try:
        yield
    finally:
        _scope.reset(token)


def require_queue_scope(root, backend, operation):
    value = _scope.get()
    if value is None:
        reject_production('uncontrolled queue '+operation)
    path, allowed, action, pid, thread = value
    if (Path(root).resolve() != path or backend is not allowed or action != operation
            or pid != os.getpid() or thread != threading.get_ident()):
        reject_production('queue scope mismatch')
