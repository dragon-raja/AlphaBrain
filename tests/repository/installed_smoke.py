"""Explicit wheel-only smoke: invoke with python -I, outside the checkout."""
import importlib
import json
from pathlib import Path
import sys


def main():
    modules = [
        'AlphaBrain.common.camera_pose',
        'AlphaBrain.research.dsol.execution.process_backend',
        'AlphaBrain.research.dsol.execution.source_release',
        'benchmarks.LIBERO.simulation_inputs',
        'benchmarks.LIBERO.continuous_authority',
        'benchmarks.Robocasa_tabletop.block_runtime',
        'benchmarks.Robocasa_tabletop.continuous_authority',
    ]
    paths = {}
    for name in modules:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(Path(sys.prefix).resolve()), (name, 'checkout fallback')
        paths[name] = str(path)
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    from AlphaBrain.research.dsol.execution.process_backend import LocalProcessBackend
    try:
        LocalProcessBackend.start(None, None, None, None)
    except ProductionDispatchDisabled:
        pass
    else:
        raise AssertionError('Direct launch was not refused')
    from benchmarks.Robocasa_tabletop.continuous_authority import execute_block
    try:
        execute_block()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unadmitted GR1 runtime was enabled')
    for name in modules[-1:], modules[4:5]:
        identity = importlib.import_module(name[0]).source_identity()
        assert len(identity) == 64
    assert not {'torch', 'jax', 'mujoco', 'wandb'} & set(sys.modules)
    print(json.dumps(dict(status='PASS', modules=paths, gpu_used=False), indent=2))


if __name__ == '__main__':
    main()
