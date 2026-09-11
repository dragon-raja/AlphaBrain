"""Compatibility CLI/import for historical CABI; not the Paper 1 policy server."""
from pathlib import Path
import importlib
import runpy
import sys
repo=str(Path(__file__).resolve().parents[2])
if repo not in sys.path:sys.path.insert(0,repo)
if __name__=='__main__':
    runpy.run_module('scripts.vla_shared.historical_pi05_policy_server',run_name='__main__')
else:
    sys.modules[__name__]=importlib.import_module('scripts.vla_shared.historical_pi05_policy_server')
