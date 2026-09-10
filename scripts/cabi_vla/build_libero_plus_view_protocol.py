"""Legacy import/CLI compatibility; implementation lives in scripts.vla_shared."""
from pathlib import Path
import importlib
import runpy
import sys

root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.insert(0, root)

if __name__ == "__main__":
    runpy.run_module("scripts.vla_shared.build_libero_plus_view_protocol", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("scripts.vla_shared.build_libero_plus_view_protocol")
