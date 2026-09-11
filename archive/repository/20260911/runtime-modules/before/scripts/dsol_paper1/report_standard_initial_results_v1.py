#!/usr/bin/env python3
"""Legacy-compatible report entry; no plotting or analysis implementation here."""

from pathlib import Path
import importlib
import runpy
import sys

repo = str(Path(__file__).resolve().parents[2])
if repo not in sys.path:
    sys.path.insert(0, repo)
if __name__ == "__main__":
    runpy.run_module("reports.paper1.initial_results", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("reports.paper1.initial_results")
