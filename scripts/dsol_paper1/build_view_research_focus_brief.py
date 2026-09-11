#!/usr/bin/env python3
"""Compatibility entry for historical report callers; implementation in reports/."""

from pathlib import Path
import importlib
import runpy
import sys

repo = str(Path(__file__).resolve().parents[2])
if repo not in sys.path:
    sys.path.insert(0, repo)
if __name__ == "__main__":
    runpy.run_module("reports.paper1.legacy_style", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("reports.paper1.legacy_style")
