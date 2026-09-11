#!/usr/bin/env python3
"""Legacy-compatible report entry; no plotting or analysis implementation here."""

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))


from pathlib import Path
import importlib
import runpy
import sys

repo = str(_REPOSITORY_ROOT)
if repo not in sys.path:
    sys.path.insert(0, repo)
if __name__ == "__main__":
    runpy.run_module("reports.paper1.initial_results", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("reports.paper1.initial_results")
