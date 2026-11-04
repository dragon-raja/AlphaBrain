"""Ledger-protected standard queue entry; this release permits CPU fakes only."""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from AlphaBrain.research.dsol.execution.controller_cli import main


if __name__ == '__main__':
    raise SystemExit(main())
