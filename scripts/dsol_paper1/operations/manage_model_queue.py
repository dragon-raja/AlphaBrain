"""Compose the fixed LIBERO model adapter; no dynamic backend or implicit start."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.LIBERO.continuous_authority_plan import main

if __name__ == '__main__':
    raise SystemExit(main())
