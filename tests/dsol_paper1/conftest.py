"""Local test imports only; do not initialize a simulator or policy server."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts/dsol_paper1", ROOT / "scripts/vla_shared", Path(__file__).parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
