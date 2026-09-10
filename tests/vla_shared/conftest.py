from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'scripts/vla_shared'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
