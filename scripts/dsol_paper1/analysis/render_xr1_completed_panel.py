"""Complete admitted XR-1 figures in a CPU plotting environment; no rollouts."""
from pathlib import Path
import argparse
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from AlphaBrain.research.dsol.analysis.xr1_fullview import render_completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--panel-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(render_completed(args.analysis_root, args.panel_root, args.output))


if __name__ == '__main__':
    main()
