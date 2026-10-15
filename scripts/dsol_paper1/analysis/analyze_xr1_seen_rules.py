"""Read-only seen-calibrated fixed-view diagnosis; no training or rollouts."""
import argparse
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from AlphaBrain.research.dsol.analysis.xr1_fullview import run_seen_rule_diagnostic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root',type=Path,required=True)
    parser.add_argument('--panel-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run_seen_rule_diagnostic(args.analysis_root,args.panel_root,args.output)
    print(args.output)


if __name__=='__main__':main()
