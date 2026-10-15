"""Review actual camera geometry and zero-pixel strata without new evaluation."""
import argparse
from AlphaBrain.research.dsol.analysis.geometry_visibility import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)
