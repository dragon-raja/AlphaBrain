"""Summarize a completed renderer calibration without applying its proposal."""

import argparse
from pathlib import Path
from AlphaBrain.research.dsol.analysis.render_calibration_results import summarize_calibration
from AlphaBrain.research.dsol.data.artifacts import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize_calibration(args.calibration)
    write(args.output, result)
    print(result['proposal'])


if __name__ == '__main__':
    main()
