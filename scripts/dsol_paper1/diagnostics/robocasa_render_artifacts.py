"""Explicit per-snapshot pixel/geometry localization; does not admit data."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    from benchmarks.Robocasa365.render_artifacts import inspect_artifact
    result = inspect_artifact(args.snapshot, args.output)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
