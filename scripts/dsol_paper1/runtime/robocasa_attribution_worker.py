"""Thin supplemental worker; phases use separate processes to release GPU RAM."""
import argparse
from pathlib import Path
import subprocess
import sys

from AlphaBrain.research.dsol.data.artifacts import read, path_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.phase:
        from benchmarks.Robocasa365.attribution_evaluation import run
        with path_mapping(read(args.plan).get('path_mappings', [])):
            run(args.plan, args.worker, args.phase, args.output, pilot=args.pilot)
        return
    worker = next(w for w in read(args.plan)['workers'] if w['worker_id'] == args.worker)
    args.output.mkdir(parents=True, exist_ok=False)
    for phase in worker['phases']:
        command = [sys.executable, __file__, '--plan', str(args.plan), '--worker', args.worker,
                   '--output', str(args.output/phase), '--phase', phase, '--execute']
        if args.pilot:
            command.append('--pilot')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
