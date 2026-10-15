"""Run one frozen resident P0 matrix worker, or its bounded engineering pilot."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    from benchmarks.Robocasa365.evaluation_matrix import run_worker
    result = run_worker(args.plan, args.worker, args.output, pilot=args.pilot, resume=args.resume)
    print(result, flush=True)


if __name__ == '__main__':
    main()
