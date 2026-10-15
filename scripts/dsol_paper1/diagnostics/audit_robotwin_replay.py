"""CPU-only replay postflight, including native renderer error logs."""
import argparse
import json
from pathlib import Path
from benchmarks.RoboTwin.replay_admission import audit_attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit_attempt(args.attempt, args.log)
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
