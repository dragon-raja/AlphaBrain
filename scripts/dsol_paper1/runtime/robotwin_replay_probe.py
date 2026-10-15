"""Bounded RoboTwin native replay probe; no simulator without --execute."""
import argparse
import json
from benchmarks.RoboTwin.replay import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--denoiser', choices=('oidn', 'optix', 'none'), default='oidn')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(vars(args)))
        return
    print(json.dumps(run(args.config, args.task, args.episode, args.output, args.denoiser), default=str))


if __name__ == '__main__':
    main()
