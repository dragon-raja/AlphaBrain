"""Replay frozen policy inputs without constructing or rendering an environment."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', default='pi05_pretrain_human300_all_lora_fast25')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or not 2 <= args.repeats <= 8:
        parser.error('Explicit --execute and 2..8 repeats required')
    from benchmarks.Robocasa365.policy_replay import replay_inputs
    result = replay_inputs(args.record, args.checkpoint, args.config, args.output, args.repeats)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
