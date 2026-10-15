"""Freeze a native-render P0 matrix plan and source; does not start GPUs."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--dualhost', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-only-render', action='store_true')
    args = parser.parse_args()
    from AlphaBrain.research.dsol.data.evaluation_matrix import prepare_plan
    plan = prepare_plan(args.program, args.dualhost, args.output, args.source,
                        policy_only_render=args.policy_only_render)
    print('Frozen', len(plan['workers']), 'workers; no GPU jobs launched')


if __name__ == '__main__':
    main()
