"""Explicit bounded PPO search run; no training on import or without --execute."""

import argparse
import json

from AlphaBrain.research.dsol.analysis.search_training import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run(args.source, args.output, args.config, execute=args.execute)
    print(json.dumps({k: result[k] for k in ("status", "total_steps", "new_vla_calls")}, indent=2))


if __name__ == "__main__":
    main()
