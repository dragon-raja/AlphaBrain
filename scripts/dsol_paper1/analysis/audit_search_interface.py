"""Thin entrypoint for read-only cached RL-search interface admission."""

import argparse
import json

from AlphaBrain.research.dsol.analysis.search_interface_audit import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Audited INITIAL32 analysis directory")
    parser.add_argument("--historical-recovery", required=True, help="Frozen view-recovery directory")
    parser.add_argument("--output", required=True, help="Must not exist")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(args.source, args.historical_recovery, args.output, args.config)
    print(json.dumps({key: result[key] for key in (
        "status", "golden_comparisons", "baseline_interface_episodes",
        "training_steps", "new_model_calls", "originals_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
