"""Read-only noise/camera audit CLI; writes only an explicitly new output file."""

import argparse

from AlphaBrain.research.dsol.analysis.search_design_audit import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
