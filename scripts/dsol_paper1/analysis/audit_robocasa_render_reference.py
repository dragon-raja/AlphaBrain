"""Read-only numerical audit; writes a new report, never a data release."""

import argparse
from pathlib import Path
from AlphaBrain.research.dsol.analysis.render_reference_audit import audit_reference
from AlphaBrain.research.dsol.data.artifacts import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--localization', type=Path, required=True)
    parser.add_argument('--repetition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_reference(args.localization, args.repetition)
    write(args.output, result)
    print(result['status'])


if __name__ == '__main__':
    main()
