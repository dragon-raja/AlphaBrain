"""Resource admission CLI; default is plan inspection, --execute performs IO."""
import argparse
import json
from pathlib import Path
from AlphaBrain.research.dsol.data.resource_import import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--record', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        run(args.plan, args.root, args.record)
    else:
        print(json.dumps(json.loads(Path(args.plan).read_text()), indent=2))


if __name__ == '__main__':
    main()
