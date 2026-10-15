"""Print a read-only metadata/path audit; never move data or certify rollouts."""
import argparse
import json
from AlphaBrain.research.dsol.data.resource_catalog import audit_file, publish_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--publish', action='store_true', help='Explicitly sync shared copy after checks; archive old bytes')
    args = parser.parse_args()
    if args.publish:
        print(json.dumps(publish_catalog(args.catalog), ensure_ascii=False, indent=2))
    result = audit_file(args.catalog)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
