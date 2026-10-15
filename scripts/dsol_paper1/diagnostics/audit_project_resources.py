"""Validate portable resources and optional machine-local bindings, without GPU."""
import argparse
import json
from pathlib import Path
from AlphaBrain.research.dsol.data.resource_graph import audit_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--locations')
    args = parser.parse_args()
    graph = json.loads(Path(args.manifest).read_text())
    locations = json.loads(Path(args.locations).read_text()) if args.locations else None
    result = audit_graph(graph, locations)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
