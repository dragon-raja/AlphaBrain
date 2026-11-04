"""Legacy queue navigation; dynamic adapter actions are disabled in this release."""
import argparse
import json
from pathlib import Path

from .queue import prepare, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'status', 'run', 'recover', 'resources', 'drain', 'resume', 'cluster-status', 'audit-results'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=3600)
    parser.add_argument('--poll-seconds', type=float, default=1)
    parser.add_argument('--host')
    parser.add_argument('--gpu')
    parser.add_argument('--reason')
    parser.add_argument('--per-status',type=int,default=2)
    parser.add_argument('--include-running',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.action in ('run', 'recover', 'audit-results'):
        parser.error('Legacy dynamic backend actions disabled; use the ledger-controlled CPU simulation entry')
    if args.action == 'cluster-status':
        if args.config is None:parser.error('cluster-status requires --config')
        from .cluster import inspect_cluster
        print(json.dumps(inspect_cluster(args.config),ensure_ascii=False,indent=2))
        return 0
    if args.root is None:parser.error('This action requires --root')
    if args.action == 'resources':
        from .resources import resource_status
        result = resource_status(args.root)
    elif args.action in ('drain', 'resume'):
        if not args.execute or not args.host or args.gpu is None or not args.reason:
            parser.error('drain/resume require --execute, --host, --gpu and --reason')
        from .resources import set_drain
        result = set_drain(args.root, args.host, args.gpu, reason=args.reason, enabled=args.action == 'drain')
    elif args.action == 'status':
        result = summary(args.root)
    elif args.action == 'prepare':
        if args.manifest is None:
            parser.error('prepare requires --manifest')
        prepare(json.loads(args.manifest.read_text()), args.root)
        result = summary(args.root)
    if args.output is not None:
        if args.output.exists():raise FileExistsError(args.output)
        from .queue import _durable_write
        _durable_write(args.output,result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
