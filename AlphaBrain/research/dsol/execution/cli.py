"""Safe queue navigation; execution adapters are imported only with --execute."""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import signal

from .queue import load, prepare, recover, run, summary
from .manifest import file_digest, verify_artifacts


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
    else:
        if args.action != 'audit-results' and not args.execute:
            parser.error('run/recover require explicit --execute; adapters can start/manage processes')
        plan, _ = load(args.root)
        verify_artifacts(plan)
        module = importlib.import_module(plan['backend']['module'])
        factory = getattr(module, plan['backend']['factory'])
        source = Path(inspect.getfile(factory)).resolve()
        if not any(Path(a['path']).resolve() == source and a['sha256'] == file_digest(source)
                   for a in plan['artifacts']):
            raise ValueError('Execution adapter source must be explicitly fingerprinted')
        backend = factory(plan, args.root)
        if args.action == 'audit-results':
            from .audit import audit_results
            result=audit_results(args.root,backend,per_status=args.per_status,include_running=args.include_running)
        elif args.action == 'run':
            def interrupt(_signal, _frame):
                raise KeyboardInterrupt('Coordinator stop requested')
            previous = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGHUP)}
            try:
                run(args.root, backend, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds)
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        else:
            recover(args.root, backend)
        if args.action != 'audit-results':result = summary(args.root)
    if args.output is not None:
        if args.output.exists():raise FileExistsError(args.output)
        from .queue import _durable_write
        _durable_write(args.output,result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
