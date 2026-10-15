"""Explicit one-shot continuation of the authorized canonical/perturbed panel."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('canonical', 'output', 'record'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--full-config', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    from AlphaBrain.research.dsol.data.xr1_continuation import supervise, supervise_full
    if args.full_config:
        if any((args.canonical, args.output, args.record)):
            parser.error('Do not mix full-config with five-view arguments')
        supervise_full(args.full_config)
    else:
        if not all((args.canonical, args.output, args.record)):
            parser.error('canonical, output and record are required')
        supervise(args.canonical, args.output, args.record)


if __name__ == '__main__':
    main()
