"""CPU-only entry to ledger-fenced standard queues; no dynamic backend imports."""
import argparse
import json
from pathlib import Path

from .controller_ledger import ControllerLedger
from .controlled_queue import recover_controlled_queue, run_controlled_queue
from .simulation_backend import SimulationBackend
from . import queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'recover', 'status'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--custody', type=Path, required=True)
    parser.add_argument('--owner', type=Path)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--permit', type=Path)
    parser.add_argument('--backend-state', type=Path)
    parser.add_argument('--authority-config', type=Path,
                        help='Explicit live-channel CPU backend configuration; never enables production')
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--explicit-recovery', action='store_true')
    parser.add_argument('--previous-owner', type=Path)
    parser.add_argument('--previous-epoch', type=int)
    args = parser.parse_args(argv)
    if args.execute:
        parser.error('Production dispatch is disabled in this release')
    ledger = ControllerLedger(args.custody)
    if args.action == 'status':
        print(json.dumps(dict(queue=queue.summary(args.root), ledger=ledger.read()[1]), indent=2))
        return 0
    if not args.simulate or args.owner is None or args.backend_state is None:
        parser.error('Only --simulate with explicit --owner and --backend-state is supported')
    if args.action == 'run' and (args.permit is None or args.epoch is None):
        parser.error('run requires an independent --permit and --epoch')
    plan, _ = queue.load(args.root)
    budget, custody = ledger.read()
    if custody.get('simulation_only') is not True:
        parser.error('Original paused custody is not a simulation ledger; prepare an isolated copy')
    keys = {t['key'] for t in plan['tasks']}
    limits = {t['key']: t['limits'] for t in budget['tasks'] if t['key'] in keys}
    if args.authority_config is not None:
        from .authority_backend import AuthoritySimulationBackend
        backend = AuthoritySimulationBackend(args.backend_state, limits,
                                              config=json.loads(args.authority_config.read_text()))
    else:
        backend = SimulationBackend(args.backend_state, limits)
    owner = json.loads(args.owner.read_text())
    if args.action == 'recover' or args.explicit_recovery:
        if args.previous_owner is None or args.previous_epoch is None:
            parser.error('Explicit recovery requires --previous-owner and --previous-epoch')
        epoch = recover_controlled_queue(args.root, ledger, json.loads(args.previous_owner.read_text()),
                                         args.previous_epoch, owner, backend, simulate=True)
    else:
        # A new process never silently attaches to a previous session. Existing
        # custody requires the explicit recovery path and a new epoch permit.
        epoch = ledger.claim(owner)
    if args.action == 'run':
        if args.epoch != epoch:
            raise RuntimeError('Requested epoch differs; dispatch remains disabled')
        run_controlled_queue(args.root, ledger, owner, epoch, args.permit, backend, simulate=True)
    print(json.dumps(dict(queue=queue.summary(args.root), epoch=epoch,
                         status=ledger.read()[1]['status'], dispatch_enabled=False, simulation_only=True), indent=2))
    return 0
