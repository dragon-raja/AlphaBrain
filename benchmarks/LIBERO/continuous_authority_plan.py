"""CPU preparation and explicit controller entry for immutable LIBERO bindings."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from AlphaBrain.research.dsol.execution import queue
from AlphaBrain.research.dsol.execution.authority_wire import Channel, identity
from AlphaBrain.research.dsol.execution.controller_ledger import ControllerLedger, prepare
from AlphaBrain.research.dsol.execution.controlled_queue import run_controlled_queue, recover_controlled_queue
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, validate
from AlphaBrain.research.dsol.execution.model_authority import ModelAuthorityBackend
from .continuous_authority import load_binding, source_identity


def with_worker_capacity(binding, workers_by_host, *, stages=('capacity', 'formal')):
    """Create a new resource binding, never edit a live binding or task identity."""
    from AlphaBrain.research.dsol.execution.resources import worker_slots
    if set(workers_by_host) != set(binding['nodes']) or not stages or not set(stages).issubset(binding['budget']['stages']):
        raise ValueError('Explicit complete node/stage resource coverage required')
    result = deepcopy(binding)
    for host, workers in workers_by_host.items():
        # Validate even when a host has no tasks in the selected stage.
        worker_slots(host, [0], workers_per_gpu=workers)
        result['nodes'][host]['workers_per_gpu'] = workers
    for spec in result['tasks'].values():
        if spec['stage'] in stages:
            devices = list(dict.fromkeys(str(s['gpu_id']) for s in spec['eligible_slots']))
            spec['eligible_slots'] = worker_slots(spec['host'], devices, workers_per_gpu=workers_by_host[spec['host']])
    return result


def plans(binding,binding_path):
    from AlphaBrain.research.dsol.execution.model_plan import plans as shared
    return shared(binding,binding_path)


def prepare_controller(binding_path,root,transport_config):
    from AlphaBrain.research.dsol.execution.model_plan import prepare_controller as shared
    return shared(binding_path,root,transport_config,load_binding=load_binding)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inspect','prepare','run','recover'])
    parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--root',type=Path)
    parser.add_argument('--transport',type=Path);parser.add_argument('--stage',choices=['admission','capacity','formal'])
    parser.add_argument('--permit',type=Path);parser.add_argument('--owner',type=Path);parser.add_argument('--epoch',type=int)
    parser.add_argument('--previous-owner',type=Path);parser.add_argument('--previous-epoch',type=int)
    parser.add_argument('--explicit-recovery',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--simulate',action='store_true')
    args=parser.parse_args(argv);binding=load_binding(args.binding)
    if args.action=='inspect':
        data=plans(binding,args.binding)
        print(json.dumps(dict(status='CPU_PREPARED_NOT_AUTHORIZED',binding_sha256=digest(binding),
            stages={k:dict(blocks=len(v['tasks']),episodes=sum(len(t['addresses']) for t in v['tasks'])) for k,v in data.items()},dispatch_enabled=False),indent=2));return 0
    if args.root is None:parser.error('--root required')
    if args.action=='prepare':
        if args.transport is None:parser.error('--transport required')
        print(json.dumps(prepare_controller(args.binding,args.root,json.loads(args.transport.read_text()))));return 0
    real=binding['mode']=='libero'
    if real and (not args.execute or args.simulate):parser.error('Real model entry requires --execute and a separate explicit permit')
    if not real and (not args.simulate or args.execute):parser.error('CPU contract requires --simulate only')
    if args.stage is None or args.owner is None:parser.error('--stage and --owner required')
    # Reject absent/template permits before constructing any backend/channel.
    if args.action=='run':
        if args.permit is None or args.epoch is None:parser.error('Explicit current --permit and --epoch required')
        permit=json.loads(args.permit.read_text())
        if permit.get('authorization')!=('EXPLICIT_MODEL_STAGE_V1' if real else 'CPU_SIMULATION_ONLY'):
            parser.error('Draft approval is not an execution permit')
        if real and permit.get('binding_sha256')!=digest(binding):parser.error('Permit binding mismatch')
    ledger=ControllerLedger(args.root/'custody');owner=json.loads(args.owner.read_text())
    config=json.loads((args.root/'transport.json').read_text())
    if config['binding_sha256']!=digest(binding) or config['mode']!=binding['mode']:
        raise ValueError('Controller transport binding changed')
    if set(config['peers'])!=set(binding['nodes']):raise ValueError('Transport peers differ from binding')
    for host,target in config['peers'].items():
        node=binding['nodes'][host]
        if target.get('agent')!='benchmarks.LIBERO.continuous_authority':raise ValueError('Unregistered worker agent')
        if target['expected']['identity']!=node['identity'] or target['expected']['source']!=source_identity():
            raise ValueError('Worker identity/source differs from reviewed release')
        # Production descriptors are fixed by the approved binding. CPU tests
        # may explicitly use a local transport with the same fixed agent.
        if real and any(target.get(k)!=node[k] for k in ('python','source','environment')):
            raise ValueError('Worker interpreter/source/environment differs from binding')
    plan=plans(binding,args.binding)[args.stage]
    if queue.load(args.root/args.stage)[0]!=plan:raise ValueError('Prepared queue changed')
    keys={t['key'] for t in plan['tasks']}
    limits={t['key']:t['limits'] for t in binding['budget']['tasks'] if t['key'] in keys}
    backend=ModelAuthorityBackend(args.root/'backend',limits,config=config)
    if args.action=='recover' or args.explicit_recovery:
        if args.previous_owner is None or args.previous_epoch is None:parser.error('Explicit previous owner/epoch required')
        epoch=recover_controlled_queue(args.root/args.stage,ledger,json.loads(args.previous_owner.read_text()),
            args.previous_epoch,owner,backend,simulate=not real,execute=real)
        if args.action=='recover':
            print(json.dumps(dict(status='RECOVERED_PAUSED',epoch=epoch,dispatch_enabled=False)));return 0
    else:
        epoch=ledger.claim(owner)
    if epoch!=args.epoch:raise RuntimeError('Current epoch differs')
    run_controlled_queue(args.root/args.stage,ledger,owner,epoch,args.permit,backend,simulate=not real,execute=real,
                         timeout_seconds=max(1,permit['finish_until']-__import__('time').time()),poll_seconds=.05)
    print(json.dumps(dict(status=ledger.read()[1]['status'],dispatch_enabled=False)));return 0


if __name__=='__main__':raise SystemExit(main())
