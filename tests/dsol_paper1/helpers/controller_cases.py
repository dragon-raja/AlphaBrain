"""Isolated CPU controller fixtures; remote targets require explicit arguments."""
import json
from pathlib import Path
import sys
import time
from AlphaBrain.research.dsol.execution import queue
from AlphaBrain.research.dsol.execution.controller_ledger import ControllerLedger, prepare
from AlphaBrain.research.dsol.execution.controlled_queue import recover_controlled_queue, run_controlled_queue
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key
from AlphaBrain.research.dsol.execution.simulation_backend import SimulationBackend
OWNER = dict(host='controller', boot_id='boot-a', session='session-a')
NEXT = dict(host='controller', boot_id='boot-b', session='session-b')
from tests.dsol_paper1.paths import ROOT as REPO


def make_case(root, *, claim=True, peers=None, task_count=3):
    root.mkdir(parents=True, exist_ok=True)
    protocol = root / 'protocol.json'; protocol.write_text('{"synthetic": true}')
    tasks = []
    for i in range(task_count):
        addresses = [dict(model='synthetic', task='task', initial=str(i), view='fixed',
                          stream='execution', repeat=0, noise='frozen')]
        tasks.append(dict(key=task_key('cpu-launch-test', file_digest(protocol), addresses),
                          addresses=addresses, output=str(root / 'unused-output' / str(i)), payload={}))
    plan = dict(format='dsol_work_queue_v1', experiment_id='cpu-launch-test',
                protocol_sha256=file_digest(protocol), tasks=tasks,
                artifacts=[dict(path=str(protocol), sha256=file_digest(protocol))],
                slots=[dict(host='fresh', id='cpu-fake')],
                backend=dict(module='MUST_NOT_IMPORT_PRODUCTION_BACKEND', factory='factory'))
    queue.prepare(plan, root / 'queue')
    budget = dict(campaign_sha256=plan['protocol_sha256'], stages=['admission', 'capacity'],
                  peers=peers or ['fresh', 'gnu', 'video'], limits=dict(calls=2*task_count, episodes=task_count), tasks=[
                      dict(key=t['key'], addresses=t['addresses'], stage='admission', limits=dict(calls=2, episodes=1))
                      for t in tasks])
    prepare(root / 'custody', budget, simulation_only=True)
    ledger = ControllerLedger(root / 'custody')
    if claim: ledger.claim(OWNER)
    backend = SimulationBackend(root / 'backend', {t['key']: t['limits'] for t in budget['tasks']})
    value = dict(authorization='CPU_SIMULATION_ONLY', ledger_plan_sha256=digest(budget),
                 queue_sha256=digest(plan), owner=OWNER, epoch=1, stage='admission',
                 issued_at=time.time() - 5, dispatch_until=time.time() + 600, finish_until=time.time() + 1200)
    permit = root / 'permit.json'; permit.write_text(json.dumps(value))
    return dict(root=root, plan=plan, ledger=ledger, backend=backend, permit=permit)


def run(case, owner=OWNER, epoch=1, **kwargs):
    kwargs.setdefault('timeout_seconds', 2)
    return run_controlled_queue(case['root'] / 'queue', case['ledger'], owner, epoch,
                                case['permit'], case['backend'], simulate=True,
                                poll_seconds=0, **kwargs)


def update_permit(case, **changes):
    data = json.loads(case['permit'].read_text()); data.update(changes)
    case['permit'].write_text(json.dumps(data))


def recover(case, ledger=None):
    ledger = ledger or case['ledger']
    return recover_controlled_queue(case['root'] / 'queue', ledger, OWNER, 1, NEXT,
                                     case['backend'], simulate=True)


def make_authority_case(root, *, claim=True, target=None, mirror=None):
    from AlphaBrain.research.dsol.execution.authority_backend import AuthoritySimulationBackend, prepare_target
    from AlphaBrain.research.dsol.execution.authority_wire import identity
    case = make_case(root, claim=claim, peers=['fresh'])
    target = dict(target) if target is not None else dict(transport='local', python=sys.executable, source=str(REPO), root=str(root/'remote-store'))
    target['expected'] = prepare_target(target)
    mirror = Path(mirror) if mirror is not None else root/'persistent-checkpoints'
    mirror.mkdir(parents=True, exist_ok=True)
    config = dict(schema='CPU_AUTHORITY_V1', authority=identity(), checkpoint_directory=str(mirror),
                  peers=dict(fresh=target))
    case['backend'] = AuthoritySimulationBackend(root/'authority-backend', case['backend'].limits, config=config)
    return case
