"""CPU-only fixtures for the actual live model transport and CLI."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from tests.dsol_paper1.helpers.controller_cases import make_case, OWNER
from tests.dsol_paper1.paths import ROOT
from AlphaBrain.research.dsol.execution.authority_wire import Channel, identity
from AlphaBrain.research.dsol.execution.controller_ledger import ControllerLedger
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest
from AlphaBrain.research.dsol.execution.model_authority import ModelAuthorityBackend
from benchmarks.LIBERO.continuous_authority import source_identity
from benchmarks.LIBERO.continuous_authority_plan import prepare_controller, plans


def run_model_case(case):
    from AlphaBrain.research.dsol.execution.controlled_queue import run_controlled_queue
    return run_controlled_queue(case['queue'], case['ledger'], OWNER, 1, case['permit'], case['backend'],
                                simulate=True, poll_seconds=.01, timeout_seconds=30)


def make_model_case(root, *, claim=True, transport=None, checkpoint=None, slots=None, checks=2, resident=False, task_count=3, persistent_inspection=False, parallel_launches=1, benchmark="libero"):
    worker_source_identity=source_identity;controller_prepare=prepare_controller;make_plans=plans
    agent='benchmarks.LIBERO.continuous_authority';schema='LIBERO_LIVE_BINDING_V1'
    if benchmark in ('gr1','gr1_trial','gr1_block','gr1_prepared'):
        from benchmarks.Robocasa_tabletop import continuous_authority as gr1
        worker_source_identity=gr1.source_identity;controller_prepare=gr1.prepare_controller;make_plans=gr1.plans
        agent='benchmarks.Robocasa_tabletop.continuous_authority';schema='GR1_CPU_BINDING_V1'
    elif benchmark!='libero':raise ValueError('Unknown CPU fixture benchmark')
    root=Path(root);base=make_case(root/'fixture',peers=['fresh'],task_count=task_count)
    budget=base['ledger'].read()[0]
    if benchmark in ('gr1_block','gr1_prepared'):
        calls=36 if benchmark=='gr1_prepared' else 8
        budget['limits'].update(calls=calls*task_count, episodes=2*task_count)
        for task in budget['tasks']:task['limits'].update(calls=calls,episodes=2)
    if benchmark=='gr1_trial':
        budget['limits']['calls']=4*task_count
        for task in budget['tasks']:task['limits']['calls']=4
    if slots is not None:base['plan']['slots']=deepcopy(slots)
    executable=Path(sys.executable).resolve()
    node=dict(identity=identity(),python_resolved=str(executable),python_sha256=file_digest(executable),source_artifacts=[])
    if transport is not None:node=deepcopy(transport['node'])
    specs={t['key']:dict(addresses=t['addresses'],output=t['output'],host='fresh',stage='admission',
        identity_experiment_id=base['plan']['experiment_id'],eligible_slots=base['plan']['slots'],
        request=({'synthetic_prepared':True} if benchmark=='gr1_prepared' else {'synthetic_block':True} if benchmark=='gr1_block' else {'synthetic_trial':True} if benchmark=='gr1_trial' else dict(synthetic=True)),
        limits=next(b['limits'] for b in budget['tasks'] if b['key']==t['key']))
        for t in base['plan']['tasks']}
    binding=dict(schema=schema,mode='cpu_contract',campaign=str(root/'fixture/protocol.json'),
                 nodes=dict(fresh=node),tasks=specs,budget=budget)
    if resident:binding['resident_workers']=True
    if checks!=2:binding['cpu_contract_checks']=checks
    binding_path=root/'binding.json';binding_path.write_text(json.dumps(binding))
    target=dict(transport='local',python=sys.executable,source=str(ROOT),root=str(root/'worker'),
                agent=agent,binding=str(binding_path),expected=dict(source=worker_source_identity()))
    if transport is not None:target.update(transport['target']);target['binding']=str(binding_path)
    with Channel(target,prepare=True,agent=target['agent'],expected_source=worker_source_identity()) as channel:
        hello=channel.hello;channel.wire.send(dict(kind='PROBE',session=hello['session']))
        assert channel.wire.receive()['kind']=='HEALTHY'
    target['expected']={k:hello[k] for k in ('source','identity','store_id')}
    checkpoint=Path(checkpoint) if checkpoint else root/'checkpoints';checkpoint.mkdir(parents=True)
    config=dict(schema='LIVE_MODEL_AUTHORITY_V1',mode='cpu_contract',binding_sha256=digest(binding),authority=identity(),
                checkpoint_directory=str(checkpoint),peers=dict(fresh=target),persistent_inspection=persistent_inspection,parallel_launches=parallel_launches)
    if resident:config['resident_workers']=True
    control=root/'control';controller_prepare(binding_path,control,config)
    ledger=ControllerLedger(control/'custody')
    if claim:ledger.claim(OWNER)
    plan=make_plans(binding,binding_path)['admission']
    permit=dict(authorization='CPU_SIMULATION_ONLY',ledger_plan_sha256=digest(budget),queue_sha256=digest(plan),
        owner=OWNER,epoch=1,stage='admission',issued_at=time.time()-1,dispatch_until=time.time()+300,finish_until=time.time()+600)
    permit_path=control/'permit.json';permit_path.write_text(json.dumps(permit))
    backend=ModelAuthorityBackend(control/'backend',{t['key']:t['limits'] for t in budget['tasks']},config=config)
    return dict(root=control,binding_path=binding_path,binding=binding,ledger=ledger,plan=plan,backend=backend,permit=permit_path,
                queue=control/'admission',worker=root/'worker')
