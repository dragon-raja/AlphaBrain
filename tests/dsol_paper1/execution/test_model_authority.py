"""Actual model-entry transport tested with its closed CPU contract executor."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from tests.dsol_paper1.helpers.controller_cases import OWNER, NEXT
from tests.dsol_paper1.helpers.model_cases import make_model_case
from tests.dsol_paper1.paths import ROOT
from AlphaBrain.research.dsol.execution import authority_wire, dispatch_policy
from AlphaBrain.research.dsol.execution.controlled_queue import run_controlled_queue, recover_controlled_queue
from AlphaBrain.research.dsol.execution.manifest import digest
from benchmarks.LIBERO.continuous_authority import validate_launch, execute_block


def run(case):
    return run_controlled_queue(case['queue'],case['ledger'],OWNER,1,case['permit'],case['backend'],
                                simulate=True,poll_seconds=.01,timeout_seconds=30)


@pytest.fixture
def case(tmp_path):return make_model_case(tmp_path)


def test_actual_live_model_transport_cpu_contract(case):
    before=set(sys.modules)
    assert run(case)['status']=='COMPLETE'
    state=case['ledger'].read()[1]
    assert state['dispatch_enabled'] is False and len(state['tasks'])==3
    assert all(r['receipt']['simulation_only'] for r in state['tasks'].values())
    assert not {'torch','mujoco','wandb'}.intersection(set(sys.modules)-before)
    rows=[json.loads(p.read_text()) for p in (case['worker']/'tasks').glob('*.json')]
    assert all(r['starts']==1 and r['status']=='COMPLETE' for r in rows)
    assert all(c['sequence']==2 for c in case['backend'].channels.values())


@pytest.mark.parametrize('fails', [False, True])
def test_launch_validation_timeout_does_not_extend_runtime_grants(case, monkeypatch, fails):
    backend=case['backend'];backend.launch_timeout=300
    original=authority_wire.Wire.receive
    observed=[]
    def receive(wire):
        if wire.timeout<=10:
            observed.append(wire)
            if fails:raise ConnectionError('lost validation channel')
        return original(wire)
    monkeypatch.setattr(authority_wire.Wire,'receive',receive)
    if fails:
        with pytest.raises(ConnectionError):run(case)
        state=case['ledger'].read()[1]
        row=next(iter(state['tasks'].values()))
        assert not state['dispatch_enabled']
        assert row['worker_claim']['outcome']=='INTENT'
        assert row['status']=='RESERVED'
        with pytest.raises(RuntimeError):run(case)
    else:
        assert run(case)['status']=='COMPLETE'
        assert all(c['channel'].wire.timeout==60 for c in backend.channels.values())
    assert observed and all(wire.timeout==60 for wire in observed)


def test_long_preparation_keeps_peers_live_without_resetting_deadline(case,monkeypatch):
    from types import SimpleNamespace
    from AlphaBrain.research.dsol.execution import model_authority
    backend=case['backend'];backend.launch_timeout=60
    now=[0.];probes=[]
    monkeypatch.setattr(model_authority.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(backend,'probe',lambda peer:probes.append(peer) or True)
    class SlowWire:
        timeout=60
        def receive(self):
            now[0]+=self.timeout
            raise TimeoutError('still validating')
    wire=SlowWire()
    with pytest.raises(TimeoutError,match='deadline'):
        backend.receive_launch_response(SimpleNamespace(wire=wire))
    assert now[0]==60 and probes==['fresh']*6 and wire.timeout==60


def test_preparation_heartbeat_loss_stops_waiting(case,monkeypatch):
    from types import SimpleNamespace
    backend=case['backend']
    monkeypatch.setattr(backend,'probe',lambda peer:False)
    class SlowWire:
        timeout=60
        def receive(self):raise TimeoutError('pending')
    wire=SlowWire()
    with pytest.raises(ConnectionError,match='Peer lost'):
        backend.receive_launch_response(SimpleNamespace(wire=wire))
    assert wire.timeout==60


@pytest.mark.parametrize('fault',['disconnect_after_ack','heartbeat_disconnect','stale_epoch','lost_done','partial_result'])
def test_live_model_failures_never_replay(case,monkeypatch,fault):
    original=authority_wire.Wire.receive
    def receive(self):
        value=original(self)
        if value.get('kind')=='ACK' and fault=='disconnect_after_ack':raise ConnectionError('injected lost ACK')
        if value.get('kind')=='ACK' and fault=='heartbeat_disconnect':
            peer=case['backend'].peer_channels['fresh'];peer.process.terminate();peer.process.wait(timeout=5)
        if value.get('kind')=='CHECK' and fault=='stale_epoch':
            permit=json.loads(case['permit'].read_text());permit['epoch']=0;case['permit'].write_text(json.dumps(permit))
        if value.get('kind')=='DONE' and fault in ['lost_done','partial_result']:
            if fault=='partial_result':
                p=case['worker']/'tasks'/(value['row']['launch']['key']+'.json')
                row=json.loads(p.read_text());row['status']='UNKNOWN';row.pop('receipt');p.write_text(json.dumps(row))
            raise ConnectionError('injected lost completion')
        return value
    monkeypatch.setattr(authority_wire.Wire,'receive',receive)
    with pytest.raises((RuntimeError,OSError)):run(case)
    monkeypatch.setattr(authority_wire.Wire,'receive',original)
    row=next(iter(case['ledger'].read()[1]['tasks'].values()))
    assert row['status']=='RESERVED' and row['reserved']['calls']==2
    with pytest.raises(RuntimeError):run(case)
    if fault=='lost_done':
        assert recover_controlled_queue(case['queue'],case['ledger'],OWNER,1,NEXT,case['backend'],simulate=True)==2
    else:
        with pytest.raises(RuntimeError):recover_controlled_queue(case['queue'],case['ledger'],OWNER,1,NEXT,case['backend'],simulate=True)
    assert len(list((case['worker']/'tasks').glob('*.json')))==1


def test_cpu_contract_cannot_call_real_executor(case):
    with pytest.raises(RuntimeError,match='CPU contract'):
        execute_block(case['binding'],{},lambda:None)
    with pytest.raises(dispatch_policy.ProductionDispatchDisabled):
        with dispatch_policy._live_model_scope(dict(operation='CPU_BLOCK_CONTRACT'),lambda:None):pass


def test_model_cli_actual_entry_and_draft_refusal(tmp_path):
    case=make_model_case(tmp_path,claim=False)
    owner=tmp_path/'owner.json';owner.write_text(json.dumps(OWNER))
    entry=ROOT/'scripts/dsol_paper1/operations/manage_model_queue.py'
    command=[sys.executable,'-B',str(entry),'run','--binding',str(case['binding_path']),
             '--root',str(case['root']),'--stage','admission','--permit',str(case['permit']),
             '--owner',str(owner),'--epoch','1','--simulate']
    p=subprocess.run(command,text=True,capture_output=True,timeout=40)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['status']=='PAUSED'
    permit=json.loads(case['permit'].read_text());permit['authorization']='TEMPLATE_NOT_AUTHORIZED'
    case['permit'].write_text(json.dumps(permit))
    p=subprocess.run(command,text=True,capture_output=True,timeout=10)
    assert p.returncode!=0 and 'Draft approval' in p.stderr


def test_frozen_namespace_and_host_assignment_remain_bound(case):
    assert all(t['identity_experiment_id']=='cpu-launch-test' for t in case['plan']['tasks'])
    assert all(t['eligible_slots']==case['plan']['slots'] for t in case['plan']['tasks'])
    # Retain original task keys, not a newly generated cross-host campaign ID.
    assert {t['key'] for t in case['plan']['tasks']}=={t['key'] for t in case['binding']['budget']['tasks']}


def test_actual_scientific_adapter_calls_original_block_once_without_old_fleet(case, monkeypatch, tmp_path):
    from benchmarks.LIBERO import continuous_evaluation, continuous_evaluation_fleet
    binding=deepcopy(case['binding']);binding['mode']='libero'
    key=next(iter(binding['tasks']));spec=binding['tasks'][key]
    spec['request']=dict(block=dict(kind='paired',index=0,tasks=12),paired='frozen-paired',nominal='frozen-nominal')
    spec['output']=str(tmp_path/'bounded-result')
    seen=[]
    def science(config,output,*,execute,block_index,guard):
        assert config=='frozen-paired' and execute and block_index==0
        guard();seen.append('paired')
        Path(output).mkdir();(Path(output)/'result.json').write_text('{}')
    monkeypatch.setattr(continuous_evaluation,'run',science)
    monkeypatch.setattr(continuous_evaluation_fleet,'validate_block',
                        lambda req,out:dict(calls=dict(score=34,execution=720),receipt_sha256='a'*64))
    launch=dict(operation='LIBERO_BLOCK',key=key,slot=dict(gpu_id='0'),
                permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1'))
    before=dict(os.environ)
    try:
        counts,stamp=execute_block(binding,launch,lambda:seen.append('guard'))
    finally:
        os.environ.clear();os.environ.update(before)
    assert seen.count('paired')==1 and counts==dict(episodes=12,score=34,paired_execution=720,nominal_execution=0)
    assert stamp=='a'*64 and (tmp_path/'bounded-result/complete.json').exists()
    with pytest.raises(dispatch_policy.ProductionDispatchDisabled):
        dispatch_policy.reject_production('benchmarks/LIBERO/search_runtime.py:PolicyRuntime.infer')


def test_live_scope_never_enables_training_fleet_or_other_thread():
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    launch=dict(operation='LIBERO_BLOCK',permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1'))
    checks=[]
    with dispatch_policy._live_model_scope(launch,lambda:checks.append(1)):
        dispatch_policy.reject_production('benchmarks/LIBERO/search_runtime.py:PolicyRuntime.infer')
        with pytest.raises(dispatch_policy.ProductionDispatchDisabled):
            dispatch_policy.reject_production('benchmarks/LIBERO/continuous_training.py:run')
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(copy_context().run,dispatch_policy.reject_production,
                               'benchmarks/LIBERO/search_runtime.py:PolicyRuntime.infer')
            with pytest.raises(dispatch_policy.ProductionDispatchDisabled):future.result()
    assert len(checks)==2


def test_real_mode_requires_new_custody_and_non_template_permission(case):
    from AlphaBrain.research.dsol.execution.worker_fence import check_permit
    budget,state=case['ledger'].read();permit=json.loads(case['permit'].read_text())
    with pytest.raises(RuntimeError,match='newly bound production custody'):
        check_permit(state,budget,case['plan'],permit,OWNER,1,time.time(),production=True)
    case['backend'].production=True
    with pytest.raises(RuntimeError):run(case)
    assert not list((case['worker']/'tasks').glob('*.json'))


def test_worker_checks_frozen_budget_before_any_execution(monkeypatch):
    from benchmarks.LIBERO import continuous_authority as agent, continuous_evaluation_fleet as fleet
    from AlphaBrain.research.dsol.execution.manifest import task_key
    row=dict(task='task',state_key='state',start=0,method='stay',execution_repeats=[0])
    req=dict(model='m',host='fresh',stage='admission',campaign_sha256='a'*64,
             block=dict(index=0,kind='paired',tasks=1,rows=[row],limits=dict(score=2,execution=5)))
    addresses=[dict(model='m',task='task',initial='state',view='0:stay',stream='execution',repeat=0,noise='paired-seed-0')]
    key=task_key('frozen','a'*64,addresses)
    spec=dict(request=req,host='fresh',stage='admission',identity_experiment_id='frozen',addresses=addresses,
              limits=dict(episodes=1,score=2,paired_execution=5,nominal_execution=0))
    binding=dict(campaign='frozen-campaign',tasks={key:spec})
    monkeypatch.setattr(fleet,'request_for',lambda *args:deepcopy(req))
    assert agent.validate_scientific_spec(binding,key)==spec['limits']
    spec['limits']['paired_execution']=4
    with pytest.raises(ValueError,match='per-block budget'):
        agent.validate_scientific_spec(binding,key)


def test_cpu_mode_cannot_select_an_arbitrary_worker_module(case,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Unregistered module reached process creation')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    target=deepcopy(case['backend'].config['peers']['fresh']);target['agent']='unreviewed_model_launcher'
    with pytest.raises(ValueError,match='reviewed agents'):
        case['backend'].channel(target)


def test_batched_live_grants_reduce_repeated_heartbeats(tmp_path):
    from AlphaBrain.research.dsol.execution.resources import worker_slots
    case=make_model_case(tmp_path,slots=worker_slots('fresh',[0,1],workers_per_gpu=2),checks=32)
    assert run(case)['status']=='COMPLETE'
    metrics=case['backend'].metrics
    assert metrics['continuation_grants']==96
    assert metrics['max_grants_per_round']>=2
    assert metrics['heartbeat_messages']==metrics['heartbeat_rounds']
    assert metrics['ledger_checks']==metrics['poll_rounds']
    assert metrics['heartbeat_messages']<metrics['continuation_grants']


def test_permit_revocation_during_batch_heartbeat_sends_no_grant(case,monkeypatch):
    original=case['backend'].probe_many;sent=[]
    def revoked(peers):
        original(peers)
        permit=json.loads(case['permit'].read_text());permit['epoch']=0
        case['permit'].write_text(json.dumps(permit))
    send=authority_wire.Wire.send
    def record(wire,value):
        if value.get('kind')=='CONTINUE':sent.append(value)
        return send(wire,value)
    monkeypatch.setattr(case['backend'],'probe_many',revoked)
    monkeypatch.setattr(authority_wire.Wire,'send',record)
    with pytest.raises(RuntimeError):run(case)
    assert not sent
    assert next(iter(case['ledger'].read()[1]['tasks'].values()))['status']=='RESERVED'


def test_resource_rebinding_preserves_science_keys_budget_and_original(case):
    from benchmarks.LIBERO.continuous_authority_plan import with_worker_capacity
    binding=deepcopy(case['binding'])
    for spec in binding['tasks'].values():spec['eligible_slots']=[dict(host='fresh',id='0',gpu_id='0')]
    original=deepcopy(binding)
    changed=with_worker_capacity(binding,{'fresh':2},stages=('admission',))
    assert binding==original and changed['budget']==binding['budget']
    assert set(changed['tasks'])==set(binding['tasks'])
    for key,spec in changed['tasks'].items():
        assert len(spec['eligible_slots'])==2
        assert {k:v for k,v in spec.items() if k!='eligible_slots'}=={k:v for k,v in binding['tasks'][key].items() if k!='eligible_slots'}


def test_production_launch_failure_atomically_disables_dispatch_before_checkpoint(case,monkeypatch):
    """Production state machine, injected failure BEFORE any GO/model startup."""
    path=case['ledger'].root/'ledger.json';state=json.loads(path.read_text())
    state.update(simulation_only=False,execution_contract='LIVE_MODEL_V1');path.write_text(json.dumps(state))
    case['backend'].production=True
    permit=json.loads(case['permit'].read_text());permit.update(authorization='EXPLICIT_MODEL_STAGE_V1',binding_sha256=case['plan']['binding_sha256'])
    case['permit'].write_text(json.dumps(permit))
    def refused(*a,**kw):raise RuntimeError('synthetic pre-GO GPU refusal')
    monkeypatch.setattr(case['backend'],'start_authorized',refused)
    with pytest.raises(RuntimeError,match='pre-GO'):
        run_controlled_queue(case['queue'],case['ledger'],OWNER,1,case['permit'],case['backend'],execute=True,timeout_seconds=30)
    final=case['ledger'].read()[1]
    assert final['status']=='HALTED' and final['dispatch_enabled'] is False
    assert len(final['tasks'])==1
    row=next(iter(final['tasks'].values()))
    assert row['status']=='RESERVED' and row['launch_status']=='UNKNOWN' and 'consumed' not in row
    assert not list((case['worker']/'tasks').glob('*.json'))
    checkpoint=json.loads((case['ledger'].root/'checkpoint.json').read_text())
    saved=json.loads(Path(checkpoint['path']).read_text())['ledger']
    assert saved['status']=='HALTED' and saved['dispatch_enabled'] is False


def test_public_endpoint_is_strict_and_hash_bound(tmp_path):
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    known=tmp_path/'known_hosts';known.write_text('public test fixture\n')
    target=dict(alias='existing-alias',public_endpoint=dict(hostname='192.0.2.1',port=2520,
        host_key_alias='[192.0.2.1]:2520',known_hosts_file=str(known),known_hosts_sha256=file_digest(known)))
    args=authority_wire.ssh_command(target)
    assert args[-1]=='existing-alias' and 'StrictHostKeyChecking=yes' in args
    assert 'HostName=192.0.2.1' in args and 'HostKeyAlias=[192.0.2.1]:2520' in args
    for field,value in [('port',True),('port',0),('hostname','bad; command'),('host_key_alias','other')]:
        bad=deepcopy(target);bad['public_endpoint'][field]=value
        with pytest.raises(ValueError):authority_wire.ssh_command(bad)
    bad=deepcopy(target);bad['public_endpoint']['IdentityFile']='forbidden'
    with pytest.raises(ValueError):authority_wire.ssh_command(bad)
    known.write_text('changed public host identity\n')
    with pytest.raises(ValueError,match='changed'):authority_wire.ssh_command(target)
    assert authority_wire.ssh_command(dict(alias='existing-alias'))[-1]=='existing-alias'


def test_actual_unix_listener_rejects_long_temporary_root():
    import tempfile
    from AlphaBrain.research.dsol.execution.local_ipc import probe_unix_listener
    with tempfile.TemporaryDirectory(prefix='dsol-') as root:
        assert probe_unix_listener(root)['status']=='PASS'
        long=Path(root)/('x'*100);long.mkdir()
        with pytest.raises(ValueError,match='short owned TMPDIR'):
            probe_unix_listener(long)
        assert not list(long.iterdir())
