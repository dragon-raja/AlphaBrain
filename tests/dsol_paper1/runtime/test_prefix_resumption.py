import json
import pytest
from types import SimpleNamespace
from copy import deepcopy
from benchmarks.LIBERO.prefix_resumption import verify_zero_attempt, accepted_version
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.authority_wire import Channel


def test_resumption_requires_hash_bound_zero_dispatch_evidence(tmp_path):
    path = tmp_path/'audit.json'
    value = dict(status='VERIFIED_PRE_DISPATCH_ZERO_MODEL_CALLS', model_calls=0,
        episodes=0, optimizer_updates=0, permit_created=False, local_center_created=False,
        processes=[], research_apps_on_target_cards=[])
    path.write_text(json.dumps(value)); stamp = file_digest(path)
    assert verify_zero_attempt(path, expected_sha256=stamp)['model_calls'] == 0
    for key, bad in [('model_calls', 1), ('episodes', 1), ('optimizer_updates', 1),
                     ('permit_created', True), ('processes', [123]), ('research_apps_on_target_cards', [123])]:
        path.write_text(json.dumps({**value, key: bad}))
        with pytest.raises(ValueError): verify_zero_attempt(path, expected_sha256=file_digest(path))
    with pytest.raises(ValueError): verify_zero_attempt(path, expected_sha256=stamp)


def test_accepted_boundary_rejects_changed_policy_and_false_episode_count(tmp_path):
    folder = tmp_path/'publications/model'; folder.mkdir(parents=True)
    policy = tmp_path/'policy.pt'; policy.write_bytes(b'original-checkpoint')
    path = folder/'0042.json'
    row = dict(run='model', version=42, episodes=1344, policy=str(policy), policy_sha256=file_digest(policy))
    path.write_text(json.dumps(row)); assert accepted_version(tmp_path, 'model') == 42
    path.write_text(json.dumps({**row, 'episodes': 1345}))
    with pytest.raises(ValueError): accepted_version(tmp_path, 'model')
    path.write_text(json.dumps(row)); policy.write_bytes(b'changed-checkpoint')
    with pytest.raises(ValueError): accepted_version(tmp_path, 'model')


def test_initial_cpu_handshake_allowance_is_restored_before_live_grants():
    class FakeWire:
        timeout = 10
        def receive(self):
            assert self.timeout == 600
            return dict(kind='HELLO', source='frozen')
    channel = Channel.__new__(Channel)
    channel.hello_timeout = 600; channel.wire = FakeWire(); channel.expected_source = 'frozen'
    assert channel.__enter__() is channel
    assert channel.wire.timeout == 10


@pytest.mark.parametrize('timeout', [0, 9, 601, True, float('inf')])
def test_invalid_handshake_budget_fails_before_opening_a_process(timeout):
    with pytest.raises(ValueError): Channel({}, hello_timeout=timeout)


@pytest.mark.parametrize('fault', [None, 'live_owner', 'excluded', 'host_bypass', 'scope', 'duplicate', 'semantics', 'actor', 'proof'])
def test_evaluation_handoff_rejects_stale_owner_scope_or_binding(tmp_path, monkeypatch, fault):
    import benchmarks.LIBERO.prefix_resumption as module
    def save(name, value):
        p = tmp_path/name; p.write_text(json.dumps(value)); return str(p)
    schedule = save('schedule.json', [dict(task='fixed')])
    owner = save('owner.json', dict(pid=123, birth_ticks=456))
    proof = save('proof.json', dict(source_identity='frozen'))
    jobs = [dict(name='model', schedule=schedule, stay=False)]
    audit = save('audit.json', dict(status='SETTLED_OWNERS_RETIRED_NO_REPLAY', remaining_jobs=jobs))
    c = dict(handoff_input_hashes={p:file_digest(p) for p in [schedule, owner, proof, audit]},
        handoff_audit=audit, retired_owners=[owner], capacity=dict(fresh=[0]), excluded_hosts=['video'],
        jobs=deepcopy(jobs), restored_proofs=dict(model={'fresh/suite':proof}))
    run = module.EvaluationHandoff.__new__(module.EvaluationHandoff)
    run.config = c; run.old = tmp_path
    run.evaluate = SimpleNamespace(cfg=dict(hosts=dict(model=['fresh'])), manifest=dict(source_identity='frozen'))
    run.protocol = dict(cells=dict(model=dict(evaluation_schedules={schedule:dict(purpose='final')})))
    monkeypatch.setattr(module, 'process_alive', lambda r: fault == 'live_owner')
    monkeypatch.setattr(module, 'accepted_version', lambda *args: 32 if fault == 'actor' else 64)
    if fault == 'excluded': c['capacity']['video'] = [0]
    if fault == 'host_bypass': run.evaluate.cfg['hosts']['model'].append('video')
    if fault == 'scope': c['jobs'] = []
    if fault == 'duplicate':
        c['jobs'] *= 2
        open(audit,'w').write(json.dumps(dict(status='SETTLED_OWNERS_RETIRED_NO_REPLAY', remaining_jobs=c['jobs'])))
        c['handoff_input_hashes'][audit] = file_digest(audit)
    if fault == 'semantics': run.protocol['cells']['model']['evaluation_schedules'][schedule]['purpose'] = 'baseline'
    if fault == 'proof': run.evaluate.manifest['source_identity'] = 'other'
    if fault is None: run.preflight()
    else:
        with pytest.raises((ValueError, RuntimeError)): run.preflight()


def test_handoff_actual_phase_entry_preserves_schedule_and_uses_disjoint_pool(tmp_path, monkeypatch):
    import benchmarks.LIBERO.prefix_resumption as module
    from AlphaBrain.research.dsol.execution.fleet_slots import FleetSlots
    from threading import RLock
    events = []
    class Budget:
        def reserve(self, key, path): events.append(('reserve', key))
        def settle(self, key, result): events.append(('settle', key))
        def status(self): return {}
    def build(name, kind, schedule, host, **kwargs):
        assert schedule == str(tmp_path/'schedule.json') and kwargs == dict(version=64, stay=True)
        assert host in ('fresh', 'gnu')
        return tmp_path/'campaign.json', 'baseline-v0064'
    def execution(*args): return {'actual_binding': True}
    budget = Budget()
    ctx = SimpleNamespace(protocol=dict(cells=dict(model=dict(family='rldx'))),
        cfg=dict(hosts=dict(rldx=['fresh', 'gnu'])),
        proofs=dict(rldx={'fresh/suite': str(tmp_path/'proof.json'), 'gnu/suite': str(tmp_path/'proof.json')}),
        manifest=dict(source_identity='frozen', release='original-source'), root=tmp_path/'context',
        build=build, execution=execution, budget=budget)
    (tmp_path/'schedule.json').write_text(json.dumps([dict(suite='suite')]))
    (tmp_path/'proof.json').write_text(json.dumps(dict(source_identity='frozen')))
    run = module.EvaluationHandoff.__new__(module.EvaluationHandoff)
    run.pool = FleetSlots(dict(fresh=[0, 1], gnu=[0, 1]))
    run.config = dict(reservation_namespace='new-owner')
    run.root=tmp_path; run.lock=RLock(); run.state={}; run.errors={}
    run.train = run.evaluate = SimpleNamespace(budget=budget)
    def stage(source, config, label, campaign, log, **kwargs):
        assert source == 'original-source' and kwargs == dict(kind='target_recovery')
        assert events[0][0] == 'reserve'
        assert json.loads(config.read_text()) == {'actual_binding': True}
        events.append(('start', label)); return dict(status='COMPLETE', actual_model_calls=1)
    monkeypatch.setattr(module, 'frozen_stage', stage)
    run.phase(ctx, 'model', 'evaluation', str(tmp_path/'schedule.json'), version=64, stay=True)
    assert [v[0] for v in events] == ['reserve', 'start', 'settle']
    assert not run.pool.snapshot()['held']
