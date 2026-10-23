"""CPU-only dynamic scheduling, frozen identities, failure and recovery tests."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from AlphaBrain.research.dsol.data.artifacts import write
from AlphaBrain.research.dsol.execution.cli import main
from AlphaBrain.research.dsol.execution.manifest import FORMAT, digest, file_digest, task_key, validate
from AlphaBrain.research.dsol.execution.queue import MissingResult, exclusive, load, prepare, recover, run, summary
from tests.dsol_paper1.paths import ROOT


@pytest.fixture
def plan(tmp_path):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{"repeat_budget": 4}')
    sha = file_digest(protocol)
    tasks = []
    for i in range(6):
        address = dict(model='m', task='t', initial=f'i{i}', view='v',
                       stream='execution', repeat=0, noise=f'n{i}')
        tasks.append(dict(key=task_key('experiment', sha, [address]), addresses=[address],
                          output=str(tmp_path / 'results' / str(i)), payload=dict(length=4 if i == 0 else 1)))
    return dict(format=FORMAT, experiment_id='experiment', protocol_sha256=sha,
                artifacts=[dict(path=str(protocol), sha256=sha)],
                slots=[dict(host='local', id='gpu0'), dict(host='remote', id='gpu7')],
                backend=dict(module='not_imported_by_default', factory='factory'), tasks=tasks)


class FakeBackend:
    """Different durations across two hosts; no actual subprocess or GPU."""
    def __init__(self, *, fail=None, admitted=True):
        self.records = {}
        self.results = {}
        self.launched = []
        self.cancelled = []
        self.fail = fail
        self.admitted = admitted

    def admit(self, slot):
        return self.admitted

    def start(self, task, slot, attempt):
        key = task['key']
        self.records[key] = dict(left=task['payload']['length'], alive=True, task=task,
                                 attempt=attempt, slot=slot)
        self.launched.append((key, slot['host'], slot['id'], attempt))
        return dict(key=key, attempt=attempt)

    def locate(self, task, slot, attempt):
        row = self.records.get(task['key'])
        if row is None:
            return None
        assert row['slot'] == slot and row['attempt'] == attempt
        return dict(key=task['key'], attempt=attempt)

    def poll(self, handle):
        row = self.records[handle['key']]
        row['left'] -= 1
        if row['left'] > 0:
            return None
        row['alive'] = False
        if handle['key'] == self.fail:
            return 2
        # Failed scientific tasks are still valid, completed trials.
        self.results[handle['key']] = dict(identity=digest(row['task']['addresses']), success=False)
        return 0

    def alive(self, handle):
        return self.records[handle['key']]['alive']

    def cancel(self, handle):
        self.cancelled.append(handle['key'])
        self.records[handle['key']]['alive'] = False

    def validate(self, task):
        if task['key'] not in self.results:
            raise MissingResult(task['key'])
        return self.results[task['key']]


def execute(root, backend, **kwargs):
    return run(root, backend, poll_seconds=0, sleep=lambda _: None, **kwargs)


def test_dynamic_two_host_drain_preserves_trial_addresses(plan, tmp_path):
    before = deepcopy(plan)
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend()
    state = execute(root, backend)
    assert state['status'] == 'COMPLETE'
    assert len(backend.launched) == 6
    assert len({key for key, _, _, _ in backend.launched}) == 6
    assert backend.launched[2][1] == 'remote'  # First local block is still busy.
    assert {host for _, host, _, _ in backend.launched} == {'local', 'remote'}
    assert all(row['receipt']['success'] is False for row in state['tasks'].values())
    assert load(root)[0] == before == plan
    assert summary(root)['counts']['COMPLETE'] == 6


def test_no_launch_admission_when_all_work_is_already_dispatched(plan, tmp_path):
    class Backend(FakeBackend):
        def admit(self, slot):
            if len(self.launched)==len(plan['tasks']):
                raise RuntimeError('No new launch: admission must not run during completion')
            return True
    root=tmp_path/'completion';prepare(plan,root)
    backend=Backend()
    assert execute(root,backend)['status']=='COMPLETE'
    assert len(backend.launched)==len(plan['tasks']) and not backend.cancelled


def test_physical_gpu_drain_does_not_cancel_and_resumes(plan, tmp_path):
    from AlphaBrain.research.dsol.execution.resources import resource_status, set_drain
    plan['slots'] = [dict(host='local', id='0'), dict(host='local', id='0:1', gpu_id='0')]
    root = tmp_path / 'drain'
    prepare(plan, root)
    backend = FakeBackend()
    ticks = []

    def sleep(_):
        ticks.append(1)
        if len(ticks) == 1:
            set_drain(root, 'local', '0', reason='capacity test')
        if len(ticks) == 5:
            info = resource_status(root)['devices'][0]
            assert info['quiescent'] and len(backend.launched) == 2
            assert not backend.cancelled
            set_drain(root, 'local', '0', reason='capacity test complete', enabled=False)

    run(root, backend, sleep=sleep, poll_seconds=0)
    assert len(backend.launched) == 6 and not backend.cancelled
    assert load(root)[0] == plan


def test_resource_control_rejects_wrong_identity(plan, tmp_path):
    from AlphaBrain.research.dsol.execution.resources import resource_status, set_drain
    root = tmp_path / 'drain'
    prepare(plan, root)
    with pytest.raises(ValueError, match='Unknown'):
        set_drain(root, 'local', 'missing', reason='test')
    with pytest.raises(ValueError, match='reason'):
        set_drain(root, 'local', 'gpu0', reason='')
    set_drain(root, 'local', 'gpu0', reason='test')
    path = root / 'resource-control.json'
    value = json.loads(path.read_text()); value['manifest_sha256'] = 'invalid'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='another manifest'):
        resource_status(root)


def test_read_only_audit_detects_changed_completed_result(plan,tmp_path):
    from AlphaBrain.research.dsol.execution.audit import audit_results
    root=tmp_path/'audit';prepare(plan,root);backend=FakeBackend();execute(root,backend)
    before=(root/'state.json').read_bytes()
    assert audit_results(root,backend,per_status=6)['status']=='PASS'
    key=plan['tasks'][0]['key'];backend.results[key]['success']=True
    assert audit_results(root,backend,per_status=6)['status']=='HOLD'
    assert before==(root/'state.json').read_bytes()


@pytest.mark.parametrize('kind', ['address', 'block', 'output', 'slot', 'protocol', 'repeat'])
def test_duplicate_or_changed_scope_rejected(plan, kind):
    if kind == 'address':
        plan['tasks'][1]['addresses'] = deepcopy(plan['tasks'][0]['addresses'])
    elif kind == 'block':
        plan['tasks'][0]['key'] = 'wrong'
    elif kind == 'output':
        plan['tasks'][1]['output'] = plan['tasks'][0]['output']
    elif kind == 'slot':
        plan['slots'].append(deepcopy(plan['slots'][0]))
    elif kind == 'protocol':
        plan['protocol_sha256'] = 'oops'
    else:
        plan['tasks'][0]['addresses'][0]['repeat'] = True
    with pytest.raises(ValueError):
        validate(plan)


def test_prepare_checks_frozen_artifacts_before_creating_output(plan, tmp_path):
    root = tmp_path / 'run'
    from pathlib import Path
    Path(plan['artifacts'][0]['path']).write_text('changed')
    with pytest.raises(ValueError):
        prepare(plan, root)
    assert not root.exists()


def test_failure_stops_owned_workers_and_requires_explicit_recovery(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend(fail=plan['tasks'][1]['key'])
    with pytest.raises(RuntimeError, match='Worker failed'):
        execute(root, backend)
    assert summary(root)['status'] == 'FAILED'
    assert not any(r['alive'] for r in backend.records.values())
    with pytest.raises(ValueError, match='explicit recovery'):
        execute(root, backend)
    backend.fail = None
    recover(root, backend)
    execute(root, backend)
    assert summary(root)['status'] == 'COMPLETE'


def test_recovery_keeps_complete_results_and_does_not_relaunch_them(plan, tmp_path):
    root = tmp_path / 'run'
    plan['tasks'][0]['payload']['length'] = 1
    plan['tasks'][1]['payload']['length'] = 2
    prepare(plan, root)
    backend = FakeBackend(fail=plan['tasks'][1]['key'])
    with pytest.raises(RuntimeError):
        execute(root, backend)
    saved = deepcopy(backend.results[plan['tasks'][0]['key']])
    backend.fail = None
    recover(root, backend)
    execute(root, backend)
    assert sum(key == plan['tasks'][0]['key'] for key, *_ in backend.launched) == 1
    assert backend.results[plan['tasks'][0]['key']] == saved


def test_live_or_unknown_launch_prevents_recovery(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend()
    _, state = load(root)
    task = plan['tasks'][0]
    backend.start(task, plan['slots'][0], 1)
    state['status'] = 'RUNNING'
    state['tasks'][task['key']].update(status='LAUNCHING', slot=plan['slots'][0], attempt=1)
    write(root / 'state.json', state)
    before = (root / 'state.json').read_bytes()
    with pytest.raises(RuntimeError, match='still alive'):
        recover(root, backend)
    assert (root / 'state.json').read_bytes() == before


def test_invalid_result_is_not_treated_as_missing(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend(fail=plan['tasks'][1]['key'])
    with pytest.raises(RuntimeError):
        execute(root, backend)
    backend.results[plan['tasks'][0]['key']] = {}  # Corrupt complete receipt.
    with pytest.raises(ValueError, match='nonempty'):
        recover(root, backend)


def test_completed_receipt_cannot_be_rewritten_on_resume(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend()
    execute(root, backend)
    _, state = load(root)
    state['status'] = 'FAILED'
    write(root / 'state.json', state)
    backend.results[plan['tasks'][0]['key']]['success'] = True
    with pytest.raises(ValueError, match='result changed'):
        recover(root, backend)


def test_coordinator_lock_prevents_duplicate_launch(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend()
    with exclusive(root):
        with pytest.raises(BlockingIOError):
            execute(root, backend)
    assert not backend.launched


def test_unadmitted_hosts_timeout_without_starting_anything(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend(admitted=False)
    ticks = iter([0, 0, 2])
    with pytest.raises(TimeoutError):
        execute(root, backend, timeout_seconds=1, clock=lambda: next(ticks))
    assert not backend.launched


def test_manifest_tampering_rejected(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    plan['tasks'][0]['payload']['length'] = 9
    write(root / 'manifest.json', plan)
    with pytest.raises(ValueError, match='manifest'):
        load(root)


def test_readonly_cli_does_not_import_backend(plan, tmp_path, capsys):
    root = tmp_path / 'run'
    manifest = tmp_path / 'input.json'
    write(manifest, plan)
    assert main(['prepare', '--root', str(root), '--manifest', str(manifest)]) == 0
    assert main(['status', '--root', str(root)]) == 0
    assert 'PREPARED' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(['run', '--root', str(root)])
    with pytest.raises(SystemExit):
        main(['recover', '--root', str(root)])


def test_start_receipt_gap_is_reconciled_and_owned_process_stopped(plan, tmp_path):
    class StartGap(FakeBackend):
        def start(self, task, slot, attempt):
            super().start(task, slot, attempt)
            raise RuntimeError('Lost return after launch')

    root = tmp_path / 'run'
    prepare(plan, root)
    backend = StartGap()
    with pytest.raises(RuntimeError, match='after launch'):
        execute(root, backend)
    assert len(backend.cancelled) == 1
    assert not any(r['alive'] for r in backend.records.values())
    assert summary(root)['counts']['LAUNCHING'] == 1
    assert recover(root, backend)['status'] == 'PREPARED'


def test_unclean_worker_remains_explicitly_blocking_recovery(plan, tmp_path):
    class CannotStop(FakeBackend):
        def cancel(self, handle):
            raise RuntimeError('Remote host unreachable')

    root = tmp_path / 'run'
    prepare(plan, root)
    backend = CannotStop(fail=plan['tasks'][1]['key'])
    with pytest.raises(RuntimeError):
        execute(root, backend)
    assert load(root)[1]['cleanup_errors']
    with pytest.raises(RuntimeError, match='still alive'):
        recover(root, backend)


@pytest.mark.parametrize('interval,deadline', [(float('nan'), 1), (0, float('inf')), (-1, 2), (0, 0)])
def test_invalid_polling_config_never_launches(plan, tmp_path, interval, deadline):
    root = tmp_path / 'run'
    prepare(plan, root)
    backend = FakeBackend()
    with pytest.raises(ValueError):
        run(root, backend, poll_seconds=interval, timeout_seconds=deadline)
    assert not backend.launched


def test_state_slot_outside_manifest_rejected(plan, tmp_path):
    root = tmp_path / 'run'
    prepare(plan, root)
    _, state = load(root)
    state['tasks'][plan['tasks'][0]['key']].update(status='LAUNCHING', slot=dict(host='other', id='9'))
    write(root / 'state.json', state)
    with pytest.raises(ValueError, match='outside'):
        load(root)


def test_explicit_cli_uses_only_fingerprinted_adapter(plan, tmp_path, monkeypatch, capsys):
    from pathlib import Path
    from types import SimpleNamespace
    from AlphaBrain.research.dsol.execution import cli

    def factory(_plan, _root):
        return FakeBackend()

    source = Path(__file__).resolve()
    plan['artifacts'].append(dict(path=str(source), sha256=file_digest(source)))
    root = tmp_path / 'run'
    prepare(plan, root)
    monkeypatch.setattr(cli.importlib, 'import_module', lambda _: SimpleNamespace(factory=factory))
    assert main(['run', '--root', str(root), '--poll-seconds', '0', '--execute']) == 0
    assert 'COMPLETE' in capsys.readouterr().out


def test_unfingerprinted_factory_is_never_called(plan, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from AlphaBrain.research.dsol.execution import cli
    called = []

    def factory(_plan, _root):
        called.append(True)
        return FakeBackend()

    root = tmp_path / 'run'
    prepare(plan, root)
    monkeypatch.setattr(cli.importlib, 'import_module', lambda _: SimpleNamespace(factory=factory))
    with pytest.raises(ValueError, match='fingerprinted'):
        main(['run', '--root', str(root), '--execute'])
    assert not called


def test_failed_atomic_write_keeps_previous_state_and_removes_temporary(tmp_path, monkeypatch):
    from AlphaBrain.research.dsol.execution import queue
    target = tmp_path / 'state.json'
    target.write_text('original')

    def fail_replace(_source, _target):
        raise OSError('replace failed')

    monkeypatch.setattr(queue.os, 'replace', fail_replace)
    with pytest.raises(OSError):
        queue._durable_write(target, {'status': 'RUNNING'})
    assert target.read_text() == 'original'
    assert list(tmp_path.iterdir()) == [target]


def test_thin_cli_help_works_outside_repo_without_side_effects(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/dsol_paper1/operations/manage_work_queue.py'), '--help'],
                            cwd=tmp_path, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'recover' in result.stdout
    assert not list(tmp_path.iterdir())


def test_planned_drain_keeps_inflight_and_never_starts_pending(tmp_path,plan):
    root=tmp_path/'queue';prepare(plan,root);backend=FakeBackend()
    state=run(root,backend,poll_seconds=0,allow_dispatch=lambda:len(backend.launched)<2)
    assert state['status']=='INTERRUPTED'
    assert len(backend.launched)==2 and not backend.cancelled
    assert sum(x['status']=='COMPLETE' for x in state['tasks'].values())==2
    assert sum(x['status']=='PENDING' for x in state['tasks'].values())==4
    recover(root,backend);run(root,backend,poll_seconds=0)
    assert len(backend.launched)==6
