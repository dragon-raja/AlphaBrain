import json
import hashlib
import subprocess
import sys
import pytest
from AlphaBrain.research.dsol.execution.learner_publication import accept, execute_update, published


def receipt():return dict(run='A-seed-0',version=1,episodes=32,operation_model_updates=0,
    optimizer_device='cpu',status='PAUSED_AT_UPDATE_BOUNDARY',policy='/synthetic/policy-0001.pt',policy_sha256='bound')


def test_stale_shared_view_waits_for_unique_hash_without_replaying_update(tmp_path):
    sequence=iter([None,'old','bound']);calls=[]
    def sha(p):calls.append(p);return next(sequence)
    result=accept(json.dumps(receipt()),tmp_path,run='A-seed-0',version=1,hasher=sha,sleep=lambda _:None)
    assert result==receipt() and len(calls)==3
    assert json.loads((tmp_path/'A-seed-0/0001.json').read_text())==result


def test_unknown_visibility_never_creates_permission(tmp_path):
    with pytest.raises(TimeoutError):accept(json.dumps(receipt()),tmp_path,run='A-seed-0',version=1,
        timeout=0,hasher=lambda p:'old')
    assert not (tmp_path/'A-seed-0/0001.json').exists()


def test_wrong_version_and_conflicting_receipt_refused(tmp_path):
    with pytest.raises(ValueError):accept(json.dumps(receipt()),tmp_path,run='A-seed-0',version=2)
    accept(json.dumps(receipt()),tmp_path,run='A-seed-0',version=1,hasher=lambda p:'bound')
    value=dict(receipt(),policy_sha256='different')
    with pytest.raises(ValueError,match='Conflicting'):accept(json.dumps(value),tmp_path,run='A-seed-0',version=1,hasher=lambda p:'different')


def update_command(tmp_path, *, version=2, exit_code=0):
    policy=tmp_path/f'policy-{version:04d}.pt';policy.write_bytes(b'cpu-test-policy')
    value=dict(receipt(),version=version,episodes=version*32,policy=str(policy),
               policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
    counter=tmp_path/'optimizer-attempts'
    code=('from pathlib import Path;import sys;'
          f'p=Path({str(counter)!r});p.write_text(p.read_text()+"x" if p.exists() else "x");'
          f'print({json.dumps(value)!r});sys.exit({exit_code})')
    return [sys.executable,'-c',code],value,counter


def test_actual_cpu_command_accepts_next_version_and_cannot_repeat(tmp_path):
    command,value,counter=update_command(tmp_path)
    root=tmp_path/'receipts';log=tmp_path/'update.log'
    result=execute_update(command,log,root,run='A-seed-0',version=2)
    assert result==value==published(root,'A-seed-0',2)
    assert json.loads(log.with_suffix('.publication.json').read_text())==value
    with pytest.raises(ValueError,match='already accepted'):
        execute_update(command,tmp_path/'another.log',root,run='A-seed-0',version=2)
    assert counter.read_text()=='x'


@pytest.mark.parametrize('fault',['nonzero_exit','stale_stdout'])
def test_failed_cpu_handoff_preserves_attempt_and_does_not_replay(tmp_path,fault):
    command,_,counter=update_command(tmp_path,version=1 if fault=='stale_stdout' else 2,
                                     exit_code=7 if fault=='nonzero_exit' else 0)
    root=tmp_path/'receipts';log=tmp_path/'update.log'
    with pytest.raises((subprocess.CalledProcessError,ValueError)):
        execute_update(command,log,root,run='A-seed-0',version=2)
    assert not (root/'A-seed-0/0002.json').exists()
    with pytest.raises(FileExistsError):execute_update(command,log,root,run='A-seed-0',version=2)
    with pytest.raises(FileExistsError):
        execute_update(command,tmp_path/'another.log',root,run='A-seed-0',version=2)
    assert counter.read_text()=='x' and log.read_text()


@pytest.mark.parametrize('version',[{},True,0,257])
def test_invalid_expected_version_never_invokes_optimizer(tmp_path,version):
    command,_,counter=update_command(tmp_path)
    with pytest.raises(ValueError,match='expected output version'):
        execute_update(command,tmp_path/'update.log',tmp_path/'receipts',run='A-seed-0',version=version)
    assert not counter.exists()


def test_explicit_short_endpoint_is_not_mistaken_for_stale_completion(tmp_path):
    value=dict(receipt(),version=8,episodes=256,status='COMPLETE',policy='/synthetic/policy-0008.pt')
    with pytest.raises(ValueError):accept(json.dumps(value),tmp_path,run=value['run'],version=8,hasher=lambda p:'bound')
    result=accept(json.dumps(value),tmp_path,run=value['run'],version=8,final_version=8,hasher=lambda p:'bound')
    assert result==value
    with pytest.raises(ValueError):
        execute_update(['must-not-run'],tmp_path/'unused.log',tmp_path/'unused',run=value['run'],version=9,final_version=8)
