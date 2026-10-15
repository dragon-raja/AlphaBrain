"""Missing-only endpoint recovery retains failures and never starts processes."""
from pathlib import Path

import pytest

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise
from AlphaBrain.research.dsol.data.rollout_replay import array_identity
from AlphaBrain.research.dsol.execution.manifest import task_key, verify_artifacts
from AlphaBrain.research.dsol.execution.queue import prepare, exclusive
from benchmarks.Robocasa365.confirmation_queue import ConfirmationBackend
from benchmarks.Robocasa365.confirmation_recovery import prepare_missing
from benchmarks.Robocasa365.probe_queue import ProbeBackend


def fixture(root):
    source = Path(__file__).resolve().parents[3]
    choice = root/'choice.json'; plan = root/'plan.json'; output = root/'old-output'
    write(choice, dict(task='Task'))
    write(plan, dict(selections={'official': {'i': dict(path=str(choice), sha256=sha(choice))}}, task_horizons={'Task': 26}))
    runtime = dict(model='official', host='host', source=str(source), plan=str(plan))
    addresses = [dict(model='official', task='Task', initial='i', view='canonical', stream='execution',
                      repeat=r, noise=f'execution/i/{r}/all-native-replans') for r in (0,1)]
    task = dict(key=task_key('test', sha(plan), addresses), addresses=addresses, output=str(output),
                payload=dict(initial='i', views=['canonical'], repeats=[0,1]))
    manifest = dict(format='dsol_work_queue_v1', experiment_id='test', protocol_sha256=sha(plan),
        slots=[dict(host='host', id='0')], tasks=[task], artifacts=[dict(path=str(plan), sha256=sha(plan))],
        runtime=runtime, backend=dict(module='benchmarks.Robocasa365.confirmation_queue', factory='factory'))
    old = root/'old'; prepare(manifest, old)
    state = read(old/'state.json'); state['status']='FAILED'
    state['tasks'][task['key']].update(status='RUNNING', attempt=1, slot=manifest['slots'][0], handle={'pid':123})
    write(old/'state.json', state)
    write(output/'identity.json', dict(model='official', initial='i', views=['canonical'], repeats=[0,1],
          noise_stream='execution', plan_sha256=sha(plan), selection_sha256=sha(choice),
          source_sha256=sha(source/'benchmarks/Robocasa365/confirmation_execution.py'),
          rollout_source_sha256=sha(source/'benchmarks/Robocasa365/evaluation_matrix.py')))
    write(output/'status.json', dict(status='FAILED', completed_rollouts=1))
    trial = dict(model='official', initial='i', view='canonical', repeat=0, noise_stream='execution',
        selection_sha256=sha(choice), horizon=26, steps=26, success=False, terminated=False, truncated=False,
        complete_task_attempt=True, plans=[dict(step=k*25, noise=array_identity(flow_noise('i','execution',0,k))) for k in range(2)])
    file = output/'cells/i/canonical/execution_0.json'; write(file, trial)
    return old, file


def test_partial_failure_retained_and_only_missing_repeat_assigned(tmp_path, monkeypatch):
    old, trial = fixture(tmp_path); original = trial.read_bytes()
    monkeypatch.setattr(ConfirmationBackend, 'alive', lambda *_: False)
    result = prepare_missing(old, tmp_path/'new', tmp_path/'results')
    assert result['missing_trials'] == 1 and len(result['retained']) == 1
    manifest = read(tmp_path/'new/manifest.json'); verify_artifacts(manifest)
    assert manifest['tasks'][0]['payload']['repeats'] == [1]
    assert trial.read_bytes() == original and not read(trial)['success']
    assert manifest['runtime'] == read(old/'manifest.json')['runtime']


def test_alive_worker_and_active_coordinator_both_block_recovery(tmp_path, monkeypatch):
    old, _ = fixture(tmp_path)
    monkeypatch.setattr(ConfirmationBackend, 'alive', lambda *_: True)
    with pytest.raises(RuntimeError, match='still alive'):
        prepare_missing(old, tmp_path/'new', tmp_path/'results')
    with exclusive(old), pytest.raises(BlockingIOError):
        prepare_missing(old, tmp_path/'new', tmp_path/'results')
    assert not (tmp_path/'results').exists()


def test_corrupt_committed_trial_not_silently_reissued(tmp_path, monkeypatch):
    old, trial = fixture(tmp_path)
    monkeypatch.setattr(ConfirmationBackend, 'alive', lambda *_: False)
    bad=read(trial);bad['plans'][0]['noise']={};write(trial,bad)
    with pytest.raises(ValueError, match='independent address'):
        prepare_missing(old, tmp_path/'new', tmp_path/'results')
    assert not (tmp_path/'results').exists()


def test_reboot_proves_old_worker_dead_without_touching_recycled_pid(tmp_path,monkeypatch):
    old,_=fixture(tmp_path);state=read(old/'state.json')
    next(iter(state['tasks'].values()))['handle']['boot']='previous-boot'
    write(old/'state.json',state)
    monkeypatch.setattr(ConfirmationBackend,'alive',lambda *_:pytest.fail('Do not inspect or signal recycled PID'))
    assert prepare_missing(old,tmp_path/'new',tmp_path/'results')['missing_trials']==1


def test_short_probe_retains_whole_committed_cells_only(tmp_path,monkeypatch):
    import numpy as np
    from benchmarks.Robocasa365.short_view_probe import prefix_means
    source=Path(__file__).resolve().parents[3];config=tmp_path/'config.json';write(config,{'test':True})
    runtime=dict(model='xr1',host='host',source=str(source),config=str(config));output=tmp_path/'old-output'
    addresses=[dict(model='xr1',task='Task',initial='i',view=v,stream='score',repeat=r,noise=f'score/i/{r}')
               for v in ['canonical','extra_002'] for r in range(8)]
    task=dict(key=task_key('probe',sha(config),addresses),addresses=addresses,output=str(output),
              payload=dict(initial='i',views=['canonical','extra_002']))
    manifest=dict(format='dsol_work_queue_v1',experiment_id='probe',protocol_sha256=sha(config),
        slots=[dict(host='host',id='0')],tasks=[task],artifacts=[dict(path=str(config),sha256=sha(config))],
        runtime=runtime,backend=dict(module='benchmarks.Robocasa365.probe_queue',factory='factory'))
    old=tmp_path/'old';prepare(manifest,old)
    write(output/'status.json',dict(status='FAILED',completed_cells=1,completed_probes=11))
    write(output/'identity.json',dict(model='xr1',initials=['i'],views=['canonical','extra_002'],
          config_sha256=sha(config),source_sha256=sha(source/'benchmarks/Robocasa365/short_view_probe.py')))
    cell=output/'cells/i/canonical';cell.mkdir(parents=True);scores=np.ones((8,5))
    np.savez(cell/'probes.npz',scores=scores,prefix_means=prefix_means(scores))
    write(cell/'complete.json',dict(arrays_sha256=sha(cell/'probes.npz'),window1_exact=True))
    monkeypatch.setattr(ProbeBackend,'alive',lambda *_:False)
    result=prepare_missing(old,tmp_path/'new',tmp_path/'results')
    assert len(result['retained'])==8 and result['missing_trials']==8
    tasks=read(tmp_path/'new/manifest.json')['tasks']
    assert len(tasks)==1 and tasks[0]['payload']==dict(initial='i',views=['extra_002'])
    assert len(tasks[0]['addresses'])==8
