from pathlib import Path

import pytest

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise
from AlphaBrain.research.dsol.data.rollout_replay import array_identity
from benchmarks.Robocasa365.confirmation_queue import ConfirmationBackend


def test_explicit_command_and_no_admission_without_real_trial(tmp_path):
    admission = tmp_path/'admission.json'; write(admission, dict(status='FAILED'))
    runtime = dict(host='fresh', python='/python', model='official', plan='/plan.json', admission=str(admission))
    backend = ConfirmationBackend(dict(runtime=runtime), tmp_path)
    task = dict(output='/output', payload=dict(initial='i', views=['canonical'], repeats=[0, 1]))
    command = backend.worker_command(task)
    assert command[-1] == '--execute'
    assert command[command.index('--repeats')+1:command.index('--output')] == ['0', '1']
    with pytest.raises(ValueError, match='admission missing'):
        backend.admit(dict(host='fresh', id='0'))


def test_resource_admission_is_host_bound_and_fingerprinted(tmp_path):
    resource = tmp_path/'resource.json'
    write(resource, dict(host='video', status='PASS'))
    plan = dict(runtime=dict(host='fresh'), resource_admission=dict(path=str(resource), sha256=sha(resource)))
    backend = ConfirmationBackend(plan, tmp_path)
    with pytest.raises(ValueError, match='another host'): backend.admit(dict(host='fresh', id='0'))
    write(resource, dict(host='fresh', status='PASS'))
    with pytest.raises(ValueError, match='changed'): backend.admit(dict(host='fresh', id='0'))


def test_failed_task_outcome_is_valid_but_wrong_noise_and_short_trial_rejected(tmp_path):
    source = Path(__file__).resolve().parents[3]
    output = tmp_path/'output'; selection = tmp_path/'choice.json'; plan = tmp_path/'plan.json'
    write(selection, dict(task='Task'))
    write(plan, dict(selections={'official': {'i': dict(path=str(selection), sha256=sha(selection))}},
                     task_horizons={'Task': 26}))
    runtime = dict(model='official', source=str(source), plan=str(plan))
    backend = ConfirmationBackend(dict(runtime=runtime), tmp_path)
    task = dict(output=str(output), payload=dict(initial='i', views=['canonical'], repeats=[0]))
    write(output/'status.json', dict(status='COMPLETE', completed_rollouts=1))
    write(output/'identity.json', dict(model='official', initial='i', views=['canonical'], repeats=[0],
          noise_stream='execution', plan_sha256=sha(plan), selection_sha256=sha(selection),
          source_sha256=sha(source/'benchmarks/Robocasa365/confirmation_execution.py'),
          rollout_source_sha256=sha(source/'benchmarks/Robocasa365/evaluation_matrix.py')))
    file = output/'cells/i/canonical/execution_0.json'
    trial = dict(model='official', initial='i', view='canonical', repeat=0, noise_stream='execution',
                 selection_sha256=sha(selection), horizon=26, steps=26, success=False,
                 terminated=False, truncated=False, complete_task_attempt=True,
                 plans=[dict(step=k*25, noise=array_identity(flow_noise('i', 'execution', 0, k))) for k in range(2)])
    write(file, trial)
    assert backend.validate(task)['status'] == 'PASS'
    bad = read(file); bad['steps'] = 20; write(file, bad)
    with pytest.raises(ValueError, match='Incomplete task'): backend.validate(task)
    bad = dict(trial); bad['plans'] = [dict(step=0, noise=array_identity(flow_noise('i', 'score', 0, 0))), trial['plans'][1]]
    write(file, bad)
    with pytest.raises(ValueError, match='independent address'): backend.validate(task)
