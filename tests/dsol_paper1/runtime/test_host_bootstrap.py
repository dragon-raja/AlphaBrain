import platform
from pathlib import Path

import pytest

from AlphaBrain.research.dsol.data.artifacts import write, sha, read
from benchmarks.Robocasa365.host_bootstrap import TransferredHostBackend


def setup_backend(tmp_path):
    worker = tmp_path/'source/benchmarks/Robocasa365/confirmation_execution.py'
    worker.parent.mkdir(parents=True); worker.write_text('frozen')
    plan_file = tmp_path/'science.json'; write(plan_file, {})
    admission = tmp_path/'old-admission.json'
    receipt = dict(status='PASS', model='xr1', plan_sha256=sha(plan_file),
        source_sha256=sha(worker), complete_task_attempts_verified=True)
    write(admission, receipt)
    replays = []
    for n in range(2):
        p = tmp_path/f'replay{n}.json'
        write(p, dict(status='PASS', model='xr1', host=platform.node(),
                      rows=[dict(actions_exact=True, velocity_exact=True)]))
        replays.append(dict(path=str(p), sha256=sha(p)))
    config = tmp_path/'bootstrap.json'
    write(config, dict(host='fresh', model='xr1', plan_sha256=sha(plan_file),
        worker_sha256=sha(worker), replays=replays))
    slots = [dict(host='fresh', id=x, gpu_id='4') for x in ('4', '4:1')]
    task = dict(key='first')
    plan = dict(runtime=dict(host='fresh', model='xr1', plan=str(plan_file),
        source=str(tmp_path/'source'), admission=str(admission), workers_per_gpu=2),
        tasks=[task, dict(key='second')], slots=slots,
        host_bootstrap=dict(path=str(config), sha256=sha(config)))
    state = dict(tasks={k:dict(status='PENDING',attempt=0) for k in ('first','second')})
    write(tmp_path/'state.json', state)
    backend = TransferredHostBackend(plan, tmp_path)
    backend.gpu_available = lambda slot: True
    backend.validate = lambda task: dict(status='PASS', trials=['verified'])
    return backend, state, slots


def test_first_block_exclusive_then_expand_without_rerun(tmp_path):
    backend, state, slots = setup_backend(tmp_path)
    assert backend.admit(slots[0])
    assert not backend.admit(slots[1])
    state['tasks']['first'].update(status='RUNNING', attempt=1)
    write(tmp_path/'state.json', state)
    assert not backend.admit(slots[1])
    state['tasks']['first'].update(status='COMPLETE', receipt=backend.validate(None))
    write(tmp_path/'state.json', state)
    assert backend.admit(slots[1])
    assert read(tmp_path/'host-admission.json')['pilot_already_in_original_budget']


def test_failed_replay_cannot_admit(tmp_path):
    backend, _, slots = setup_backend(tmp_path)
    write(tmp_path/'replay0.json', dict(status='HOLD'))
    with pytest.raises(ValueError, match='replay changed'): backend.admit(slots[0])


def test_round_robin_preserves_tasks_and_order_within_task():
    from benchmarks.Robocasa365.queue_handoff import interleave_tasks
    rows = [dict(key=f'{task}{i}', addresses=[dict(task=task)]) for task in ('a','b') for i in range(3)]
    assert [r['key'] for r in interleave_tasks(rows)] == ['a0','b0','a1','b1','a2','b2']


def test_historical_inline_coordinator_requires_exact_literal_root():
    from benchmarks.Robocasa365.queue_handoff import bound_inline_queue
    code="from AlphaBrain.research.dsol.execution.cli import main; main(['run','--root','/queue','--execute'])"
    assert bound_inline_queue(['python','-c',code],Path('/queue'))
    assert not bound_inline_queue(['python','-c',code],Path('/another'))
    assert not bound_inline_queue(['python','-c',"print('/queue')"],Path('/queue'))
    assert not bound_inline_queue(['python','-c',code+"; print('side effect')"],Path('/queue'))
