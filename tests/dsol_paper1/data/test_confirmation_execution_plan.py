"""CPU tests for endpoint publication and model-partitioned trial ownership."""
from pathlib import Path

import pytest

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.confirmation_execution import build_queue, publish_host_partition, assigned_initials
from AlphaBrain.research.dsol.data.recovery_confirmation import prepare_execution_plan
from AlphaBrain.research.dsol.execution.manifest import verify_artifacts


def fixture(tmp_path):
    root = tmp_path
    initials = [f'i{i:02d}' for i in range(80)]
    release = root/'release'
    write(release/'authorization.json', dict(counts=dict(initials=80, max_full_rollouts=48640)))
    write(root/'original.json', dict(task_horizons={'Task': 900}))
    write(release/'proposal.json', dict(sources={'original_p0_plan': str(root/'original.json')},
                                       population={'tasks': ['Task']}))
    write(root/'population.json', dict(initial_ids=initials, authorization=str(release/'authorization.json')))
    write(root/'probe.json', dict(panel_plan=str(root/'population.json')))
    choices = root/'choices'; write(choices/'binding.json', dict(model='xr1'))
    records = []
    for initial in initials:
        path = choices/f'{initial}.json'
        write(path, dict(model='xr1', initial=initial, task='Task', binding_sha256=sha(choices/'binding.json'),
              execution_repeats=list(range(8)), evaluation_views=['canonical']+[f'extra_{i:03d}' for i in range(32)],
              new_rollouts=264))
        records.append(dict(initial=initial, path=str(path), sha256=sha(path)))
    write(choices/'status.json', dict(status='COMPLETE', model='xr1', completed_initials=80,
                                     pending_initials=[], records=records))
    return root, release, choices


def test_model_partition_is_explicit_and_bounded(tmp_path):
    root, release, choices = fixture(tmp_path)
    with pytest.raises(ValueError, match='explicit model partition'):
        prepare_execution_plan(release, root/'probe.json', {'xr1': choices}, root/'bad.json')
    plan = prepare_execution_plan(release, root/'probe.json', {'xr1': choices}, root/'plan.json', model_partition='xr1')
    assert plan['max_rollouts'] == 24320 and plan['aggregate_max_rollouts'] == 48640
    assert plan['rollout_count'] == 80*33*8


def test_pilot_and_full_partition_exactly_cover_frozen_trials(tmp_path):
    root, release, choices = fixture(tmp_path)
    plan = prepare_execution_plan(release, root/'probe.json', {'xr1': choices}, root/'plan.json', model_partition='xr1')
    runtime = dict(model='xr1', host='gnu', plan=str(root/'plan.json'), result_root=str(root/'results'),
                   source=str(Path(__file__).resolve().parents[3]), gpu_ids=list(range(8)), admission=str(root/'admission.json'))
    write(root/'runtime.json', runtime)
    pilot = build_queue(root/'runtime.json', root/'pilot-control', pilot=True)
    assert len(pilot['tasks']) == 1 and len(pilot['tasks'][0]['addresses']) == 16
    verify_artifacts(pilot)
    with pytest.raises(FileNotFoundError):
        build_queue(root/'runtime.json', root/'absent-admission')
    write(root/'admission.json', dict(status='PASS', model='xr1', plan_sha256=sha(root/'plan.json'),
          complete_task_attempts_verified=True, pilot_initial='i00', pilot_views=['canonical', 'extra_002'],
          pilot_repeats=list(range(8))))
    full = build_queue(root/'runtime.json', root/'full-control')
    key = lambda a: (a['initial'], a['view'], a['repeat'])
    a = {key(a) for t in pilot['tasks'] for a in t['addresses']}
    b = {key(a) for t in full['tasks'] for a in t['addresses']}
    assert not a & b and len(a | b) == plan['rollout_count']
    assert len(read(root/'results/ownership.json')['records'])*8 == plan['rollout_count']
    assert len(full['slots']) == 8
    verify_artifacts(full)


@pytest.mark.parametrize('workers_per_gpu',[1,2])
def test_two_host_pilots_and_full_queues_are_disjoint_complete(tmp_path,workers_per_gpu):
    root, release, choices = fixture(tmp_path)
    plan = prepare_execution_plan(release, root/'probe.json', {'xr1': choices}, root/'plan.json', model_partition='xr1')
    partition = publish_host_partition(root/'plan.json', root/'partition.json', {'fresh': 8, 'video': 8})
    assert [len(v) for v in partition['hosts'].values()] == [40, 40]
    all_addresses = set()
    for host in ('fresh', 'video'):
        runtime = dict(model='xr1', host=host, plan=str(root/'plan.json'), host_partition=str(root/'partition.json'),
                       result_root=str(root/host/'results'), source=str(Path(__file__).resolve().parents[3]),
                       gpu_ids=list(range(8)), admission=str(root/host/'admission.json'),workers_per_gpu=workers_per_gpu)
        runtime_file = root/host/'runtime.json'; write(runtime_file, runtime)
        pilot = build_queue(runtime_file, root/host/'pilot', pilot=True)
        write(runtime['admission'], dict(status='PASS', model='xr1', plan_sha256=sha(root/'plan.json'),
              host=host, host_partition_sha256=sha(root/'partition.json'), complete_task_attempts_verified=True,
              pilot_initial=partition['hosts'][host][0], pilot_views=['canonical', 'extra_002'], pilot_repeats=list(range(8))))
        full = build_queue(runtime_file, root/host/'full')
        assert len(full['slots'])==8*workers_per_gpu and len(pilot['slots'])==1
        for queue in (pilot, full):
            verify_artifacts(queue)
            for task in queue['tasks']:
                for a in task['addresses']:
                    key = a['initial'], a['view'], a['repeat']
                    assert key not in all_addresses
                    all_addresses.add(key)
    assert len(all_addresses) == plan['rollout_count']


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'wrong_plan'])
def test_host_partition_rejects_unsafe_ownership(tmp_path, damage):
    root, release, choices = fixture(tmp_path)
    plan = prepare_execution_plan(release, root/'probe.json', {'xr1': choices}, root/'plan.json', model_partition='xr1')
    partition = publish_host_partition(root/'plan.json', root/'partition.json', {'fresh': 8, 'video': 8})
    if damage == 'duplicate': partition['hosts']['video'].append(partition['hosts']['fresh'][0])
    elif damage == 'missing': partition['hosts']['video'].pop()
    else: partition['plan_sha256'] = 'invalid'
    write(root/'partition.json', partition)
    with pytest.raises(ValueError, match='partition'):
        assigned_initials(dict(host='video', model='xr1', plan=str(root/'plan.json'),
                              host_partition=str(root/'partition.json')), plan)
