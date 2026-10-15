from copy import deepcopy

import pytest

from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.execution.manifest import task_key, file_digest
from AlphaBrain.research.dsol.execution.queue import prepare, load, run
from AlphaBrain.research.dsol.execution.repartition import repartition


def fixture(root):
    protocol=root/'protocol';protocol.write_text('fixed')
    sha=file_digest(protocol);tasks=[]
    for i in range(4):
        a=dict(model='m',task='t',initial=str(i),view='v',stream='score',repeat=0,noise=str(i))
        tasks.append(dict(key=task_key('e',sha,[a]),addresses=[a],payload={},output=str(root/f'out{i}')))
    plan=dict(format='dsol_work_queue_v1',experiment_id='e',protocol_sha256=sha,tasks=tasks,
              slots=[dict(host='fresh',id='0')],artifacts=[dict(path=str(protocol),sha256=sha)],
              runtime=dict(host='fresh',source='frozen',python='same'),backend=dict(module='adapter',factory='factory'))
    prepare(plan,root/'old')
    state=read(root/'old/state.json');state['status']='RUNNING'
    state['tasks'][tasks[0]['key']].update(status='RUNNING',slot=plan['slots'][0],handle=dict(pid=123),attempt=1)
    write(root/'old/state.json',state)
    dest=[dict(root=str(root/'fresh'),task_keys=[t['key'] for t in tasks[:2]],slots=plan['slots'],
               runtime=plan['runtime'],controller_artifacts=[]),
          dict(root=str(root/'video'),task_keys=[t['key'] for t in tasks[2:]],slots=[dict(host='video',id='0')],
               runtime=dict(plan['runtime'],host='video'),controller_artifacts=[])]
    return plan,state,dest


def test_disjoint_handoff_adopts_running_without_restart(tmp_path):
    plan,state,dest=fixture(tmp_path)
    result=repartition(tmp_path/'old',dest,tmp_path/'evidence')
    assert result['blocks']==dict(fresh=2,video=2)
    a,s=load(tmp_path/'fresh');b,_=load(tmp_path/'video')
    assert s['tasks'][plan['tasks'][0]['key']]==state['tasks'][plan['tasks'][0]['key']]
    assert a['tasks']+b['tasks']==plan['tasks']
    assert read(tmp_path/'old/state.json')==state
    with pytest.raises(ValueError,match='Retired'):run(tmp_path/'old',None)


def test_drain_survives_controller_replacement(tmp_path):
    from AlphaBrain.research.dsol.execution.resources import set_drain, drained_slots
    plan, state, dest = fixture(tmp_path)
    set_drain(tmp_path/'old', 'fresh', '0', reason='Keep existing work; do not refill')
    repartition(tmp_path/'old', dest, tmp_path/'evidence')
    new, rows = load(tmp_path/'fresh')
    assert drained_slots(tmp_path/'fresh', new) == {('fresh', '0')}
    assert rows['tasks'][plan['tasks'][0]['key']]['status'] == 'RUNNING'


def test_new_drain_is_installed_before_new_coordinator_starts(tmp_path):
    from AlphaBrain.research.dsol.execution.resources import drained_slots
    _, _, dest = fixture(tmp_path)
    dest[0]['initial_drained_gpus'] = ['0']
    repartition(tmp_path/'old', dest, tmp_path/'evidence')
    new, _ = load(tmp_path/'fresh')
    assert drained_slots(tmp_path/'fresh', new) == {('fresh', '0')}


def test_unfingerprinted_resource_admission_rejected(tmp_path):
    _, _, dest = fixture(tmp_path)
    dest[1]['resource_admission'] = dict(path='/unknown', sha256='0'*64)
    with pytest.raises(ValueError, match='immutable artifact'):
        repartition(tmp_path/'old', dest, tmp_path/'evidence')
    assert not (tmp_path/'old/retired.json').exists()


@pytest.mark.parametrize('bad',['overlap','missing','live_moved','runtime_changed','started_moved'])
def test_invalid_handoffs_rejected(tmp_path,bad):
    plan,state,dest=fixture(tmp_path)
    if bad=='overlap':dest[1]['task_keys'].append(dest[0]['task_keys'][0])
    elif bad=='missing':dest[1]['task_keys'].pop()
    elif bad=='live_moved':dest[0]['task_keys'][0],dest[1]['task_keys'][0]=dest[1]['task_keys'][0],dest[0]['task_keys'][0]
    elif bad=='runtime_changed':dest[1]['runtime']['source']='other'
    else:
        state['tasks'][dest[1]['task_keys'][0]]['attempt']=1;write(tmp_path/'old/state.json',state)
    with pytest.raises(ValueError):repartition(tmp_path/'old',dest,tmp_path/'evidence')
    assert not (tmp_path/'old/retired.json').exists()


def test_controller_only_handoff_does_not_signal_live_worker(tmp_path):
    import subprocess
    import sys
    from benchmarks.Robocasa365.queue_handoff import handoff
    from benchmarks.Robocasa365.probe_queue import process_identity
    plan,state,dest=fixture(tmp_path)
    worker=subprocess.Popen(['sleep','30'],start_new_session=True)
    controller=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',str(tmp_path/'old')],start_new_session=True)
    try:
        state['tasks'][plan['tasks'][0]['key']]['handle']=process_identity(worker.pid)
        write(tmp_path/'old/state.json',state);(tmp_path/'source').mkdir()
        write(tmp_path/'config.json',dict(old_control=str(tmp_path/'old'),coordinator_pid=controller.pid,
              controller_source=str(tmp_path/'source'),evidence_root=str(tmp_path/'evidence'),receipt=str(tmp_path/'receipt.json'),
              destinations={d['runtime']['host']:dict(control=d['root'],slots=d['slots'],capacity=1,workers_per_gpu=1) for d in dest}))
        result=handoff(tmp_path/'config.json')
        controller.wait(timeout=3)
        assert controller.returncode==-9 and worker.poll() is None
        assert result['no_worker_signals_sent']
        assert load(tmp_path/'fresh')[1]['tasks'][plan['tasks'][0]['key']]['handle']['pid']==worker.pid
    finally:
        worker.terminate();worker.wait(timeout=3)
        if controller.poll() is None:controller.kill();controller.wait(timeout=3)


@pytest.mark.parametrize('mismatch', ['slot', 'environment'])
def test_slot_mismatch_keeps_original_coordinator_alive(tmp_path, mismatch):
    import subprocess
    import sys
    from benchmarks.Robocasa365.queue_handoff import handoff
    plan, state, dest = fixture(tmp_path)
    controller=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',str(tmp_path/'old')],start_new_session=True)
    try:
        (tmp_path/'source').mkdir()
        if mismatch == 'slot':
            dest[0]['slots'] = [dict(host='fresh', id='0', gpu_id='0')]
        destinations={d['runtime']['host']:dict(control=d['root'],slots=d['slots'],capacity=1,workers_per_gpu=1) for d in dest}
        if mismatch == 'environment':
            plan['runtime']['environment']={}
            from AlphaBrain.research.dsol.execution.manifest import digest
            state['manifest_sha256']=digest(plan)
            write(tmp_path/'old/manifest.json',plan);write(tmp_path/'old/state.json',state)
            destinations['video']['environment']={'LD_LIBRARY_PATH':'different'}
        write(tmp_path/'config.json', dict(old_control=str(tmp_path/'old'),coordinator_pid=controller.pid,
            controller_source=str(tmp_path/'source'),evidence_root=str(tmp_path/'evidence'),receipt=str(tmp_path/'receipt.json'),
            destinations=destinations))
        with pytest.raises(ValueError, match='host or slot|Runtime environment'): handoff(tmp_path/'config.json')
        assert controller.poll() is None
        assert not read(tmp_path/'receipt.json')['coordinator_stopped']
    finally:
        controller.terminate();controller.wait(timeout=3)
