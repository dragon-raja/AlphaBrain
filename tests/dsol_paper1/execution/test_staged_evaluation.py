"""Actual orchestration entry with CPU backends and durable unknown attempts."""
import json
from pathlib import Path
import pytest
from AlphaBrain.research.dsol.execution import staged_evaluation
from AlphaBrain.research.dsol.execution.manifest import file_digest


@pytest.fixture
def case(tmp_path):
    phases = []
    limits = dict(episodes=2, score=10, paired_execution=10, nominal_execution=0)
    for i in range(2):
        path = tmp_path / f'campaign-{i}.json'; path.write_text('{}')
        phases.append(dict(phase=f'wave-{i}', campaign=str(path), sha256=file_digest(path),
                           limits=limits, searches=1))
    allocation = dict(phases=phases, limits={k: 2*v for k,v in limits.items()})
    events = []; states = []
    def stage(phase, path, cpu=False): events.append(('start', phase, cpu))
    def completion(phase, cpu):
        return dict(status='COMPLETE', actual_model_calls=0 if cpu else 12,
                    actual_manipulation_episodes=0 if cpu else 2)
    kwargs = dict(contract_phase='contract', stage=stage, completion=completion,
                  gate=lambda p: events.append(('gate', p)),
                  report=lambda: events.append(('report',)),
                  status=lambda s, **v: states.append(dict(status=s, **v)),
                  verify_sources=lambda: None)
    return allocation, tmp_path / 'control', kwargs, events, states


def test_actual_entry_orders_contract_gate_and_all_work(case):
    a, root, kw, events, states = case
    staged_evaluation.run(a, root, **kw)
    assert events == [('start','contract',True),('start','wave-0',False),('gate','wave-0'),
                      ('start','wave-1',False),('report',)]
    assert states[-1] == dict(status='COMPLETE', completed_searches=2, completed_episodes=4, calls_settled=24)
    with pytest.raises(FileExistsError): staged_evaluation.run(a, root, **kw)


@pytest.mark.parametrize('failure', ['contract', 'unknown_start', 'first_gate', 'partial_wave', 'changed_input'])
def test_no_automatic_replay_or_report_on_failed_boundary(case, failure):
    a, root, kw, events, states = case
    old = kw['stage']
    def stage(name, path, cpu=False):
        old(name,path,cpu)
        if (failure == 'contract' and cpu or failure == 'unknown_start' and name == 'wave-0'
                or failure == 'partial_wave' and name == 'wave-1'):
            raise RuntimeError('unknown outcome; do not replay')
    kw['stage'] = stage
    if failure == 'first_gate':
        def gate(_): raise ValueError('trajectory mismatch')
        kw['gate'] = gate
    if failure == 'changed_input': Path(a['phases'][0]['campaign']).write_text('changed')
    with pytest.raises((RuntimeError, ValueError)): staged_evaluation.run(a, root, **kw)
    assert states[-1]['status'] == 'HALTED_NO_RETRY'
    assert ('report',) not in events
    if failure != 'partial_wave': assert ('start','wave-1',False) not in events
    if failure == 'unknown_start':
        assert (root/'wave-0.attempt.json').is_file()
        assert not (root/'wave-0.accepted.json').exists()
    with pytest.raises(FileExistsError): staged_evaluation.run(a, root, **kw)
