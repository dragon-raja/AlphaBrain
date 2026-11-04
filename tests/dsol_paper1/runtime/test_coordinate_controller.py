import json
import pytest
from benchmarks.LIBERO import coordinate_probe
from AlphaBrain.research.dsol.execution.manifest import file_digest


def setup_config(tmp_path):
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    protocol=save('protocol.json',dict(call_caps=dict(score=10,paired_execution=0,nominal_execution=0)))
    campaigns=[save(f'phase-{i}.json',dict(limits=dict(score=6,paired_execution=0,nominal_execution=0))) for i in range(2)]
    prior=save('prior.json',dict(status='KNOWN_TERMINATED',actual_model_calls=2,
        live_owned_processes=[],unknown_calls=0,actual_episodes=0))
    output=tmp_path/'result';output.mkdir()
    path=save('config.json',dict(output=str(output),protocol=protocol,total_call_cap=10,campaigns=campaigns,
        prior_attempt_accounting=prior,input_hashes={prior:file_digest(prior)}))
    return path,output


def test_known_prior_consumption_charged_and_completed_reservations_settled(tmp_path,monkeypatch):
    config,output=setup_config(tmp_path);stages=[]
    def stage(c,phase,campaign,kind,cpu):
        stages.append((phase,cpu));return dict(actual_model_calls=1 if phase=='probe-00' else 2)
    monkeypatch.setattr(coordinate_probe,'run_stage',stage)
    from reports.paper1 import coordinate_recovery
    monkeypatch.setattr(coordinate_recovery,'build',lambda *a:None)
    coordinate_probe.run(config,execute=True)
    result=json.loads((output/'current.json').read_text())
    assert result['actual_model_calls']==5 and result['prior_engineering_calls']==2
    assert result['scientific_model_calls']==3 and not result['dispatch_enabled']
    assert json.loads((output/'reservations/probe-01.json').read_text())['cumulative']==9
    assert stages==[('probe-gate',True),('probe-00',False),('probe-01',False)]


def test_unknown_phase_keeps_reservation_and_never_launches_next(tmp_path,monkeypatch):
    config,output=setup_config(tmp_path);stages=[]
    def stage(c,phase,campaign,kind,cpu):
        stages.append(phase)
        if not cpu:raise RuntimeError('lost worker response')
    monkeypatch.setattr(coordinate_probe,'run_stage',stage)
    with pytest.raises(RuntimeError,match='lost worker'):
        coordinate_probe.run(config,execute=True)
    result=json.loads((output/'current.json').read_text())
    assert result['status']=='HALTED_NO_RETRY' and result['committed_cap']==8
    assert result['actual_model_calls_known']==2 and stages==['probe-gate','probe-00']
    with pytest.raises(FileExistsError):coordinate_probe.run(config,execute=True)
