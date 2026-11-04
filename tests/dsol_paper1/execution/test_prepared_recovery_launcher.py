"""Opt-in run-local allocation tests; all heavyweight IO/exports are replaced."""
import importlib.util
import json
import os
from pathlib import Path
import pytest


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    directory = os.environ.get('DSOL_RECOVERY_OPT_PREPARATION')
    if not directory: pytest.skip('Explicit run-local preparation required')
    monkeypatch.syspath_prepend(directory)
    path = Path(directory) / 'recovery_control_stage2048_opt2.py'
    spec = importlib.util.spec_from_file_location('prepared_recovery_under_test', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    real_run = m.RUN
    schedule = m.read(real_run / 'protocol/recovery/schedule.json')
    monkeypatch.setattr(m, 'RUN', tmp_path)
    (tmp_path / 'protocol/recovery').mkdir(parents=True)
    (tmp_path / 'protocol/recovery/schedule.json').write_text(json.dumps(schedule))
    (tmp_path / 'protocol/campaign.json').write_text(json.dumps(dict(paired_config='synthetic', score_noise='synthetic', input_hashes={})))
    refs = tmp_path / 'refs.json'; refs.write_text('{}')
    actor = tmp_path / 'actor.pt'; actor.write_text('CPU fixture, never evaluated')
    controls = tmp_path / 'controls.pt'; controls.write_text('CPU fixture')
    names = ['old_ppo'] + [f'{a}-seed-{s}' for a in ('A', 'B') for s in range(3)]
    monkeypatch.setattr(m, 'baseline_references', lambda: refs)
    monkeypatch.setattr(m, 'export_actors', lambda: {n: dict(path=str(actor), controls=str(controls)) for n in names})
    validated = []; monkeypatch.setattr(m, 'validate_campaign', lambda p: validated.append(str(p)))
    return m, schedule, validated


def test_actual_prepare_preserves_all_science_with_larger_resident_waves(prepared):
    m, schedule, validated = prepared
    result = m.prepare()
    assert len(result['phases']) == len(validated) == 23
    assert result['phases'][0]['searches'] == 3
    assert result['limits'] == dict(episodes=36480, score=328320, paired_execution=2407680, nominal_execution=0)
    expected = [r for split in ('validation','seen_new_initials','unseen_test') for r in schedule if r['split'] == split]
    actual = []
    for phase in result['phases']:
        c = m.read(phase['campaign'])
        assert 1 <= len(c['blocks']) <= 288
        assert c['training_stage'] in c['input_hashes']
        assert all(b['index'] == i and b['host'] == ['fresh','gnu','video'][i % 3] for i,b in enumerate(c['blocks']))
        actual.extend(r for b in c['blocks'] for r in b['rows'])
    assert actual == expected
    with pytest.raises(FileExistsError): m.prepare()


def test_export_failure_prevents_campaign_creation(prepared, monkeypatch):
    m, _, validated = prepared
    def failed(): raise RuntimeError('stage checkpoints not pinned')
    monkeypatch.setattr(m, 'export_actors', failed)
    with pytest.raises(RuntimeError): m.prepare()
    assert not validated and not (m.RUN / 'protocol/recovery/allocation.json').exists()


@pytest.fixture
def pipeline(prepared, monkeypatch):
    m, schedule, validated = prepared
    directory = Path(os.environ['DSOL_RECOVERY_OPT_PREPARATION'])
    path = directory / 'recovery_pipeline_stage2048_opt2.py'
    spec = importlib.util.spec_from_file_location('prepared_pipeline_under_test', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'RUN', m.RUN)
    monkeypatch.setattr(runner, 'prepare', m.prepare)
    import stage2048_contract
    monkeypatch.setattr(stage2048_contract, 'pins', lambda: {'synthetic': True})
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    save(m.RUN / 'training/current.json', dict(status='TRAINING_COMPLETE', stage_sha256=stage2048_contract.STAGE_SHA))
    save(m.RUN / 'current.json', dict(status='BASELINE_COMPLETE_PPO_NOT_STARTED'))
    save(m.RUN / 'recovery-optimization-20261026/baseline-preparation.json', {'synthetic': True})
    (m.RUN / 'preparation').mkdir(); (m.RUN / 'evaluation-recovery').mkdir(); (m.RUN / 'evidence').mkdir()
    for name in ['recovery_pipeline_stage2048_opt2.py','recovery_control_stage2048_opt2.py','recovery_stage_stage2048_opt2.py','final_report_stage2048.py','stage2048_contract.py']:
        (m.RUN / 'preparation' / name).write_bytes((directory / name).read_bytes())
    events = []
    def stage(phase, campaign, cpu=False):
        c = m.read(campaign); events.append((phase, cpu))
        save(m.RUN / 'execution-recovery' / (phase + ('-cpu' if cpu else '-real')) / 'completion.json',
             dict(status='COMPLETE', actual_model_calls=0, actual_manipulation_episodes=0 if cpu else c['limits']['episodes']))
    monkeypatch.setattr(runner, 'stage', stage)
    monkeypatch.setattr(runner, 'gate', lambda p: events.append(('GATE', True)))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: events.append(('REPORT', True)))
    return runner, events


def test_actual_pipeline_gates_then_runs_all_registered_outcomes(pipeline):
    runner, events = pipeline; runner.main()
    assert events[:3] == [('contract', True), ('wave-000', False), ('GATE', True)]
    assert len([e for e in events if e[1] is False]) == 23 and events[-1] == ('REPORT', True)
    current = runner.read(runner.RUN / 'evaluation-recovery/current.json')
    assert current['status'] == 'COMPLETE' and current['completed_episodes'] == 36480
    with pytest.raises(FileExistsError): runner.main()


def test_actual_pipeline_gate_failure_never_dispatches_second_wave(pipeline, monkeypatch):
    runner, events = pipeline
    def fail(p): raise ValueError('prediction mismatch')
    monkeypatch.setattr(runner, 'gate', fail)
    with pytest.raises(ValueError): runner.main()
    assert events == [('contract', True), ('wave-000', False)]


def test_real_stage_contract_receipt_path_is_part_of_entry_contract(tmp_path, monkeypatch):
    """Do not rename the CPU phase independently of the frozen real entry."""
    directory = os.environ.get('DSOL_RECOVERY_OPT_PREPARATION')
    if not directory: pytest.skip('Explicit run-local preparation required')
    monkeypatch.syspath_prepend(directory)
    spec = importlib.util.spec_from_file_location('actual_recovery_stage_preflight',
        Path(directory) / 'recovery_stage_stage2048_opt2.py')
    stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
    from stage2048_contract import STAGE_SHA
    monkeypatch.setattr(stage, 'RUN', tmp_path)
    campaign = tmp_path / 'campaign.json'; campaign.write_text('{}')
    monkeypatch.setattr(stage, 'validate_campaign', lambda p:
        dict(blocks=[{}], training_stage='stage', input_hashes=dict(stage=STAGE_SHA)))
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))
    save(tmp_path/'optimization-deploy-20261025-0025/real-gate.json',
         dict(status='PASS_EXACT_TRANSPORT_ONLY'))
    save(tmp_path/'training/current.json', dict(status='TRAINING_COMPLETE', stage_sha256=STAGE_SHA))
    save(tmp_path/'execution-recovery/renamed-contract-cpu/completion.json', dict(status='COMPLETE'))
    class PassedPreflight(Exception): pass
    original_read = stage.read
    def read(path):
        if Path(path) == tmp_path/'execution-r2/wave-004-real/binding.json':
            raise PassedPreflight('Before constructing bindings, permits or remote processes')
        return original_read(path)
    monkeypatch.setattr(stage, 'read', read)
    with pytest.raises(FileNotFoundError, match='contract-cpu'):
        stage.main('wrong-contract-name', campaign, False)
    save(tmp_path/'execution-recovery/contract-cpu/completion.json', dict(status='COMPLETE'))
    with pytest.raises(PassedPreflight):
        stage.main('canonical-contract-name', campaign, False)
