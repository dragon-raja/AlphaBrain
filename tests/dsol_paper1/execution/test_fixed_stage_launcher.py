"""Opt-in run-local launcher checks with all collection/optimizer work replaced.

Set DSOL_FIXED_STAGE_PREPARATION to the frozen run's preparation directory.
The generic amendment and checkpoint tests remain in portable tests.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    directory = os.environ.get('DSOL_FIXED_STAGE_PREPARATION')
    if not directory:
        pytest.skip('Explicit run-local preparation required; no production launch')
    monkeypatch.syspath_prepend(directory)
    path = Path(directory) / 'training_pipeline_stage2048.py'
    spec = importlib.util.spec_from_file_location('fixed_stage_under_test', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.RUN = tmp_path; m.ROOT = tmp_path / 'stage'; m.ROOT.mkdir()
    stage = dict(updates=64, episodes_per_run=2048, training_calls_cap=196608)
    monkeypatch.setattr(m, 'contract', lambda: stage)
    (tmp_path / 'preparation').mkdir()
    for name in m.NAMES: (tmp_path / 'preparation' / name).write_text('frozen fixture')
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    save(tmp_path / 'training/first-batch-gate.json', dict(status='PASS_MATCHED_AB_REWARD_ONLY'))
    for v in range(64):
        save(tmp_path / 'execution-ppo' / f'update-{v:04d}-real/completion.json',
             dict(status='COMPLETE', actual_search_episodes=192, actual_model_calls=100, dispatch_enabled=False))
    save(m.ROOT / 'handoff.json', dict(status='OLD_CONTROLLERS_STOPPED_AT_SETTLED_COLLECTION',
         collected_version=62, stage_sha256=m.STAGE_SHA,
         completion_sha256=m.sha(tmp_path / 'execution-ppo/update-0062-real/completion.json')))
    versions = {name: 62 for name in m.runs()}; events = []
    def published(root, name, version):
        assert versions[name] == version
        return dict(version=version, episodes=32 * version)
    def update(v):
        assert set(versions.values()) == {v}
        events.append(('update', v))
        versions.update({name: v + 1 for name in versions})
    monkeypatch.setattr(m, 'published', published)
    monkeypatch.setattr(m, 'update', update)
    monkeypatch.setattr(m, 'collect', lambda v: events.append(('collect', v)))
    monkeypatch.setattr(m, 'pin_stage', lambda: events.append(('pin', 64)))
    return m, stage, events, save


def test_actual_entry_reuses_settled_batch_and_stops_at_64(launcher):
    m, _, events, _ = launcher
    m.main()
    assert events == [('update', 62), ('collect', 63), ('update', 63), ('pin', 64)]
    status = m.read(m.RUN / 'training/current.json')
    assert status['status'] == 'TRAINING_COMPLETE'
    assert status['episodes_per_run'] == 2048 and status['calls_settled'] == 6400
    assert status['full_resume_checkpoints_pinned']
    with pytest.raises(FileExistsError): m.main()
    assert len(events) == 4


def test_checkpoint_pin_failure_does_not_enable_evaluation(launcher, monkeypatch):
    m, _, _, _ = launcher
    def fail(): raise OSError('export unavailable')
    monkeypatch.setattr(m, 'pin_stage', fail)
    with pytest.raises(OSError): m.main()
    assert m.read(m.RUN / 'training/current.json')['status'] == 'PINNING_STAGE_CHECKPOINTS'


def test_reserved_or_failed_collection_never_updates(launcher):
    m, _, events, save = launcher
    path = m.RUN / 'execution-ppo/update-0062-real/completion.json'
    proof = m.read(path); proof['dispatch_enabled'] = True; save(path, proof)
    h = m.read(m.ROOT / 'handoff.json'); h['completion_sha256'] = m.sha(path); save(m.ROOT / 'handoff.json', h)
    with pytest.raises(AssertionError): m.main()
    assert events == []


def test_reduced_budget_enforced_before_next_optimizer(launcher):
    m, stage, events, _ = launcher
    stage['training_calls_cap'] = 6200
    with pytest.raises(AssertionError): m.main()
    assert events == []


def test_unknown_collection_result_not_replayed(launcher, monkeypatch):
    m, _, events, _ = launcher
    def fail(v):
        events.append(('unknown', v)); raise ConnectionError('result unknown')
    monkeypatch.setattr(m, 'collect', fail)
    with pytest.raises(ConnectionError): m.main()
    assert events == [('update', 62), ('unknown', 63)]
    with pytest.raises(FileExistsError): m.main()
    assert len(events) == 2


def test_unknown_optimizer_result_not_replayed(launcher, monkeypatch):
    m, _, events, _ = launcher
    def fail(v):
        events.append(('unknown_update', v)); raise ConnectionError('publication unavailable')
    monkeypatch.setattr(m, 'update', fail)
    with pytest.raises(ConnectionError): m.main()
    with pytest.raises(FileExistsError): m.main()
    assert events == [('unknown_update', 62)]


def test_modified_settled_receipt_rejected(launcher):
    m, _, events, _ = launcher
    p = m.RUN / 'execution-ppo/update-0062-real/completion.json'
    p.write_text(p.read_text() + '\n')
    with pytest.raises(AssertionError): m.main()
    assert events == []
