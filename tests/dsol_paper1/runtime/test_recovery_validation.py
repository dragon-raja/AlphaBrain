"""Scope byte verification without weakening the complete recovery contract."""
import json
import os
from pathlib import Path
from collections import Counter
import pytest
from benchmarks.LIBERO import controlled_recovery as adapter
from AlphaBrain.research.dsol.data import native_recovery
from AlphaBrain.research.dsol.execution.manifest import file_digest


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return str(path)
    weights = save('model/model.safetensors', 'synthetic-weights')
    norm = save('model/assets/physical-intelligence/libero/norm_stats.json', {})
    model = dict(path=str(tmp_path / 'model'), weights_sha256=file_digest(weights),
                 normalization_sha256=file_digest(norm))
    inputs = save('inputs.json', dict(records=[]))
    sim = save('sim.json', dict(inputs=inputs, models=dict(canonical=model)))
    execution = save('execution.json', {})
    design = save('design.json', dict(operation_policy=dict(checkpoint_sha256=model['weights_sha256'])))
    paired = save('paired.json', dict(design=design, cohorts=dict(A=dict(sim_config=sim, execution_noise=execution))))
    noise = save('noise.json', {})
    baseline = save('baseline.json', dict(states=[], paired_config=paired, score_noise=noise))
    rows = [dict(state_key='state', view_id=i, method='local', suite='libero_spatial',
                 task='task', split='validation') for i in (1, 2)]
    monkeypatch.setattr(native_recovery, 'rows', lambda *a: rows)
    schedule = save('schedule.json', rows)
    artifacts = {i: save(f'reference-{i}.json', dict(value=i)) for i in (0, 1, 2)}
    refs = save('refs.json', dict(status='PASS_COMPLETE_BASELINE_IDENTITIES', views={
        f'state::{i}': dict(artifacts={p: file_digest(p)}) for i, p in artifacts.items()}))
    bound = [weights, norm, inputs, sim, execution, design, paired, noise, baseline, schedule, refs]
    c = dict(schema=native_recovery.SCHEMA, model='native_reference', score_scale=.16535300100378683,
             query_cap=8, selection_members=[0, 1], validation_members=[2, 3],
             baseline_campaign=baseline, schedule=schedule, paired_config=paired, score_noise=noise,
             baseline_references=refs, actors={}, blocks=native_recovery.blocks(rows, size=1),
             limits=native_recovery.limits(rows), input_hashes={p: file_digest(p) for p in bound})
    path = save('campaign.json', c)
    return path, c, artifacts, weights


def test_full_preflight_and_scoped_worker_check_exact_same_global_inputs(campaign, monkeypatch):
    path, c, refs, _ = campaign; calls = []
    def traced(p): calls.append(str(p)); return file_digest(p)
    monkeypatch.setattr(adapter, 'file_digest', traced)
    assert adapter.validate_campaign(path) == c
    counts = Counter(calls)
    assert all(counts[p] == 1 for p in refs.values())
    calls.clear()
    assert adapter.validate_campaign(path, block_index=0) == c
    assert set(c['input_hashes']).issubset(calls)
    assert refs[0] in calls and refs[1] in calls and refs[2] not in calls


def test_other_blocks_corruption_caught_by_full_preflight_and_own_worker(campaign):
    path, _, refs, _ = campaign
    Path(refs[2]).write_text('changed')
    adapter.validate_campaign(path, block_index=0)
    for scope in ({}, dict(block_index=1)):
        with pytest.raises(ValueError, match='Referenced baseline'): adapter.validate_campaign(path, **scope)


def test_same_size_same_mtime_change_is_rehashed(campaign):
    path, _, refs, _ = campaign
    adapter.validate_campaign(path, block_index=0)
    p = Path(refs[1]); stat = p.stat(); p.write_text(p.read_text().replace('1', '9'))
    os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='Referenced baseline'): adapter.validate_campaign(path, block_index=0)


def test_global_weights_still_checked_on_every_block(campaign):
    path, _, _, weights = campaign
    adapter.validate_campaign(path, block_index=0); Path(weights).write_text('changed')
    with pytest.raises(ValueError, match='Recovery input changed'): adapter.validate_campaign(path, block_index=0)


@pytest.mark.parametrize('index', [-1, 2, True, 0.5, '0'])
def test_invalid_scope_rejected(campaign, index):
    with pytest.raises(ValueError, match='block index'): adapter.validate_campaign(campaign[0], block_index=index)


def test_unrelated_block_contract_still_checked(campaign):
    path, c, _, _ = campaign
    c['blocks'][1]['limits']['score'] += 1; Path(path).write_text(json.dumps(c))
    with pytest.raises(ValueError, match='budget or address'): adapter.validate_campaign(path, block_index=0)


def test_actual_preparation_and_validator_bind_inherited_missing_model_assets(campaign, tmp_path):
    from benchmarks.LIBERO.recovery_preparation import prepare_campaigns
    path, c, _, weights = campaign
    c['input_hashes'].pop(weights)
    norm = str(Path(weights).parent / 'assets/physical-intelligence/libero/norm_stats.json')
    c['input_hashes'].pop(norm)
    Path(path).write_text(json.dumps(c))
    with pytest.raises(ValueError, match='Unbound recovery asset'):
        adapter.validate_campaign(path)
    rows = [r for b in c['blocks'] for r in b['rows']]
    result = prepare_campaigns(c, rows, tmp_path / 'prepared', prefix='bound', expected_limits=c['limits'])
    assert len(result['phases']) == 1
    prepared = adapter.validate_campaign(result['phases'][0]['campaign'])
    assert prepared['input_hashes'][weights] == file_digest(weights)
    assert prepared['input_hashes'][norm] == file_digest(norm)
    assert prepared['blocks'][0]['rows'] == rows
    assert prepared['limits'] == c['limits']
    assert weights not in c['input_hashes']  # Failed historical input is unchanged.
    with pytest.raises(FileExistsError):
        prepare_campaigns(c, rows, tmp_path / 'prepared', prefix='bound', expected_limits=c['limits'])


@pytest.mark.parametrize('kind', ['model_bytes', 'conflicting_identity', 'changed_config'])
def test_preparation_cannot_bless_changed_assets(campaign, kind):
    from benchmarks.LIBERO.recovery_preparation import bind_model_assets
    _, c, _, weights = campaign
    if kind == 'model_bytes': Path(weights).write_text('changed')
    elif kind == 'conflicting_identity': c['input_hashes'][weights] = '0' * 64
    else:
        paired = json.loads(Path(c['paired_config']).read_text())
        Path(paired['cohorts']['A']['sim_config']).write_text('{}')
    with pytest.raises(ValueError): bind_model_assets(c)
