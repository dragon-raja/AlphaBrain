import json

import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import sha
from AlphaBrain.research.dsol.data.evaluation_preparation import prepare_population, verify_files
from AlphaBrain.research.dsol.data.rig_training import candidate_catalog
from benchmarks.Robocasa365.evaluation_render import pixel_check, pixel_disposition
from benchmarks.Robocasa365.evaluation_render import physical_fields
from types import SimpleNamespace


def fixture_files(tmp_path):
    rows = []
    for split, layout in [('pretrain', 11), ('target', 1)]:
        root = tmp_path / split
        root.mkdir()
        (root / 'payload').write_bytes(b'immutable')
        meta = dict(task='Example', split=split, scene_seed=0, layout_id=layout, style_id=layout)
        manifest = dict(format='robocasa_eval_snapshot_bundle_v1', metadata=meta,
                        files={'payload': dict(bytes=9, sha256=sha(root / 'payload'))})
        (root / 'manifest.json').write_text(json.dumps(manifest))
        rows.append(dict(meta, path=str(root), navigation=False))
    pop = tmp_path / 'population.json'
    cat = tmp_path / 'catalog.json'
    pop.write_text(json.dumps(rows))
    cat.write_text(json.dumps(dict(catalog=candidate_catalog())))
    return pop, cat


def test_preserves_population_and_seals_target(tmp_path):
    pop, cat = fixture_files(tmp_path)
    plan = prepare_population(pop, cat)
    assert plan['counts']['candidate_observations'] == 194
    assert [j['use'] for j in plan['jobs']] == ['seen_layout_pool', 'sealed_layout_test']
    assert len({j['initial_id'] for j in plan['jobs']}) == 2


@pytest.mark.parametrize('mutation', ['duplicate', 'layout', 'hash'])
def test_rejects_identity_or_integrity_changes(tmp_path, mutation):
    pop, cat = fixture_files(tmp_path)
    rows = json.loads(pop.read_text())
    if mutation == 'duplicate':
        rows.append(rows[0])
    elif mutation == 'layout':
        rows[0]['layout_id'] = 1
    else:
        (tmp_path / 'pretrain/payload').write_bytes(b'corrupted')
    pop.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        prepare_population(pop, cat)


def test_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        verify_files(tmp_path, {'../outside': {}})


def test_pixel_gate_strict():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    assert pixel_check(image, image)['pass']
    changed = image.copy()
    changed[0, 0, 0] = 2
    assert pixel_check(changed, image)['pass']
    changed[0, 0, 0] = 3
    assert not pixel_check(changed, image)['pass']
    with pytest.raises(ValueError):
        pixel_check(image.astype(float), image)


def test_physical_field_capture_is_copied_and_excludes_solver_warmstart():
    sim = SimpleNamespace(get_state=lambda: np.zeros(5),
        data=SimpleNamespace(ctrl=np.ones(2), qacc_warmstart=np.ones(3)),
        model=SimpleNamespace(body_pos=np.zeros((2, 3))))
    before = physical_fields(sim)
    sim.data.ctrl[0] = 5
    assert before['ctrl'][0] == 1
    assert 'qacc_warmstart' not in before
    assert set(before) == {'sim_state', 'ctrl', 'body_pos'}


def test_quarantine_never_converts_failed_pixel_gate_to_pass():
    checks = {k: {'pass': True, 'max_abs': 0} for k in ['left', 'right', 'wrist']}
    assert pixel_disposition(checks) == 'PASS'
    checks['wrist'] = {'pass': False, 'max_abs': 1}
    assert pixel_disposition(checks) == 'HOLD_CANONICAL_PIXELS'
    checks['wrist']['max_abs'] = 2
    with pytest.raises(ValueError):
        pixel_disposition(checks)


def test_explicit_mean_profile_preserves_legacy_default_and_hard_limits():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    changed = image.copy()
    changed.reshape(-1)[:700] = 1
    assert not pixel_check(changed, image)['pass']
    assert pixel_check(changed, image, .005)['pass']
    changed[0, 0, 0] = 3
    assert not pixel_check(changed, image, .005)['pass']
    with pytest.raises(ValueError):
        pixel_check(changed, image, 1.)
