import json
from pathlib import Path
from AlphaBrain.research.dsol.data.resource_graph import audit_graph


def test_public_manifest_has_no_personal_filesystem_binding():
    root = Path(__file__).resolve().parents[2]
    text = (root / 'configs/resources/paper1_resources.json').read_text()
    assert '/share/' not in text and '/workspace/' not in text
    value = json.loads(text)
    assert audit_graph(value)['status'] == 'PASS'
    for binding in value['execution_bindings']:
        assert (root / binding['adapter']).exists()


def test_robocasa_paper_backend_is_not_relabelled_as_alphabrain():
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / 'configs/resources/paper1_resources.json').read_text())
    binding = next(x for x in value['execution_bindings'] if x['id'] == 'robocasa_paper1')
    assert binding['backend'] == 'openpi_jax'


def test_official_download_lock_has_complete_declared_scope():
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / 'configs/resources/robotwin_official_clean50.lock.json').read_text())
    resources = value['resources']
    assert len(resources) == 54
    assert len([x for x in resources if x['id'].startswith('official-demo-clean-')]) == 50
    assert len([x for x in resources if x['id'].startswith('official-asset-')]) == 3
    assert len({x['id'] for x in resources}) == 54
    assert all(len(x['sha256']) == 64 and value['upstream_revision'] in x['url'] for x in resources)


def test_official_execution_bundle_does_not_use_legacy_pilot():
    root = Path(__file__).resolve().parents[2]
    graph = json.loads((root / 'configs/resources/paper1_resources.json').read_text())
    binding = next(x for x in graph['execution_bindings'] if x['id'] == 'robotwin_future_policy')
    assert binding['status'] == 'not_runtime_validated'
    assert 'robotwin_legacy_clean_pilot' not in binding['resources']
    assert set(binding['required_revisions']) == {
        'robotwin_official_source', 'robotwin_xpolicylab', 'robotwin_current_clean50',
        'robotwin_official_assets', 'robotwin_pi05_checkpoint'}


def test_robocasa_payload_and_release_are_distinct():
    root = Path(__file__).resolve().parents[2]
    graph = json.loads((root / 'configs/resources/paper1_resources.json').read_text())
    resources = {x['id']: x for x in graph['resources']}
    release = resources['robocasa_matched_data']
    assert set(release['parents']) == {'robocasa_render_payload', 'robocasa_render_recovery'}
    assert resources['robotwin_multiview_derived']['availability'] == 'not_created'
