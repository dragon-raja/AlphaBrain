import pytest
from AlphaBrain.research.dsol.data.official_resources import robotwin_clean_plan


def tree():
    return {'revision': 'abc', 'files': [
        {'type': 'file', 'path': path, 'size': 10, 'lfs': {'oid': 'a' * 64}}
        for path in ['dataset/test/demo_clean.zip', 'objects.zip', 'embodiments.zip', 'background_texture.zip']]}


def test_official_plan_pins_sources_and_separates_packaging():
    plan = robotwin_clean_plan(tree(), 'abc', expected_tasks=1)
    assert len(plan['resources']) == 4
    assert '/demo_clean/test' in plan['resources'][0]['destination']
    assert '/resolve/abc/' in plan['resources'][0]['url']
    assert not plan['scope']['randomized500_included']


def test_official_plan_rejects_missing_coverage():
    with pytest.raises(ValueError, match='coverage'):
        robotwin_clean_plan(tree(), 'abc')


def test_official_plan_rejects_unpinned_hash():
    value = tree()
    del value['files'][0]['lfs']
    with pytest.raises(ValueError, match='SHA256'):
        robotwin_clean_plan(value, 'abc', expected_tasks=1)
