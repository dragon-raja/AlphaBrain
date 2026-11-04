import json

import pytest

from reports.paper1.native_mainline_interim import build, check_noise, digest, verified_results


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, seeds=3):
    root, prepared = tmp_path / 'run', tmp_path / 'prepared'
    def row(method):
        return dict(trial_id=method, method=method, state_key='state', view_id=0 if method == 'reference' else 1,
                    suite='suite', task='task', initial=1, split='validation', execution_repeats=[0, 1],
                    angles_deg=[0, 0] if method == 'reference' else [10, 0],
                    max_steps=10, settle_steps=10, replan_steps=10)
    all_ours = [row(f'G-seed-{i}') for i in range(3)]
    write(prepared / 'ours-blocks.json', [dict(rows=all_ours)])
    for phase, rows in [('baseline', [row('stay'), row('reference')]), ('ours', all_ours[:seeds])]:
        label = phase + '-000'
        write(root / 'campaigns' / (label + '.json'), dict(blocks=[dict(index=0, rows=rows)]))
        tasks = [dict(r, repeat=repeat, success=r['method'] not in ('stay', 'G-seed-2'),
                      queries=2, selected_angles=[1, 1], execution_noise_sha256=['noise-' + str(repeat)])
                 for r in rows for repeat in r['execution_repeats']]
        searches = [dict(r, queries=2, query_cap=8, reason='stop') for r in rows]
        folder = root / 'execution' / (label + '-real') / 'results/block-0000'
        write(folder / 'results.json', dict(tasks=tasks, searches=searches))
        write(folder / 'complete.json', dict(status='COMPLETE', artifacts={'results.json': digest(folder / 'results.json')}))
        write(root / 'settlements' / (label + '.json'), dict(status='COMPLETE', actual_episodes=len(tasks)))
    return root, prepared


def test_complete_panel_counts_baseline_reuse_and_harm(tmp_path):
    root, prepared = fixture(tmp_path)
    result = build(root, prepared, tmp_path / 'report')
    assert result['paired']['all']['pairs'] == 6
    assert result['paired']['all']['rescue_count'] == 4
    assert result['paired']['all']['harm_count'] == 0
    assert result['paired']['all']['task_macro']['net_recovery'] == pytest.approx(2 / 3)
    assert result['full_baseline']['all/reference']['episodes'] == 2
    assert result['search_costs']['all']['searches'] == 3


def test_incomplete_three_seed_task_is_excluded(tmp_path):
    root, prepared = fixture(tmp_path, seeds=2)
    result = build(root, prepared, tmp_path / 'report')
    assert result['full_tasks'] == []
    assert result['paired'] == {}
    assert result['excluded_partial_task_outcomes'] == 4


def test_changed_results_rejected(tmp_path):
    root, _ = fixture(tmp_path)
    folder = root / 'execution/ours-000-real/results/block-0000'
    with (folder / 'results.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='changed result'):
        verified_results(folder)


def test_noise_requires_matching_prefix():
    check_noise({'execution_noise_sha256': ['a']}, {'execution_noise_sha256': ['a', 'b']})
    with pytest.raises(ValueError, match='noise'):
        check_noise({'execution_noise_sha256': ['a']}, {'execution_noise_sha256': ['b']})


def test_complete_report_consumes_bound_baseline_supplement_without_reexecution(tmp_path):
    root, prepared = fixture(tmp_path)
    campaign = root / 'campaigns/baseline-000.json'
    folder = root / 'execution/baseline-000-real/results/block-0000'
    rows = json.loads(campaign.read_text())['blocks'][0]['rows']
    group = dict(output=str(folder), rows=rows, receipt_sha256=digest(folder / 'complete.json'),
                 source_record=str(campaign), source_record_sha256=digest(campaign))
    (root / 'settlements/baseline-000.json').unlink()
    result = build(root, prepared, tmp_path / 'report', baseline_groups=[group], require_complete=True)
    assert result['complete_frozen_panel'] is True
    assert result['settled_outcomes'] == dict(baseline=4, ours=6)
    assert result['paired']['all']['rescue_count'] == 4
    assert result['new_model_calls'] == 0
    with pytest.raises(ValueError, match='Repeated search'):
        build(root, prepared, tmp_path / 'duplicate', baseline_groups=[group, group], require_complete=True)
    group['receipt_sha256'] = 'changed'
    with pytest.raises(ValueError, match='provenance'):
        build(root, prepared, tmp_path / 'changed', baseline_groups=[group], require_complete=True)


def test_final_report_refuses_missing_seed_or_missing_baseline(tmp_path):
    root, prepared = fixture(tmp_path, seeds=2)
    with pytest.raises(ValueError, match='Complete frozen'):
        build(root, prepared, tmp_path / 'missing-seed', require_complete=True)
    (root / 'settlements/baseline-000.json').unlink()
    with pytest.raises(ValueError, match='Complete frozen'):
        build(root, prepared, tmp_path / 'missing-baseline', require_complete=True)
