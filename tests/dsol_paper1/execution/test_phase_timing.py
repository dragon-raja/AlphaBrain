import copy

import pytest

from AlphaBrain.research.dsol.execution.phase_timing import summarize_completed_stages


def stage(phase='one', seconds=10, calls=10, episodes=2, **metrics):
    return dict(phase=phase, seconds=seconds, actual_model_calls=calls,
                actual_episodes=episodes, status='COMPLETE', cpu=False,
                dispatch_enabled=False, metrics=metrics)


def test_aggregate_rates_are_weighted_and_inputs_unchanged():
    rows = [stage(), stage('two', seconds=90, calls=30, episodes=8)]
    before = copy.deepcopy(rows)
    result = summarize_completed_stages(iter(rows))
    assert result['calls_per_stage_second'] == .4
    assert result['episodes_per_stage_hour'] == 360
    assert result['mean_stage_seconds'] == 50
    assert rows == before
    assert result['dispatch_authority'] is False


def test_missing_metric_does_not_mean_zero_and_overlap_is_not_subtracted():
    result = summarize_completed_stages([
        stage(poll_seconds=12, completion_inspection_seconds=8), stage('two')])
    timer = result['timers']['poll_seconds']
    assert timer['stages'] == 1
    assert timer['timer_to_covered_wall_ratio'] == 1.2
    assert timer['seconds_per_covered_stage'] == 12
    assert result['stage_wall_seconds'] == 20
    assert 'overlap' in result['limitation']


@pytest.mark.parametrize('change', [dict(status='RUNNING'), dict(cpu=True),
    dict(dispatch_enabled=True), dict(cpu=None), dict(seconds=0),
    dict(seconds=float('nan')), dict(seconds=float('inf')),
    dict(actual_model_calls=-1), dict(actual_episodes=True),
    dict(actual_model_calls=2.5), dict(metrics={'poll_seconds': -1}),
    dict(metrics={'poll_seconds': float('nan')}), dict(phase='')])
def test_invalid_or_unsettled_evidence_rejected(change):
    row = stage()
    row.update(change)
    with pytest.raises(ValueError):
        summarize_completed_stages([row])


def test_duplicate_and_empty_cohorts_rejected():
    for rows in ([], [stage(), stage()]):
        with pytest.raises(ValueError):
            summarize_completed_stages(rows)


def test_zero_calls_allowed_for_settled_non_model_work():
    result = summarize_completed_stages([stage(calls=0, episodes=0)])
    assert result['calls_per_stage_second'] == 0
