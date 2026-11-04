import pytest

from AlphaBrain.research.dsol.data.native_comparison_plan import comparison_budget


def test_full_40_task_comparison_includes_both_online_methods():
    result = comparison_budget([220, 280, 300, 520] * 10, initials=10, views=27, repeats=4)
    assert result['states'] == 400 and result['perturbed_starts'] == 10400
    assert result['methods']['panel']['episodes'] == 43200
    assert result['methods']['panel']['total_calls'] == 2894400
    for name in ('local_near_best_accel', 'continuous_ppo'):
        assert result['methods'][name]['searches'] == 10400
        assert result['methods'][name]['episodes'] == 41600
        assert result['methods'][name]['selection_scoring_calls'] == 166400
        assert result['methods'][name]['validation_scoring_calls'] == 20800
        assert result['methods'][name]['total_calls'] == 2932800
    assert result['episodes'] == 126400
    assert result['total_calls'] == 8760000
    assert result['training_updates'] == 0


def test_one_state_two_views_does_not_charge_return_as_another_panel():
    result = comparison_budget([6], initials=1, views=2, repeats=1)
    assert result['methods']['panel']['execution_calls'] == 4
    assert result['methods']['panel']['scoring_calls'] == 8
    assert result['episodes'] == 4  # nominal, perturbed, local, PPO
    assert result['total_calls'] == 52


@pytest.mark.parametrize('horizons,initials,views,repeats', [([],1,2,1), ([0],1,2,1), ([6],0,2,1), ([6],1,1,1), ([6],1,2,True)])
def test_invalid_or_empty_design_is_rejected(horizons, initials, views, repeats):
    with pytest.raises(ValueError):
        comparison_budget(horizons, initials=initials, views=views, repeats=repeats)
