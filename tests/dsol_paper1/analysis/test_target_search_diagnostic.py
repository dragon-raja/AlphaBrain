import pytest
from AlphaBrain.research.dsol.analysis.target_search_diagnostic import endpoint_outcomes, pose_key, training_diagnostic


def case(success, repeat=0):
    return dict(state_key='s',selected_angles=[1.,2.],executions=[dict(repeat=repeat,success=success)])


def test_duplicate_noise_is_not_an_independent_trial_and_conflict_is_visible():
    a=case(True)
    assert endpoint_outcomes([a,a])[pose_key('s',[1,2])]['repeats']==1
    out=endpoint_outcomes([a,case(False)])[pose_key('s',[1,2])]
    assert out['conflicting'] and out['success'] is None
    assert endpoint_outcomes([a,case(False,1)])[pose_key('s',[1,2])]['success']==.5


def test_missing_pose_remains_unlabelled_and_states_stay_separate():
    a=case(True);b=dict(case(False),state_key='other')
    labels=endpoint_outcomes([a,b]);assert len(labels)==2
    assert pose_key('s',[1.01,2]) not in labels


def test_reward_audit_rejects_unreconciled_costs():
    es=[dict(run='r',policy_update=0,initial_accel=1.,best_accel=.98,queries=3,
             initial_identity=dict(state_key=str(i))) for i in range(8)]
    u=dict(run='r',update=1,transitions=16,reward_settlement=dict(denominator=1.,rewards=[0.]*8),
           means=dict(mean_stop_probability=.2,approximate_kl=.01,clip_fraction=.1))
    assert training_diagnostic(es,[u])[0]['max_reward_reconciliation_error']<1e-9
    u['reward_settlement']['rewards'][0]=.1
    with pytest.raises(ValueError,match='reconcile'):training_diagnostic(es,[u])


def test_reward_audit_supports_complete_mainline_batch():
    es=[dict(run='r',policy_update=0,initial_accel=1.,best_accel=.98,queries=3,
             initial_identity=dict(state_key=str(i))) for i in range(32)]
    u=dict(run='r',update=1,transitions=64,reward_settlement=dict(denominator=1.,rewards=[0.]*32),
           means=dict(mean_stop_probability=.2,approximate_kl=.01,clip_fraction=.1))
    result=training_diagnostic(es,[u],episodes_per_update=32)[0]
    assert result['max_reward_reconciliation_error']<1e-9
    assert result['covered_states']==32
    with pytest.raises(ValueError,match='complete'):
        training_diagnostic(es[:-1],[u],episodes_per_update=32)
