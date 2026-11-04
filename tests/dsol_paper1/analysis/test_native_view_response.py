import pytest
from AlphaBrain.research.dsol.analysis.native_view_response import analyze
from AlphaBrain.research.dsol.data.native_view_panel import views


def sample():
    states=[dict(state_key='s',task='task',initial=0)]
    tasks=[dict(state_key='s',view_id=v,repeat=r,success=bool(v==0 or (v in (25,26) and r<2))) for v in range(27) for r in range(4)]
    scores=[dict(state_key='s',view_id=v,member=m,score=float(v!=1)) for v in range(27) for m in range(4)]
    return states,tasks,scores


def test_lowest_score_failure_and_oracle_are_not_conflated():
    states,tasks,scores=sample();out=analyze(states,views(),tasks,scores)['states'][0]
    assert out['primary_min_view']==1
    assert out['original_starts']==dict(episodes=8,stay_success=4,nominal_success=8,selected_success=0,nominal_losses=4,rescues=0,harms=4,recovery_of_nominal_losses=0)
    assert out['global_oracle_success_rate']==1
    assert out['uniform_random_expected_success_rate']==pytest.approx(2/27)


def test_missing_or_duplicated_outcomes_are_not_silently_excluded():
    states,tasks,scores=sample()
    with pytest.raises(ValueError,match='execution surface'):analyze(states,views(),tasks[:-1],scores)
    with pytest.raises(ValueError,match='score surface'):analyze(states,views(),tasks,scores[:-1]+scores[:1])
