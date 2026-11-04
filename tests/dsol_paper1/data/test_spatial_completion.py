import pytest
from AlphaBrain.research.dsol.data.spatial_completion import plan


def test_reuse_is_deterministic_and_never_success_selected():
    expected=[dict(state_key='s',view_id=v,repeat=m) for v in range(3) for m in (0,1)]
    prior=[dict(expected[0],source_result=p,success=False,steps=720,replans=90,trajectory_identity='same') for p in ('z','a')]
    result=plan(expected,prior,maximum_replans=90)
    assert result['counts']==dict(expected=6,prior_actual=2,reused=1,duplicates=1,new_operations=5)
    assert result['reuse'][0]['prior']['source_result']=='a'
    assert result['limits']['paired_execution']==450
    assert result['missing']==expected[1:]
    assert 'prior' not in expected[0]


def test_disagreement_and_unknown_prior_cannot_be_reused():
    expected=[dict(state_key='s',view_id=0,repeat=0)]
    a=dict(expected[0],source_result='a',success=True,steps=10,replans=2,trajectory_identity='a')
    with pytest.raises(ValueError,match='disagrees'):plan(expected,[a,dict(a,source_result='b',success=False)],maximum_replans=90)
    with pytest.raises(ValueError):plan(expected,[dict(a,success=None)],maximum_replans=90)
    with pytest.raises(ValueError):plan(expected,[dict(a,view_id=1)],maximum_replans=90)
    with pytest.raises(ValueError):plan(expected+expected,[],maximum_replans=90)
