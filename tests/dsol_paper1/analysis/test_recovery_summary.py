import pytest
from AlphaBrain.research.dsol.analysis.recovery_summary import paired_summary


def test_task_weighting_and_harm_are_not_hidden_by_more_examples():
    rows=[dict(task='a',state_key=str(i),view_id=1,repeat=0,method='B',
        success=True,stay_success=False,nominal_success=True) for i in range(3)]
    rows.append(dict(task='b',state_key='last',view_id=1,repeat=0,method='B',
        success=False,stay_success=True,nominal_success=True))
    value=paired_summary(rows,bootstrap_samples=100)
    assert value['rescue_count']==3 and value['harm_count']==1
    assert value['task_macro']['success']==.5 and value['task_macro']['net_recovery']==0
    with pytest.raises(ValueError,match='Repeated'):paired_summary(rows+[rows[0]])
    rows[0]['success']=None
    with pytest.raises(ValueError,match='Boolean'):paired_summary(rows)
