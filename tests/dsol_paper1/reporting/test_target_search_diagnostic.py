import csv
import pytest
from reports.paper1.target_search_diagnostic import compare_pi05


def test_comparison_joins_physical_angles_not_view_numbers_or_partial_panel(tmp_path):
    report=tmp_path/'old';report.mkdir();out=tmp_path/'new';out.mkdir()
    with (report/'baseline-spatial.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['state_key','yaw_deg','pitch_deg','success_repeats','execution_repeats'])
        w.writeheader();w.writerows([dict(state_key='s',yaw_deg=y,pitch_deg=0,success_repeats=n,execution_repeats=2) for y,n in [(0,2),(20,1)]])
    rows=[dict(method=m,state_key='s',selected_angles=[a,0],executions=[dict(success=False),dict(success=True)]) for m,a in [('reference',0),('stay',20)]]
    result=compare_pi05(rows,report,out)
    assert result['matched']['reference']['pi05_success']==1.
    assert result['matched']['stay']['pi05_success']==.5
    rows[1]['selected_angles']=[-20,0]
    with pytest.raises(KeyError):compare_pi05(rows,report,out)
