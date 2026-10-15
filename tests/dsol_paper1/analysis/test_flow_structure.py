import numpy as np

from AlphaBrain.research.dsol.analysis.flow_structure import aggregate, choices, RULES


def test_undefined_direction_cannot_win_or_silently_fall_back():
    s = np.ones((3, 8, 5))
    s[0] = 0
    s[0, 0, 0] = np.nan
    s[2] = 2
    assert choices(s) == [1, 1, 1]
    assert choices(s * np.nan) == [-1, -1, -1]


def test_missing_winner_is_unknown_not_zero_or_dropped():
    rows=[]
    for i in range(2):
        row=dict(initial=i,task='task',split='target',selection={})
        for mode in ('include','exclude'):
            row['selection'][mode]={}
            for rule in RULES:
                unknown=i==1 and rule!='original'
                row['selection'][mode][rule]=[dict(view='new' if rule!='original' else 'old',
                    success=None if unknown else .5,outcomes=None if unknown else [0,1]) for _ in range(3)]
        rows.append(row)
    result=aggregate(rows)['all']['include']['near'][0]
    assert result['success'] is None and result['label_coverage']==1
    assert result['success_bounds']==[.25,.75]
    assert result['paired_delta_bounds']==[-.25,.25]
