from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.analysis.core_selection import choose_structure
from AlphaBrain.research.dsol.data.search_core import VARIANTS


def outcomes():
    rows=[dict(split='validation',seed=s,state_key=f't{t}-i{i}',task=f't{t}',view_id=v,repeat=r,success=False) for s in (0,1) for t in range(8) for i in (0,1) for v in (1,2) for r in (0,1)]
    return {v:deepcopy(rows) for v in VARIANTS}


def test_selection_full_pairing_and_deterministic_ties():
    x=outcomes();assert choose_structure(x)['selected_base']=='U0'
    for v in ('U2','U12'):
        for r in x[v]:r['success']=True
    assert choose_structure(x)['selected_base']=='U2'
    x['U2'][0]['split']='unseen_test'
    with pytest.raises(ValueError):choose_structure(x)


def test_reject_partial_or_duplicate():
    x=outcomes();x['U0'].pop()
    with pytest.raises(ValueError):choose_structure(x)
    x=outcomes();x['U1'][-1]=x['U1'][0]
    with pytest.raises(ValueError):choose_structure(x)
