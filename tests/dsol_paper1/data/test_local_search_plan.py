from copy import deepcopy
import json
from pathlib import Path
import pytest

from AlphaBrain.research.dsol.data.local_search_plan import validate_plan


def plan():
    return json.loads((Path(__file__).resolve().parents[3] / 'configs/dsol_paper1/search/dynamic_local_v1.json').read_text())


def test_budget_counts_queries_and_execution_separately():
    result = validate_plan(plan())
    assert result['evaluation_episodes'] == 192
    assert result['evaluation_search_trajectories'] == 544  # stay uses one query
    assert result['evaluation_execution_trajectories'] == 10560
    assert result['future_trajectories'] == 19296


@pytest.mark.parametrize('change', ['overlap', 'horizon', 'heldout', 'duplicate_seed'])
def test_reject_silent_protocol_drift(change):
    p = deepcopy(plan())
    if change == 'overlap': p['evaluation']['tasks'][0]['indices'][0] = 0
    elif change == 'horizon': p['evaluation']['horizons']['libero_spatial'] = 300
    elif change == 'heldout': p['evaluation']['tasks'][2]['seen_by_actor'] = True
    else: p['evaluation']['execution_repeat_ids'][1] = 0
    with pytest.raises(ValueError): validate_plan(p)
