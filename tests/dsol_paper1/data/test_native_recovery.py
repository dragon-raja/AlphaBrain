from copy import deepcopy
from collections import Counter
import pytest
from AlphaBrain.research.dsol.data import native_recovery as panel
from AlphaBrain.research.dsol.data.native_baseline import HORIZONS


def fixture():
    states=[];records=[]
    for suite,horizon in HORIZONS.items():
        for task in range(10):
            split='train' if task<5 else 'validation' if task<7 else 'test_unseen_task'
            for initial in range(10):
                i=len(states);key=f'{suite}/{task}/{initial}'
                states.append(dict(cohort='A',initial=i,state_key=key,task=f'{suite}/{task}',suite=suite,max_steps=horizon))
                records.append(dict(state=dict(pair_key=key,split=split,
                    initialization=dict(init_state_index=(20 if split=='validation' else 40)+initial))))
    return states,records


def test_complete_outcome_independent_recovery_panel():
    states,records=fixture();rows=panel.rows(states,records)
    assert len(rows)==18240
    assert Counter(r['split'] for r in rows)==dict(validation=3200,seen_new_initials=6400,unseen_test=8640)
    assert panel.limits(rows)==dict(episodes=36480,score=328320,paired_execution=2407680,nominal_execution=0)
    assert len({(r['state_key'],r['view_id'],r['method']) for r in rows})==len(rows)
    for i in range(400):
        local=[r for r in rows if r['initial']==i and r['method']=='local']
        assert len(local)==8 and {((r['view_id']-1)%8) for r in local}==set(range(8))
    blocks=panel.blocks(rows)
    assert sum(b['limits']['episodes'] for b in blocks)==36480
    assert len({str(a) for b in blocks for a in b['addresses']})==36480


def test_evaluation_state_cannot_enter_training_initial_range():
    states,records=fixture();records=deepcopy(records)
    records[0]['state']['initialization']['init_state_index']=0
    with pytest.raises(ValueError,match='overlaps'):panel.rows(states,records)


def test_larger_waves_preserve_first_gate_order_addresses_and_budget():
    states, records = fixture(); rows = panel.rows(states, records)
    blocks = []
    for split in ('validation', 'seen_new_initials', 'unseen_test'):
        blocks.extend(panel.blocks([r for r in rows if r['split'] == split]))
    old = panel.waves(blocks); new = panel.waves(blocks, size=288)
    assert old[0] == new[0] == blocks[:1]
    assert [b for wave in old for b in wave] == [b for wave in new for b in wave] == blocks
    assert len(old) == 44 and len(new) == 23
    for key, value in panel.limits(rows).items():
        assert sum(b['limits'][key] for wave in new for b in wave) == value


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_wave_size_requires_positive_integer(size):
    with pytest.raises(ValueError): panel.waves([{}], size=size)


def test_baseline_hash_obligations_deduplicate_and_reject_conflicts():
    refs = dict(views={'s::0': dict(artifacts={'nominal': 'a'}),
                       's::1': dict(artifacts={'query': 'b'})})
    rows = [dict(state_key='s', view_id=1)] * 3
    assert panel.reference_artifacts(refs, rows) == {'nominal': 'a', 'query': 'b'}
    refs['views']['s::1']['artifacts']['nominal'] = 'conflict'
    with pytest.raises(ValueError, match='Conflicting'): panel.reference_artifacts(refs, rows)
