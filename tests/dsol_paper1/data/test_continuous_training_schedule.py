from collections import Counter
from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.continuous_training_schedule import (
    balanced_schedule, validate_balanced_schedule, validate_preserved_episodes)


def source():
    return [dict(episode=i,initial=i//32,task=f'task-{i//64}',state_key=f'state-{i//32}',
                 angles_deg=[float(i%20-10),11.],proposal_seed=1000+i) for i in range(512)]


def test_every_batch_balanced_and_exact_original_episode_set_preserved():
    old=source();before=deepcopy(old);mixed=balanced_schedule(old,list(range(16)))
    assert old==before and mixed==balanced_schedule(old,list(range(16)))
    assert mixed!=balanced_schedule(old,list(range(16)),seed=1)
    validate_preserved_episodes(old,mixed)
    report=validate_balanced_schedule(mixed,list(range(16)))
    assert report['episodes']==512 and report['episodes_per_initial_per_update']==2
    for j in range(16):
        assert Counter(r['task'] for r in mixed[j*32:(j+1)*32])==Counter({f'task-{i}':4 for i in range(8)})


@pytest.mark.parametrize('change',['duplicate','seed','pose','state','unbalanced'])
def test_drift_and_missing_identity_rejected(change):
    old=source();mixed=balanced_schedule(old,list(range(16)))
    if change=='duplicate':mixed[0]['source_episode']=mixed[1]['source_episode']
    elif change=='seed':mixed[0]['proposal_seed']+=1
    elif change=='pose':mixed[0]['angles_deg'][0]+=.1
    elif change=='state':mixed[0]['state_key']='wrong'
    else:mixed[0]['initial']=99
    with pytest.raises(ValueError):validate_preserved_episodes(old,mixed)


def test_grouped_source_cannot_masquerade_as_mixed_and_bad_quotas_fail():
    old=source();grouped=[dict(r,source_episode=r['episode']) for r in old]
    with pytest.raises(ValueError,match='equal quotas'):validate_balanced_schedule(grouped,list(range(16)))
    with pytest.raises(ValueError):balanced_schedule(old,list(range(15)))
    with pytest.raises(ValueError):balanced_schedule(old[:-1],list(range(16)))
