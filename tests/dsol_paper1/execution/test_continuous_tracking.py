import json
from pathlib import Path
import pytest
from reports.paper1.continuous_training_tracking import update_rows,MEANS


def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def fixture(root):
    write(root/'latest.json',{'update':1})
    write(root/'updates/001.json',dict(update=1,completed_episodes=32,queries=64,scoring_trajectories=128,means={k:.1 for k in MEANS}))
    for i in range(32):
        write(root/'episodes'/f'{i:04d}.json',dict(metrics=dict(episode=i,policy_update=0,queries=2,decisions=1,
            initial_accel=2.,best_accel=1.,total_reward_including_initial=.98,reason='stop',private_path='MUST_NOT_EXPORT')))


def test_scalars_separate_axes_and_no_private_fields(tmp_path):
    fixture(tmp_path);rows=update_rows(tmp_path);r=rows[0]
    assert r['episodes']==32 and r['queries']==64 and r['decision_steps']==32
    assert r['search/mean_accel_improvement']==1 and r['search/stop_fraction']==1
    assert 'private_path' not in str(rows) and 'success_rate' not in str(rows)
    write(tmp_path/'inflight.json',dict(kind='update',index=1))
    assert update_rows(tmp_path)==[]


def test_invalid_or_nonfinite_log_rejected(tmp_path):
    fixture(tmp_path)
    p=tmp_path/'updates/001.json';d=json.loads(p.read_text());d['means']['entropy']=float('nan');write(p,d)
    with pytest.raises(ValueError,match='Nonfinite'):update_rows(tmp_path)


def test_missing_training_output_is_not_failure(tmp_path):
    assert update_rows(tmp_path/'not-started')==[]
