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


def test_worker_load_and_simulator_timing_remain_separate(tmp_path):
    fixture(tmp_path)
    for w in range(4):
        write(tmp_path/'batches/update-000'/f'worker-{w}/status.json',dict(status='COMPLETE',version=0,
            elapsed_seconds=100+w,episode_timings=[dict(episode=i,model_prepare_seconds=2.,
                simulator_start_seconds=3.) for i in range(w,32,4)]))
    row=update_rows(tmp_path)[0]
    assert row['sampling/max_worker_loop_seconds']==103
    assert row['sampling/max_worker_model_prepare_seconds']==16
    assert row['sampling/mean_simulator_start_seconds']==3
    assert not any('batch_wall' in k or 'success' in k for k in row)


def test_corrected_regularizer_does_not_relabel_historical_entropy(tmp_path):
    from reports.paper1.continuous_training_tracking import CORRECTED_MEANS
    fixture(tmp_path)
    path=tmp_path/'updates/001.json';data=json.loads(path.read_text())
    data['means']={k:.2 for k in CORRECTED_MEANS};write(path,data)
    row=update_rows(tmp_path)[0]
    assert 'optimization/entropy' not in row
    assert row['optimization/exploration_regularizer']==.2
    assert row['optimization/stop_entropy']==.2


def test_tracking_checks_revision_when_trainer_binding_arrives(tmp_path):
    from reports.paper1.continuous_training_tracking import validate_tracking_revision
    cfg=dict(group='continuous-pose-ppo-stop-mixed-v2',training_revision='continuous_ppo_stop_mixed_v2')
    validate_tracking_revision(cfg,tmp_path)
    write(tmp_path/'binding.json',dict(exploration_rule='unit_tanh_detached_continue_v1'))
    validate_tracking_revision(cfg,tmp_path)
    with pytest.raises(ValueError):validate_tracking_revision({'group':'continuous-pose-ppo-v1'},tmp_path)
    with pytest.raises(ValueError):validate_tracking_revision({'group':cfg['group']},tmp_path)
    write(tmp_path/'binding.json',{})
    with pytest.raises(ValueError):validate_tracking_revision(cfg,tmp_path)
