"""Independent W&B sidecar for committed continuous PPO updates, no model imports."""
import argparse
import json
import math
from pathlib import Path
import time
from AlphaBrain.research.dsol.execution.experiment_tracking import digest, private_settings

MEANS=('loss','gradient_norm','policy_loss','value_loss','entropy','approximate_kl','clip_fraction')
CORRECTED_MEANS=tuple(k for k in MEANS if k!='entropy')+(
    'exploration_regularizer','stop_entropy','motion_unit_entropy','mean_stop_probability')
EPISODE_FIELDS=('queries','decisions','initial_accel','best_accel','total_reward_including_initial')


def read(path):return json.loads(Path(path).read_text())


def validate_tracking_revision(cfg, root):
    corrected=cfg['group']=='continuous-pose-ppo-stop-mixed-v2'
    if corrected and cfg.get('training_revision')!='continuous_ppo_stop_mixed_v2':
        raise ValueError('Corrected tracking requires an explicit training revision')
    path=Path(root)/'binding.json'
    # The sidecar may start before the trainer; verify again when binding arrives.
    if path.exists():
        bound=read(path).get('exploration_rule')=='unit_tanh_detached_continue_v1'
        if bound!=corrected:raise ValueError('Training revision and W&B group differ')


def update_rows(root):
    """Only optimizer-committed batches; never infer a task-success metric."""
    root=Path(root)
    if not (root/'latest.json').exists():return []
    latest=read(root/'latest.json')['update']
    if (root/'inflight.json').exists():
        op=read(root/'inflight.json')
        if op['kind']=='update':latest=min(latest,op['index']-1)
    rows=[];decisions=0
    for index in range(1,latest+1):
        update=read(root/'updates'/f'{index:03d}.json')
        episodes=[read(root/'episodes'/f'{i:04d}.json')['metrics'] for i in range((index-1)*32,index*32)]
        if update['completed_episodes']!=index*32 or update['update']!=index:
            raise ValueError('Unexpected continuous update identity')
        if any(e['episode']!=i or e['policy_update']!=index-1 for i,e in enumerate(episodes,(index-1)*32)):
            raise ValueError('Episode/update identity mismatch')
        decisions+=sum(e['decisions'] for e in episodes)
        row=dict(episodes=index*32,updates=index,decision_steps=decisions,
                 queries=update['queries'],denoising_trajectories=update['scoring_trajectories'])
        if row['denoising_trajectories']!=2*row['queries']:raise ValueError('Query accounting mismatch')
        keys=CORRECTED_MEANS if 'exploration_regularizer' in update['means'] else MEANS
        row.update({'optimization/'+k:update['means'][k] for k in keys})
        row.update({'search/mean_'+k:sum(e[k] for e in episodes)/32 for k in EPISODE_FIELDS})
        row['search/mean_accel_improvement']=sum(e['initial_accel']-e['best_accel'] for e in episodes)/32
        row['search/stop_fraction']=sum(e['reason']=='stop' for e in episodes)/32
        row['search/budget_fraction']=sum(e['reason']=='budget' for e in episodes)/32
        if all('elapsed_seconds' in e for e in episodes):
            row['search/mean_episode_seconds']=sum(e['elapsed_seconds'] for e in episodes)/32
        if 'elapsed_seconds' in update:row['optimization/elapsed_seconds']=update['elapsed_seconds']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite tracking scalar')
        rows.append(row)
    return rows


def track(config, *, follow=False):
    cfg=read(config);root=Path(cfg['training_output']);output=Path(cfg['tracking_output'])
    if cfg['mode'] not in ('offline','online') or cfg['group'] not in (
            'continuous-pose-ppo-v1','continuous-pose-ppo-stop-mixed-v2'):
        raise ValueError('Explicit mode and separate continuous group required')
    validate_tracking_revision(cfg,root)
    output.mkdir(parents=True,exist_ok=False)
    import wandb
    run=wandb.init(entity=cfg['entity'],project=cfg['project'],id=cfg['run_id'],
        name=cfg['name'],group=cfg['group'],job_type='continuous-search-training',
        mode=cfg['mode'],resume='never',dir=str(output),save_code=False,settings=private_settings(wandb),
        config=cfg['public_config'])
    receipt=dict(run_id=cfg['run_id'],mode=cfg['mode'],url=run.url if cfg['mode']=='online' else None,
                 status='WATCHING',updates=0,config_sha256=digest(cfg))
    def save():
        temporary=output/'receipt.json.tmp';temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(output/'receipt.json')
    try:
        run.define_metric('episodes');run.define_metric('optimization/*',step_metric='episodes')
        run.define_metric('search/*',step_metric='episodes')
        for key in ('updates','decision_steps','queries','denoising_trajectories'):run.define_metric(key,step_metric='episodes')
        save();cursor=0;started=time.monotonic()
        while True:
            validate_tracking_revision(cfg,root)
            rows=update_rows(root)
            for row in rows[cursor:]:run.log(row)
            cursor=len(rows);receipt['updates']=cursor;save()
            failure=root/'failure.json';status=root/'status.json'
            if failure.exists():
                ledger=root/'model-attempts.jsonl'
                run.summary['attempted_trajectories']=len(ledger.read_text().splitlines()) if ledger.exists() else 0
                run.summary['completed_episodes']=len(list((root/'episodes').glob('*.json')))
                run.summary['training_status']='FAILED_STOPPED';receipt['status']='TRAINING_FAILED_STOPPED';break
            if status.exists() and read(status)['status']=='COMPLETE':
                run.summary['training_status']='COMPLETE';receipt['status']='COMPLETE';break
            if not follow:
                receipt['status']='OFFLINE_VALIDATED' if cfg['mode']=='offline' else 'SNAPSHOT_PUBLISHED';break
            if time.monotonic()-started>cfg.get('timeout_seconds',86400):raise TimeoutError('Sidecar deadline; trainer unchanged')
            time.sleep(cfg.get('poll_seconds',10))
        run.summary['recorded_updates']=cursor;save()
    except BaseException as error:
        receipt.update(status='TRACKING_FAILED',error=type(error).__name__);save();run.finish(exit_code=1);raise
    run.finish();return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--follow',action='store_true')
    a=p.parse_args();print(json.dumps(track(a.config,follow=a.follow),indent=2))
