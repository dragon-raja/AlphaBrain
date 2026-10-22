"""Independent W&B evaluation sidecar: only complete paired task results."""
import argparse
import json
from pathlib import Path
import time
from AlphaBrain.research.dsol.data.artifacts import read, write as write_json
from AlphaBrain.research.dsol.execution.experiment_tracking import private_settings


def paired_rows(root):
    rows=[]
    for path in sorted(Path(root).glob('paired-*.json')):
        data=read(path)
        row={'evaluation/complete_pairs':data['complete_pairs'],
             'evaluation/completed_tasks':data['completed_tasks']}
        for scope, methods in data['scopes'].items():
            for method, metrics in methods.items():
                for key, value in metrics.items():
                    if value is not None:row[f'real_task/{scope}/{method}/{key}']=value
        rows.append(row)
    return rows


def track(config, *, follow=False):
    cfg=read(config);root=Path(cfg['evaluation_output']);out=Path(cfg['tracking_output'])
    if cfg['group']!='continuous-pose-real-evaluation-v1' or cfg['mode'] not in ('offline','online'):
        raise ValueError('Separate explicit continuous real evaluation group required')
    out.mkdir(parents=True,exist_ok=False)
    import wandb
    run=wandb.init(entity=cfg['entity'],project=cfg['project'],id=cfg['run_id'],name=cfg['name'],
        group=cfg['group'],job_type='real-paired-task-evaluation',mode=cfg['mode'],resume='never',
        dir=str(out),save_code=False,settings=private_settings(wandb),config=cfg['public_config'])
    receipt=dict(status='WATCHING',run_id=cfg['run_id'],url=run.url if cfg['mode']=='online' else None,paired_blocks=0)
    run.define_metric('evaluation/complete_pairs')
    run.define_metric('real_task/*',step_metric='evaluation/complete_pairs')
    cursor=0;started=time.monotonic()
    try:
        while True:
            rows=paired_rows(root)
            for row in rows[cursor:]:run.log(row)
            cursor=len(rows);receipt['paired_blocks']=cursor
            if (root/'failure.json').exists():
                receipt['status']='FAILED_STOPPED';run.summary['evaluation_status']='FAILED_STOPPED'
                run.summary['completed_tasks']=read(root/'failure.json')['completed_tasks']
                write_json(out/'receipt.json',receipt);break
            if (root/'status.json').exists() and read(root/'status.json')['status']=='COMPLETE':
                receipt['status']='COMPLETE';run.summary['evaluation_status']='COMPLETE'
                write_json(out/'receipt.json',receipt);break
            if not follow:
                receipt['status']='OFFLINE_VALIDATED' if cfg['mode']=='offline' else 'SNAPSHOT'
                write_json(out/'receipt.json',receipt);break
            write_json(out/'receipt.json',receipt)
            if time.monotonic()-started>cfg.get('timeout_seconds',86400):raise TimeoutError('Tracking timeout')
            time.sleep(10)
    except BaseException as error:
        receipt.update(status='TRACKING_FAILED',error=type(error).__name__)
        write_json(out/'receipt.json',receipt);run.finish(exit_code=1);raise
    run.finish();return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--follow',action='store_true');a=p.parse_args()
    print(json.dumps(track(a.config,follow=a.follow),indent=2))
