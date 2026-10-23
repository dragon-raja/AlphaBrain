"""Separate model-condition evaluation dashboards, complete blocks only."""
import argparse
from pathlib import Path
import time
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.queue import _durable_write
from AlphaBrain.research.dsol.analysis.continuous_evaluation import summarize
from AlphaBrain.research.dsol.execution.experiment_tracking import private_settings
from benchmarks.LIBERO.continuous_evaluation_fleet import inventory,request_for,validate_block


def snapshot(campaign,model,cache=None):
    cfg,plans=inventory(campaign);cache={} if cache is None else cache
    tasks=[];searches=[];nominal=[];complete=[];counts=dict(score=0,execution=0,nominal=0)
    for block in plans[model]:
        root=Path(cfg['output'])/model/'blocks'/f"{block['index']:02d}"
        if not (root/'complete.json').exists():continue
        sha=file_digest(root/'complete.json')
        if block['index'] not in cache or cache[block['index']][0]!=sha:
            validation=validate_block(request_for(campaign,model,block['index']),root)
            cache[block['index']]=(sha,read(root/'results.json'),validation)
        _,payload,validation=cache[block['index']];complete.append(block['index'])
        if block['kind']=='paired':tasks+=payload['tasks'];searches+=payload['searches']
        else:nominal+=payload['tasks']
        for name,value in validation['calls'].items():
            counts['nominal' if block['kind']=='nominal' else name]+=value
    stats=summarize(tasks,searches)
    row={'evaluation/completed_tasks':len(tasks)+len(nominal),'evaluation/complete_pairs':stats['complete_pairs'],
         'evaluation/completed_blocks':len(complete),'evaluation/missing_blocks':24-len(complete)}
    for scope,methods in stats['scopes'].items():
        for method,metrics in methods.items():
            for key,value in metrics.items():
                if value is not None:row[f'real_task/{scope}/{method}/{key}']=value
    for scope in ['all']+sorted({r['task'] for r in nominal}):
        rows=nominal if scope=='all' else [r for r in nominal if r['task']==scope]
        row[f'nominal/{scope}/tasks']=len(rows)
        if rows:row[f'nominal/{scope}/success_rate']=sum(r['success'] for r in rows)/len(rows)
    row.update({f'calls/{k}':v for k,v in counts.items()})
    return dict(status='COMPLETE' if len(complete)==24 else 'PARTIAL',completed_blocks=complete,scalars=row)


def track(config,*,follow=False):
    cfg=read(config)
    if cfg['group']!='continuous-pose-model-comparison-real-v2' or cfg['mode'] not in ('offline','online'):
        raise ValueError('Explicit separate matched-model evaluation group required')
    root=Path(cfg['tracking_output']);root.mkdir(parents=True,exist_ok=False)
    import wandb
    run=wandb.init(entity=cfg['entity'],project=cfg['project'],id=cfg['run_id'],name=cfg['name'],
        group=cfg['group'],job_type='matched-real-evaluation',mode=cfg['mode'],resume='never',
        dir=str(root),save_code=False,settings=private_settings(wandb),config=cfg['public_config'])
    run.define_metric('evaluation/completed_tasks')
    for name in ('real_task/*','nominal/*','calls/*'):run.define_metric(name,step_metric='evaluation/completed_tasks')
    cache={};previous=None;started=time.monotonic()
    try:
        while True:
            value=snapshot(cfg['campaign'],cfg['condition'],cache)
            if value['status']=='COMPLETE':value=snapshot(cfg['campaign'],cfg['condition'])
            if value['completed_blocks']!=previous:
                # Empty output is a readiness check, never a zero-success result.
                if value['completed_blocks']:run.log(value['scalars'])
                previous=value['completed_blocks']
            campaign=read(cfg['campaign']);failed=bool(list(Path(campaign['control']).glob('failure-*.json')))
            status='FAILED_STOPPED' if failed else value['status']
            _durable_write(root/'receipt.json',dict(status=status,url=run.url if cfg['mode']=='online' else None,
                run_id=cfg['run_id'],completed_blocks=previous,mode=cfg['mode']))
            if status in ('COMPLETE','FAILED_STOPPED') or not follow:break
            if time.monotonic()-started>cfg.get('timeout_seconds',86400):raise TimeoutError('Tracking deadline')
            time.sleep(10)
        run.summary['evaluation_status']=status
    except BaseException:
        run.finish(exit_code=1);raise
    run.finish();return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--follow',action='store_true')
    a=p.parse_args();track(a.config,follow=a.follow)
