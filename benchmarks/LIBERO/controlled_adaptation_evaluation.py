"""Frozen few-shot endpoint panel assembly and paired descriptive summaries."""
from pathlib import Path
from collections import defaultdict
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest
from .controlled_target_recovery import limits,SCHEMA,row_key
from .controlled_campaign import run_stage


def endpoint_actors(training):
    root=Path(training);result={}
    for version in (0,1,4,8):
        p=read(root/('initial-publications.json' if version==0 else f'endpoints/update-{version:02d}.json'))
        for run,spec in p.items():
            if version==0 and not run.startswith('source-first_batch_fixed'):continue
            if version==1 and '-window-' in run:continue
            result[f'actor:{run}:u{version}']=dict(path=spec['policy'],version=version)
    if len(result)!=33:raise ValueError('Exact 33 predeclared policy endpoints required')
    if read(root/'first-batch-gate.json')['status']!='PASS':raise ValueError('Cannot reuse first endpoint without exact equality')
    return result


def campaign(common,rows,actors,*,block_size=8,hosts=('fresh',)):
    from .controlled_target_evaluation import execution_hosts
    hosts=execution_hosts(dict(execution_hosts=list(hosts)))
    if type(block_size) is not int or not 1<=block_size<=8:
        raise ValueError('Bounded short evaluation blocks required')
    c=dict(common,schema=SCHEMA,actors=actors,blocks=[])
    c['execution_hosts']=hosts
    c['input_hashes']=dict(common['input_hashes'])
    for spec in actors.values():c['input_hashes'][spec['path']]=file_digest(spec['path'])
    contract=read(common['sim_config'])['operation_contract']
    for i in range(0,len(rows),block_size):
        group=rows[i:i+block_size]
        c['blocks'].append(dict(index=len(c['blocks']),host=hosts[len(c['blocks'])%len(hosts)],rows=group,limits=limits(group,contract)))
    c['limits']=limits(rows,contract)
    return c


def run(config,*,training):
    root=Path(config['output']);common=read(config['evaluation_common']);rows=read(common['schedule'])
    actors=endpoint_actors(training);results=[]
    for phase in ('baseline','endpoints'):
        selected=[r for r in rows if r['method'].startswith('actor:') == (phase=='endpoints')]
        c=campaign(common,selected,actors if phase=='endpoints' else {},
            block_size=config.get('evaluation_block_size',8),hosts=config.get('execution_hosts',['fresh']))
        cp=root/f'{phase}-campaign.json';write(cp,c)
        reservation=root/'reservations'/f'{phase}.json'
        if reservation.exists():raise ValueError('Existing evaluation reservation; no automatic replay')
        write(reservation,dict(campaign=str(cp),sha256=file_digest(cp),limits=c['limits'],status='RESERVED'))
        run_stage(config,phase,cp,kind='target_recovery',cpu=True)
        result=run_stage(config,phase,cp,kind='target_recovery',cpu=False);results.append(result)
        write(root/'settlements'/f'{phase}.json',result)
        summarize(root)
    write(root/'current.json',dict(status='COMPLETE',episodes=sum(x['actual_episodes'] for x in results),
        actual_model_calls=sum(x['actual_model_calls'] for x in results),dispatch_enabled=False,
        scope='FastWAM development only; not physical-robot evidence'))
    return results


def summarize(root):
    """Report all completed conditions; partial coverage is never called final."""
    root=Path(root);tasks=[]
    for phase in ('baseline','endpoints'):
        for path in sorted((root/f'execution/{phase}-real/results').glob('block-*/results.json')):
            # Only settled complete blocks enter descriptive progress.
            if not (path.parent/'complete.json').exists():continue
            tasks.extend(read(path)['tasks'])
    by_method=defaultdict(list)
    for row in tasks:by_method[row['method']].append(row)
    key=lambda r:(r['state_key'],r['start'],r['repeat'])
    stays={key(r):r['success'] for r in by_method['stay']};summary=[]
    for method,rows in sorted(by_method.items()):
        paired=[r for r in rows if key(r) in stays];n=len(rows)
        summary.append(dict(method=method,episodes=n,successes=sum(r['success'] for r in rows),
            success_rate=sum(r['success'] for r in rows)/n if n else None,
            paired_with_stay=len(paired),rescued=sum(r['success'] and not stays[key(r)] for r in paired),
            damaged=sum(not r['success'] and stays[key(r)] for r in paired),
            mean_queries=sum(r['queries'] for r in rows)/n if n else None))
    result=dict(status='COMPLETE' if len(tasks)==2336 else 'PARTIAL',episodes=len(tasks),planned=2336,
        results=summary,scope='8 development tasks x 2 initial states; all predeclared rows retained; not a real-robot result')
    write(root/'summary.json',result);return result
