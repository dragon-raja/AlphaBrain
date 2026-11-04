"""Read-only report of bound completed batches and paired operation endpoints."""
from pathlib import Path
import argparse,json
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha
from AlphaBrain.research.dsol.analysis.prefix_comparison import compare_endpoint,learning_batch
from AlphaBrain.research.dsol.plotting.prefix_comparison import learning_curves


def build(run,output,*,allow_partial=False):
    run=Path(run);output=Path(output);output.mkdir(parents=True,exist_ok=False);p=read(run/'protocol-r2.json');curves={};endpoints={};provenance={}
    overrides=read(run/'endpoint-bindings.json')['overrides'] if (run/'endpoint-bindings.json').exists() else {}
    for name,cell in p['cells'].items():
        root=run/'training-r2'/name;points=[]
        for path in sorted(x for x in (root/'updates').glob('*.json') if x.stem.isdigit()):
            u=read(path);n=u['update'];records=[read(root/'episodes'/f'{i:04d}.json') for i in range(32*(n-1),32*n)]
            if any(sha(root/r['path'])!=r['sha256'] for r in records):raise ValueError('Changed stored training episode')
            points.append(learning_batch(u,[r['metrics'] for r in records]));provenance[str(path)]=sha(path)
        curves[name]=points
    for family,full in [('pi05',10),('groot_n15',4),('rldx',4)]:
        arms={};missing=[]
        for arm,prefix,phase in [('stay',3,'stay'),('p3',3,'paired-evaluation'),('full',full,'paired-evaluation')]:
            cell=family+'-p'+str(prefix);branch='execution-rldx-r3' if family=='rldx' else 'execution-r2';root=run/branch/cell/'execution'/(phase+'-real')
            override=overrides.get(family+':'+arm)
            if override:
                if sha(override['campaign'])!=override['campaign_sha256'] or sha(override['handoff'])!=override['handoff_sha256']:
                    raise ValueError('Changed endpoint owner handoff')
                root=Path(override['phase_directory'])
            if not (root/'completion.json').exists():missing.append(arm);continue
            proof=read(root/'completion.json')
            if proof['status']!='COMPLETE' or proof['dispatch_enabled'] is not False:raise ValueError('Unsettled endpoint')
            campaign_dir='campaigns-rldx-r3' if family=='rldx' else 'campaigns-r2';campaign=Path(override['campaign']) if override else run/campaign_dir/cell/(phase+'.json');c=read(campaign);rows=[]
            if c.get('stay',False)!=(arm=='stay') or c['cell']!=cell:raise ValueError('Wrong endpoint method/owner')
            from benchmarks.LIBERO.prefix_pilot import request_for,validate_block,validate_campaign
            validate_campaign(campaign)
            for b in c['blocks']:
                folder=root/'results'/f"block-{b['index']:04d}";validate_block(request_for(campaign,b['index']),folder)
                path=folder/'results.json';rows.extend(read(path)['tasks']);provenance[str(path)]=sha(path)
            arms[arm]=rows
        if missing:
            if not allow_partial:raise ValueError('Missing endpoint '+family+': '+','.join(missing))
            endpoints[family]=dict(status='PENDING_COMPLETE_PAIRED_ENDPOINT',missing=missing);continue
        expected={(r['state_key'],r['start'],rep) for r in read(p['cells'][family+'-p3']['evaluation_schedule']) for rep in (0,1)}
        endpoints[family]=dict(status='COMPLETE',**compare_endpoint(arms,expected=expected))
    write(output/'learning-curves.json',curves);write(output/'endpoints.json',endpoints);write(output/'provenance.json',provenance);learning_curves(curves,output/'learning-curves')
    lines=['# P3 versus full native prefix: bounded development pilot','',
      'One seed per arm, 256 episodes / 8 updates. Original G reward, frozen operation models. Complete endpoint: 8 tasks × 2 initials × 2 starts × 2 repeats. Task bootstrap intervals are exploratory; this is not convergence or final held-out generalization evidence. Reward scales differ by prefix and are not operation-success metrics.','',
      '| Model | Train P3/full | Stay | P3-PPO | Full-PPO | Full − Stay |','|---|---:|---:|---:|---:|---:|']
    for family,full in [('pi05',10),('groot_n15',4),('rldx',4)]:
        trained='/'.join(str(curves[family+'-p'+str(k)][-1]['episodes']) if curves[family+'-p'+str(k)] else '0' for k in (3,full));e=endpoints[family]
        if e['status']!='COMPLETE':lines.append(f'| {family} P3/P{full} | {trained} | pending | pending | pending | pending |');continue
        rates=[e['rates'][a] for a in ['stay','p3','full']];rates=[f"{v['successes']}/64 ({100*v['success_rate']:.2f}%)" for v in rates]
        contrast=e['contrasts']['full-minus-stay'];ci=contrast['task_bootstrap_95'];delta=f"{100*contrast['delta']:+.2f} pp [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]"
        lines.append('| '+family+f' P3/P{full} | '+trained+' | '+' | '.join(rates)+' | '+delta+' |')
    lines+=['','All failures and known interface-call costs remain recorded in the run budget. CPU/GPU engineering smoke episodes are excluded from scientific endpoints. No prefix is promoted automatically.','', '[Learning curves](learning-curves.pdf)']
    (output/'README.md').write_text('\n'.join(lines)+'\n');return endpoints


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();print(json.dumps(build(a.run,a.output,allow_partial=a.allow_partial)))
if __name__=='__main__':main()
