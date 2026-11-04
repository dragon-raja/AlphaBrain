"""Complete cold-start G report; immutable historical B is matched explicitly."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,datetime
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.data import reward_pilot
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
from AlphaBrain.research.dsol.analysis.recovery_summary import paired_summary,paired_method_contrast


def build(root,output,*,config_path,historical_root):
    root,output,old=map(Path,(root,output,historical_root));cfg=read(config_path)
    if output.exists():raise ValueError('New output required')
    state=read(root/'current.json')
    if state['status']!='ALL_SCIENTIFIC_WORK_COMPLETE' or state['dispatch_enabled']:raise ValueError('Full completion required')
    protocol=reward_pilot.load_protocol(cfg['protocol'],cfg['input_hashes'][cfg['protocol']])
    if protocol['initial_update']!=0:raise ValueError('Cold-start report only')
    evidence={};rows=[];searches=[];cache={};calls=0
    def checked(path,expected):
        if file_digest(path)!=expected:raise ValueError('Changed artifact: '+str(path))
        evidence[str(path)]=expected;return read(path)
    refs=checked(cfg['baseline_references'],cfg['input_hashes'][cfg['baseline_references']])['views']
    def baseline(state,view,repeat):
        key=(state,view,repeat)
        if key not in cache:
            ref=refs[f'{state}::{view}'];path=ref['episode_results'][repeat]
            cache[key]=checked(path,ref['artifacts'][path])
            if (cache[key]['state_key'],cache[key]['view_id'],cache[key]['repeat'])!=key:raise ValueError('Baseline identity changed')
        return cache[key]
    def results(directory):
        completion=read(directory/'completion.json')
        if completion['status']!='COMPLETE' or completion['dispatch_enabled']:raise ValueError('Unsettled phase')
        for spec in read(directory/'binding.json')['tasks'].values():
            out=Path(spec['output']);receipt=read(out/'complete.json')
            if receipt['status']!='COMPLETE' or receipt['request_sha256']!=digest(spec['request']):raise ValueError('Unbound result')
            result=checked(out/'results.json',receipt['artifacts']['results.json'])
            expected={(r['state_key'],r['view_id'],r['method'],rep) for r in spec['request']['block']['rows'] for rep in (0,1)}
            actual=[(t['state_key'],t['view_id'],t['method'],t['repeat']) for t in result['tasks']]
            if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Operation coverage differs')
            yield result
    allocation=read(root/'evaluation/allocation.json')
    for phase in allocation['phases']:
        checked(phase['path'],phase['sha256']);folder=root/'execution'/(phase['phase']+'-real')
        calls+=read(folder/'completion.json')['actual_model_calls']
        for result in results(folder):
            searches.extend(result['searches'])
            for t in result['tasks']:
                stay=baseline(t['state_key'],t['view_id'],t['repeat']);nominal=baseline(t['state_key'],0,t['repeat'])
                n=min(t['replans'],stay['replans'])
                if t['execution_noise_sha256'][:n]!=stay['execution_noise_sha256'][:n]:raise ValueError('Noise mismatch')
                rows.append(dict(t,stay_success=stay['success'],nominal_success=nominal['success']))
    if (len(rows)!=19200 or len(searches)!=9600 or calls!=state['actual_model_calls']['evaluation']
            or len({(r['method'],r['state_key'],r['view_id']) for r in searches})!=9600):raise ValueError('Full G coverage differs')
    expected=reward_pilot.evaluation_rows(protocol,read(cfg['original_recovery_schedule']))
    if {(r['method'],r['state_key'],r['view_id']) for r in searches}!={(r['method'],r['state_key'],r['view_id']) for r in expected}:raise ValueError('Unexpected G panel')
    historical=[]
    for phase in read(old/'protocol/recovery/allocation.json')['phases']:
        for result in results(old/'execution-recovery'/(phase['phase']+'-real')):
            historical.extend(t for t in result['tasks'] if t['method'].startswith('B-seed-'))
    per_method={};aggregates={};contrasts={}
    for split,count in [('seen_new_initials',3200),('validation',1280),('unseen_test',1920)]:
        rates=[]
        for seed in (0,1,2):
            name=f'G-seed-{seed}';selected=[t for t in rows if t['split']==split and t['method']==name]
            if len(selected)!=count:raise ValueError('Split coverage differs')
            metrics=paired_summary(selected);sr=[s for s in searches if s['split']==split and s['method']==name]
            metrics['mean_queries']=float(np.mean([s['queries'] for s in sr]));per_method[split+'/'+name]=metrics;rates.append(metrics['task_macro']['success'])
        aggregates[split]=dict(seed_success_rates=rates,mean_success=float(np.mean(rates)),seed_std=float(np.std(rates,ddof=1)))
        seeds=(0,1,2) if split=='unseen_test' else (0,)
        g=[t for t in rows if t['split']==split and int(t['method'][-1]) in seeds]
        b=[t for t in historical if t['split']==split and int(t['method'][-1]) in seeds]
        contrasts[split]=paired_method_contrast(g,b)
    training={}
    for seed in (0,1,2):
        folder=root/f'training/runs/G-seed-{seed}';episodes=[read(p)['metrics'] for p in sorted((folder/'episodes').glob('*.json'))]
        updates=[read(folder/f'updates/{i:03d}.json') for i in range(1,65)]
        if len(episodes)!=2048 or updates[-1]['completed_episodes']!=2048:raise ValueError('Training incomplete')
        training[str(seed)]=dict(episodes=2048,updates=64,calls=2*sum(t['queries'] for t in episodes),scale=[u['reward_settlement']['scale'] for u in updates])
    if sum(x['calls'] for x in training.values())!=state['actual_model_calls']['training']:raise ValueError('Training accounting differs; explicit interruption supplement required')
    summary=dict(status='COMPLETE_CPU_REPORT',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),searches=9600,episodes=19200,
        per_method=per_method,aggregates=aggregates,matched_G_minus_B=contrasts,training=training,calls=state['actual_model_calls'],
        limits=['B three-seed comparison only on12 unseen tasks; seen/development compare seed0, no missing seed imputation.','Existing held-out task results previously inspected; locked confirmation, not new blind test.','Shared initial weights, independent sampling/optimizer seeds.','Online reward scale does not remove existing observation normalization or canonical-coordinate prior.','Two-member diagnostics are not confidence guarantees.'])
    output.mkdir(parents=True);write(output/'summary.json',summary);write(output/'input-hashes.json',evidence)
    fields=['method','split','task','state_key','view_id','repeat','success','stay_success','nominal_success','queries','selected_angles']
    with (output/'paired-outcomes.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write(output/'search-diagnostics.json',searches)
    lines=['# 原生G完整从零训练与恢复评测','', '全部9600搜索、19200操作轨迹；三个种子各2048训练回合。操作模型冻结。','', '| 数据划分 | G三种子平均成功率 | 严格配对G−B（百分点） | B对照种子 |','|---|---:|---:|---|']
    for split,a in aggregates.items():
        contrast=contrasts[split];lines.append(f"| {split} | {a['mean_success']:.2%} | {contrast['difference_pp']:+.2f} | {contrast['policy_seeds']} |")
    lines+=['','G三种子均值与只含seed0的G−B列不是同一个统计量；逐种子、扰动保持/规范参考、挽救/损害和探索性区间见summary.json。','',*summary['limits'],'',f"实际调用：{summary['calls']}；独立预校准调用为0。旧B预校准1600次是历史成本，不当作本轮实际发生。"]
    (output/'README.md').write_text('\n'.join(lines)+'\n');write(output/'completion.json',dict(status='COMPLETE_CPU_REPORT',summary_sha256=file_digest(output/'summary.json')))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--historical-root',required=True)
    a=p.parse_args();build(a.run,a.output,config_path=a.config,historical_root=a.historical_root)
